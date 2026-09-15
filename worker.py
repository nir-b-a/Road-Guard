"""
worker.py -- the GPU side of the Road Guard processing architecture.

It is the bridge between the upload server and the CV pipeline. The loop:

    1. GET  {SERVER_URL}/api/internal/next-job   (atomically claims a queued drive)
    2. Download that drive's 7 files from Cloudflare R2 to a local temp dir
    3. Run the existing pipeline   (main.process_video_with_models)
    4. Upload ONE evidence clip PER VIOLATION to R2 (<sessionId>/out/<violation_id>.mp4), plus its
       plate picture when the plate was found (<sessionId>/out/<violation_id>_plate.png)
    5. POST one /api/internal/violation per detected violation
    6. POST /api/internal/drive/<id>/complete  (processed | failed)

Steps 1-3 run on the main thread; steps 4-6 (the "ship" tail -- x264 transcode, the R2 PUTs,
the violation POSTs) are handed to a background thread, so the worker goes straight back to
claiming the next drive instead of idling through an encode that needs neither the GPU nor
the models. The tail owns its job dir and is the only thing that marks a drive complete, so
a drive is never reported processed while its clips are still uploading. SHIP_ASYNC=0 puts
it all back on one thread.

The big video bytes go R2 <-> worker directly; only small JSON crosses to the server.
The YOLO / lane / tire / plate models are loaded ONCE and reused for every job (the 100 MB
weights are not reloaded per clip); only the per-vehicle evidence buffer is rebuilt per clip,
because track ids restart at 1 for every video.

Everything that is not transport (loading the models, unpacking the pipeline's export bundle,
building the violation records) lives in worker_common.py, shared with worker_offline.py.

After every drive it prints a per-stage timing report (run_timing.py): which model spent how
long on that video, plus the download/upload and render costs. This worker polls forever, so
the report is per drive rather than at exit; worker_offline.py prints the same tables plus a
grand total at the end of its run.

Config (env vars, see backend/.env.example for the shared R2_* names):
    SERVER_URL        backend base URL                 (default http://localhost:5000)
    INTERNAL_TOKEN    shared secret -> x-internal-token header (optional)
    WORKER_ID         identifies this worker in logs/claims    (default <host name>-<6 random hex>)
    POLL_SECONDS      idle poll interval                       (default 5)
    WORK_DIR          scratch dir for downloads/outputs        (default ./_worker_jobs)
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_ENDPOINT
    FFMPEG_BIN        ffmpeg used to encode evidence clips          (default "ffmpeg" on PATH)
    FFPROBE_BIN       ffprobe used to skip re-encoding H.264 clips (default "ffprobe" on PATH)
    H264_CRF          x264 quality (lower = better/bigger; 30 ~ dashcam)            (default 30)
    H264_PRESET       x264 speed/efficiency preset                              (default slow)
    H264_GOP          keyframe interval in frames                                 (default 30)
    SHIP_ASYNC        transcode+upload on a background thread (0 = inline)          (default 1)
    MAX_PENDING_SHIPS drives allowed to queue for shipping before the loop waits    (default 2)
    STOP_FILE         stop-request file the worker watches        (default <WORK_DIR>/worker.stop)

Run:  python worker.py
      python worker.py --keep-annotated   # also keep each drive's annotated video, in
                                          # <WORK_DIR>/annotated/<sessionId>_annotated.mp4
                                          # (--keep-anotated works too)
      python worker_offline.py <folder>   # the same processing with no server and no R2

Stop (from another terminal; see worker_stop.py):
      python worker.py --stop             # finish the current drive, then exit
      python worker.py --stop-now         # hand the current drive back to the queue, then exit
"""
import os
import sys
import time
import shutil
import secrets
import socket
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import boto3
import requests


def _load_env_file(path):
    """Minimal .env reader (no python-dotenv dependency). Sets only keys not already in
    the environment, so a real shell export still wins. Lets the worker reuse the R2
    credentials already configured in backend/.env."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


# Reuse backend/.env (R2_* + INTERNAL_TOKEN) so creds live in one place; a worker.env
# next to this file can override / add worker-only vars (SERVER_URL, WORKER_ID, ...).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                      # so `import main` works from any cwd
    sys.path.insert(0, _HERE)
_load_env_file(os.path.join(_HERE, "worker.env"))
_load_env_file(os.path.join(_HERE, "backend", ".env"))

import worker_common as wc                     # noqa: E402  (after sys.path is set)
import run_timing                              # noqa: E402  per-stage stopwatch (per-job report)
import worker_stop                             # noqa: E402  --stop / --stop-now / signals

SERVER_URL     = os.environ.get("SERVER_URL", "http://localhost:5000").rstrip("/")
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")
# The host name alone is shared by every worker on one machine, so two workers started from two
# terminals would claim drives under the same id -- and the server's release check (only the
# worker holding a claim may hand it back) could not tell them apart. A random suffix, not the
# start time, because two workers launched in the same second must still differ. A new id per
# run is harmless: a worker only ever releases drives it claimed in this process.
WORKER_ID      = os.environ.get("WORKER_ID") or f"{socket.gethostname()}-{secrets.token_hex(3)}"
POLL_SECONDS   = float(os.environ.get("POLL_SECONDS", "5"))
WORK_DIR       = os.environ.get("WORK_DIR", os.path.join(_HERE, "_worker_jobs"))
STOP_FILE      = os.environ.get("STOP_FILE", os.path.join(WORK_DIR, "worker.stop"))

# The pipeline renders a full annotated video for every drive, but it lives in the job dir,
# which ship() deletes. --keep-annotated moves it out first, into ANNOTATED_DIR.
# --keep-anotated (one "n") is accepted too: flags are matched exactly and nothing rejects an
# unknown one, so that typo used to be silently ignored and the video deleted.
KEEP_ANNOTATED = any(flag in sys.argv for flag in ("--keep-annotated", "--keep-anotated"))
ANNOTATED_DIR  = os.path.join(WORK_DIR, "annotated")

R2_BUCKET   = os.environ.get("R2_BUCKET")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")

# Evidence clips now arrive ALREADY H.264: violations/clip_encoder.py pipes the annotated
# frames straight into one libx264 encode, so the old mp4v-then-transcode double hop is gone.
# The transcode below survives only as the safety net for a renderer that had no ffmpeg and
# fell back to mp4v -- which no browser decodes in a <video> element, so an un-transcoded clip
# reaches the dashboard as a black rectangle with working controls and no error.
# The x264 flags come from clip_encoder (imported lazily, inside transcode_h264) so the two
# encode paths cannot drift apart -- and so worker.py still STARTS if that import is broken.
FFMPEG_BIN  = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")

# Shipping a drive's evidence (x264 transcode + the R2 PUTs) costs minutes and needs neither
# the GPU nor the models -- it is ffmpeg subprocesses and network I/O, both of which release
# the GIL. Running it on a background thread lets this worker claim and start the NEXT drive
# immediately instead of idling through the encode. SHIP_ASYNC=0 restores the old fully
# synchronous behaviour (one drive completely finished before the next is claimed).
SHIP_ASYNC        = os.environ.get("SHIP_ASYNC", "1").lower() not in ("0", "false", "no")
MAX_PENDING_SHIPS = int(os.environ.get("MAX_PENDING_SHIPS", "2"))

# One ship thread: the tails serialise against each other (they contend for the same CPU and
# the same uplink anyway) but overlap with the pipeline, which is the whole point.
_SHIP_POOL = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="ship")
              if SHIP_ASYNC else None)
_SHIP_SLOTS = threading.BoundedSemaphore(max(1, MAX_PENDING_SHIPS))

# Drive.files keys -> the on-disk filename the pipeline expects beside the video.
SENSOR_KEYS = {"frames": "frames.csv", "gps": "gps.csv", "gravity": "gravity.csv",
               "gyro": "gyro.csv", "linacc": "linacc.csv", "intrinsics": "intrinsics.json"}


def _headers():
    h = {"x-worker-id": WORKER_ID}
    if INTERNAL_TOKEN:
        h["x-internal-token"] = INTERNAL_TOKEN
    return h


def make_s3():
    missing = [v for v in ("R2_BUCKET", "R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
               if not os.environ.get(v)]
    if missing:
        sys.exit("[worker] missing R2 env vars: " + ", ".join(missing))
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def claim_job():
    """Return the claimed drive dict, or None if the queue is empty / server is down."""
    try:
        r = requests.get(SERVER_URL + "/api/internal/next-job", headers=_headers(), timeout=30)
        r.raise_for_status()
        return r.json().get("data")
    except requests.RequestException as e:
        print(f"[worker] next-job poll failed: {e}")
        return None


def report_violation(payload):
    """POST one violation. `payload` comes from worker_common.build_violation_payloads, which
    carries the backend's fields plus pipeline detail; only the fields the API accepts are sent."""
    body = {k: payload[k] for k in ("driveId", "videoClipPath", "carId", "calculatedSpeed",
                                    "lat", "lon", "violationType", "plateImagePath")}
    r = requests.post(SERVER_URL + "/api/internal/violation", json=body, headers=_headers(), timeout=30)
    if not r.ok:
        print(f"[worker] violation POST failed ({r.status_code}): {r.text}")


def complete(drive_id, status, error=None):
    body = {"status": status}
    if error:
        body["error"] = error[:500]
    try:
        requests.post(SERVER_URL + f"/api/internal/drive/{drive_id}/complete",
                      json=body, headers=_headers(), timeout=30)
    except requests.RequestException as e:
        print(f"[worker] complete POST failed: {e}")


def release(drive_id, reason):
    """Hand a claimed but unfinished drive straight back to the server's queue, instead of leaving
    it 'processing' until the server's stale-job timeout (JOB_STALE_MINUTES) requeues it. Best
    effort: if this call fails, that timeout is still the safety net."""
    try:
        r = requests.post(SERVER_URL + f"/api/internal/drive/{drive_id}/release",
                          json={"reason": reason}, headers=_headers(), timeout=15)
    except requests.RequestException as e:
        print(f"[worker] release POST failed ({e}) -- the server requeues drive {drive_id} "
              f"after its stale-job timeout")
        return
    if r.ok:
        print(f"[worker] drive {drive_id} handed back to the server's queue")
    else:
        print(f"[worker] release of drive {drive_id} refused ({r.status_code}): {r.text[:300]}")


def clip_first_seconds(src_path, dst_path, seconds=5):
    """FALLBACK evidence only: write the first `seconds` of `src_path` to `dst_path`. Used when a
    per-violation clip could not be rendered, so a violation is still reported with SOMETHING a
    reviewer can open (the backend requires a non-null videoClipPath). Returns the frame count
    written. Goes through the same single-pass H.264 writer as the real clips, which itself falls
    back to mp4v if ffmpeg is missing -- so this never needs ffmpeg on PATH either."""
    import cv2

    from violations.clip_encoder import open_clip_writer
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video to clip: {src_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = max(1, int(round(fps * seconds)))
    writer = open_clip_writer(dst_path, fps, (width, height))
    written = 0
    try:
        while written < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()
    if written == 0:
        raise RuntimeError(f"no frames read from {src_path}")
    return written


def video_codec(path):
    """The container's video codec name via ffprobe (e.g. "h264"), or None if unknowable."""
    try:
        proc = subprocess.run(
            [FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return (proc.stdout or "").strip().lower() or None


def transcode_h264(src_path, dst_path):
    """Re-encode `src_path` to browser-playable H.264 at `dst_path`. -> dst_path, or None.

    SAFETY NET ONLY. Clips are now written as H.264 in a single pass by
    violations/clip_encoder.py, so the common case returns None immediately and the already-
    H.264 file is uploaded untouched -- no second lossy generation, no second encode. This
    still fires for a clip that fell back to mp4v because the renderer had no ffmpeg.

    Returning None (rather than raising) is deliberate: a missing ffmpeg or a failed encode
    must not lose a real detection. The caller uploads the original instead, which a reviewer
    can still download and open in VLC -- degraded, but never dropped.

    -movflags +faststart puts the moov atom first so the dashboard can start playing while
    the rest of the clip is still arriving; OpenCV writes it last, which forces a full
    download before the first frame appears.
    """
    if video_codec(src_path) == "h264":
        return None                     # already web-playable -> upload as-is
    from violations.clip_encoder import h264_output_args
    cmd = [
        FFMPEG_BIN, "-y", "-loglevel", "error",
        "-i", src_path,
        *h264_output_args(),            # same CRF/preset/pix_fmt/faststart as the live encoder
        dst_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        print(f"[worker] ffmpeg not found ({FFMPEG_BIN!r}) -- uploading the mp4v original, "
              f"which browsers CANNOT play. Install ffmpeg or set FFMPEG_BIN.")
        return None
    except subprocess.TimeoutExpired:
        print("[worker] ffmpeg timed out after 900s -- uploading the mp4v original")
        return None
    if proc.returncode != 0 or not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
        print(f"[worker] ffmpeg failed (exit {proc.returncode}) -- uploading the mp4v original\n"
              f"         {(proc.stderr or '').strip()[:300]}")
        return None
    return dst_path


def upload_clip(s3, local_path, key, job_dir, label):
    """Upload one clip to `key`, transcoding first only if it is not already H.264."""
    stem = os.path.splitext(os.path.basename(key))[0]
    h264_local = os.path.join(job_dir, f"h264_{stem}.mp4")
    src_bytes = os.path.getsize(local_path)

    encoded = transcode_h264(local_path, h264_local)
    to_upload = encoded or local_path
    if encoded:
        out_bytes = os.path.getsize(encoded)
        print(f"[worker] {label}: H.264 {src_bytes / 1e6:.1f} MB -> {out_bytes / 1e6:.1f} MB")
    else:
        print(f"[worker] {label}: {src_bytes / 1e6:.1f} MB (already H.264, uploaded as-is)")

    s3.upload_file(to_upload, R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})
    return key


def download_session(s3, files, job_dir):
    """Pull the drive's artefacts from R2 into `job_dir` under the names the pipeline expects
    beside the video (that is how main.py auto-detects an Android clip). -> the video's path."""
    video_key = files.get("video")
    if not video_key:
        raise RuntimeError("drive has no video key")
    video_local = os.path.join(job_dir, os.path.basename(video_key))
    s3.download_file(R2_BUCKET, video_key, video_local)
    for fkey, fname in SENSOR_KEYS.items():
        key = files.get(fkey)
        if key:
            s3.download_file(R2_BUCKET, key, os.path.join(job_dir, fname))
    return video_local


def upload_evidence(s3, manifest, clips, session, out_dir, video_local, job_dir):
    """Upload one clip per violation to <session>/out/. -> {violation_id: R2 key}.

    A violation whose clip failed to render falls back to a single 5-second cut of the raw video
    (uploaded at most once), so `videoClipPath` is never null.

    Every clip is transcoded to H.264 on the way out -- see transcode_h264 for why."""
    keys = {}
    fallback_key = None
    for rec in manifest.get("violations", []):
        stem = rec["violation_id"]
        rel = clips.get(stem)
        if rel:
            keys[stem] = upload_clip(s3, os.path.join(out_dir, rel),
                                     f"{session}/out/{stem}.mp4", job_dir, stem)
            continue
        if fallback_key is None:                       # render failed -> ship the raw head clip
            local = os.path.join(job_dir, "evidence_5s.mp4")
            clip_first_seconds(video_local, local, seconds=5)
            print(f"[worker] {stem}: no rendered clip; falling back to a 5 s cut")
            fallback_key = upload_clip(s3, local, f"{session}/out/evidence_5s.mp4",
                                       job_dir, "evidence_5s")
        keys[stem] = fallback_key
    return keys


def upload_plate_images(s3, plates, session, out_dir):
    """Upload each violation's plate picture to <session>/out/<violation_id>_plate.png.
    -> {violation_id: R2 key}.

    Best effort, unlike the clips: the backend requires a clip, but a plate picture is optional,
    so a failed upload is logged and that violation is reported without one."""
    keys = {}
    for stem, rel in plates.items():
        key = f"{session}/out/{stem}_plate.png"
        try:
            s3.upload_file(os.path.join(out_dir, rel), R2_BUCKET, key,
                           ExtraArgs={"ContentType": "image/png"})
        except Exception as e:
            print(f"[worker] {stem}: plate picture upload failed ({e}); reporting it without one")
            continue
        keys[stem] = key
    return keys


def keep_annotated_video(path, session):
    """--keep-annotated: move the drive's annotated video out of its job dir (which ship()
    deletes) to ANNOTATED_DIR/<session>_annotated.mp4. Failing to keep it never fails the drive."""
    if not path or not os.path.isfile(path):
        print(f"[worker] --keep-annotated: no annotated video was rendered for {session}")
        return
    dest = os.path.join(ANNOTATED_DIR, f"{session}_annotated.mp4")
    try:
        os.makedirs(ANNOTATED_DIR, exist_ok=True)
        shutil.move(path, dest)
    except OSError as e:
        print(f"[worker] --keep-annotated: could not keep {path} ({e})")
        return
    print(f"[worker] annotated video kept -> {dest}")


class PendingShip:
    """One drive's finished pipeline output, waiting to be transcoded, uploaded and reported.

    Handed from process() (main thread) to ship() (the ship thread). It OWNS `job_dir` from
    the moment it is created: nothing else may delete that folder, because the clips, the
    source video (the 5 s fallback cut) and gps.csv (the violation's GPS fix) are all still
    read from it while the upload runs."""
    __slots__ = ("drive_id", "session", "job_dir", "out_dir", "video_local", "manifest", "clips",
                 "plates")

    def __init__(self, drive_id, session, job_dir, out_dir, video_local, manifest, clips, plates):
        self.drive_id    = drive_id
        self.session     = session
        self.job_dir     = job_dir
        self.out_dir     = out_dir
        self.video_local = video_local
        self.manifest    = manifest
        self.clips       = clips
        self.plates      = plates


def process(s3, job, models, timer) -> PendingShip:
    """Download the drive, run the pipeline, unpack the evidence bundle.

    Stops at the point where only transport is left and returns a PendingShip; the transcode,
    upload, violation POSTs, the drive's `complete` call and the cleanup all happen in ship(),
    which normally runs on a background thread so this worker can claim the next drive while
    the previous one's clips are still being encoded and pushed to R2."""
    drive_id = job["driveId"]
    session  = job["sessionId"]
    files    = job["files"] or {}
    job_dir  = os.path.join(WORK_DIR, session)
    out_dir  = os.path.join(job_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    # 2. Download every artifact to its expected local filename.
    with timer.measure("worker.download"):
        video_local = download_session(s3, files, job_dir)
    video_name  = os.path.splitext(os.path.basename(video_local))[0]
    print(f"[worker] {session}: {wc.describe_mode(wc.session_mode(job_dir))}")

    # 3. Run the pipeline (Android clip: ego pose reconstructed from the sensor CSVs).
    #    `job` (with no upload_url) switches the export stage on, so the pipeline renders one
    #    annotated clip per violation and writes the ranked bundle locally instead of PUTting it.
    import main                       # lazy: a missing GPU stack fails per-job, not at startup
    wc.attach_models(main, models)     # fresh evidence buffer for this clip
    wc.reset_speed_limit_lookup()      # re-arm the Overpass circuit breaker for this drive
    result = main.process_video_with_models(
        video_local, models.yolo, models.lane, models.tire,
        out_dir=out_dir, is_simulation=False, job={"job_id": session})
    if KEEP_ANNOTATED:
        keep_annotated_video(result.get("annotated_video"), session)

    # 4. Unpack the bundle into out/violations/<violation_id>/. The clips are only MOVED into
    #    place here (a local rename); encoding and uploading them is ship()'s job.
    violations_dir = os.path.join(out_dir, "violations")
    manifest = wc.unpack_bundle(os.path.join(out_dir, f"{video_name}_violations_bundle.tar.gz"),
                                violations_dir)
    has_violations = bool(manifest and manifest.get("violations"))
    clips = wc.move_clips(manifest, out_dir, violations_dir) if has_violations else {}
    plates = wc.collect_plate_images(manifest, out_dir, violations_dir) if has_violations else {}
    return PendingShip(drive_id, session, job_dir, out_dir, video_local, manifest, clips, plates)


def ship(s3, p: PendingShip) -> None:
    """The transport tail: transcode + upload one clip per violation, POST the violations,
    mark the drive complete, then delete the job dir.

    Runs on the ship thread (see submit_ship). Everything it touches lives under
    `p.job_dir` or is stateless transport, so it never races the pipeline running on the main
    thread for the NEXT drive. It is also the only place that marks this drive processed --
    a drive must not be reported complete while its evidence clips are still uploading, or the
    dashboard shows a finished drive whose videoClipPath 404s."""
    t0 = time.perf_counter()
    try:
        if not p.manifest or not p.manifest.get("violations"):
            print(f"[worker] drive {p.drive_id}: 0 violation(s)")
            complete(p.drive_id, "processed")
            return

        keys = upload_evidence(s3, p.manifest, p.clips, p.session, p.out_dir,
                               p.video_local, p.job_dir)
        plate_keys = upload_plate_images(s3, p.plates, p.session, p.out_dir)

        # 5. Report each violation, with its own clip, its GPS fix and its real type.
        payloads = wc.build_violation_payloads(
            p.manifest, keys, drive_id=p.drive_id, session_id=p.session, session_dir=p.job_dir,
            detected_at=datetime.now(timezone.utc).isoformat(), plate_images=plate_keys)
        for v in payloads:
            report_violation(v)

        complete(p.drive_id, "processed")
        print(f"[worker] drive {p.drive_id}: {len(payloads)} violation(s) reported "
              f"({sum(1 for v in payloads if v['plate'])} with a plate read, "
              f"{sum(1 for v in payloads if v['plateImagePath'])} with a plate picture)")
        print(f"[ship] drive {p.drive_id}: transcode + upload took "
              f"{run_timing.fmt_dur(time.perf_counter() - t0)}")
    except Exception as e:
        traceback.print_exc()
        complete(p.drive_id, "failed", str(e))
        print(f"[ship] drive {p.drive_id} failed: {e}")
    finally:
        shutil.rmtree(p.job_dir, ignore_errors=True)


def submit_ship(s3, pending: PendingShip) -> None:
    """Hand the tail to the ship thread, or run it inline when SHIP_ASYNC=0.

    Blocks once MAX_PENDING_SHIPS drives are already queued: the pipeline must not run so far
    ahead of the uploads that finished job dirs (a full source video each) pile up on disk."""
    if _SHIP_POOL is None:
        ship(s3, pending)                      # synchronous -- exactly the pre-thread behaviour
        return

    _SHIP_SLOTS.acquire()

    def _run():
        try:
            ship(s3, pending)
        finally:
            _SHIP_SLOTS.release()

    _SHIP_POOL.submit(_run)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    s3 = make_s3()

    # Headless-safe (this often runs in a container with no display) + load the detectors once.
    # The stopwatch wraps the imports and the loaders so the per-job report can say what the
    # start-up cost was and, per drive, how long each model spent on that video.
    timer = run_timing.RunTimer()
    with timer.measure("load.imports"):
        import cloud_env
        cloud_env.init()
        import main as pipeline
    print("[worker] loading models ...")
    timer.install_models(pipeline)
    with timer.measure("load.total"):
        models = wc.load_models(pipeline, repo_root=_HERE)

    # Draw the detected lane markings into the annotated video and every evidence clip, so the
    # reviewer on the dashboard sees the line a crossing was reported against. Same overlay the
    # offline runs produce (lane_render.py); silently skipped if the lane model didn't load.
    lanes = None
    if models.lane is not None:
        import lane_render
        lanes = lane_render.install(pipeline, overlay=True)

    # After lane_render, so the stopwatch times the renderers that actually run.
    timer.install_pipeline(pipeline)
    timer.install_worker(wc)
    # Ctrl+C / SIGTERM / `python worker.py --stop[-now]` from now on stop the loop at a safe point.
    stopper = worker_stop.StopControl(STOP_FILE).install()
    print(f"[worker] ready -- polling {SERVER_URL} every {POLL_SECONDS}s as '{WORKER_ID}'")
    print("[worker] to stop: python worker.py --stop (after the current drive) "
          "| python worker.py --stop-now (hand the current drive back to the queue)")

    if _SHIP_POOL is not None:
        print(f"[worker] evidence shipping runs off-thread "
              f"(up to {MAX_PENDING_SHIPS} drive(s) queued; SHIP_ASYNC=0 to disable)")
    if KEEP_ANNOTATED:
        print(f"[worker] --keep-annotated: each drive's annotated video is kept in {ANNOTATED_DIR}")

    try:
        while not stopper.requested:
            job = claim_job()
            if not job:
                stopper.idle(POLL_SECONDS)
                continue

            drive_id = job["driveId"]
            session  = job["sessionId"]
            if stopper.requested:
                # The stop arrived while this claim was in flight: start no new work.
                release(drive_id, "worker stopping; drive not started")
                break
            print(f"[worker] claimed drive {drive_id} (session {session})")
            pending = None
            try:
                with stopper.drive(), timer.session(session):
                    try:
                        if lanes is not None:
                            lanes.reset()       # forget the previous drive's lane polygons
                        pending = process(s3, job, models, timer)
                    except Exception as e:
                        traceback.print_exc()
                        complete(drive_id, "failed", str(e))
                        shutil.rmtree(os.path.join(WORK_DIR, session), ignore_errors=True)
                        print(f"[worker] drive {drive_id} failed: {e}")
            except KeyboardInterrupt:
                # Stop-now mid-pipeline. Nothing has been reported for this drive yet, so the
                # server can safely hand it to any worker again -- tell it now rather than
                # leaving the drive 'processing' until the stale-job timeout.
                print(f"\n[worker] stop now: abandoning drive {drive_id} mid-processing")
                release(drive_id, "worker stopped mid-processing")
                shutil.rmtree(os.path.join(WORK_DIR, session), ignore_errors=True)
                break
            # This worker polls forever, so the breakdown is printed per drive rather than at
            # exit. It covers the pipeline only -- the transcode/upload tail is timed by ship().
            timer.report_last_session(title=f"RUN TIMING -- drive {drive_id} (session {session})")
            if pending is not None:
                # Blocks only if the ship queue is already full; otherwise we go straight back
                # to claim_job() while this drive's clips encode and upload behind us.
                submit_ship(s3, pending)
    finally:
        if _SHIP_POOL is not None:
            print("[worker] draining pending evidence uploads before exit ... "
                  "(Ctrl+C again abandons them: those drives may then get duplicate violations)")
            _SHIP_POOL.shutdown(wait=True)   # never drop a detection on the way out
    print("[worker] stopped")


if __name__ == "__main__":
    if "--stop" in sys.argv or "--stop-now" in sys.argv:
        # Not a worker: just leave a request for the running one(s) and exit (worker_stop.py).
        wanted = worker_stop.NOW if "--stop-now" in sys.argv else worker_stop.GRACEFUL
        level = worker_stop.request_stop(STOP_FILE, wanted)
        print(f"[worker] {'stop-now' if level == worker_stop.NOW else 'stop'} requested ({STOP_FILE}); "
              f"every worker using this WORK_DIR picks it up within a second")
        sys.exit(0)
    try:
        main()
    except KeyboardInterrupt:
        # main()'s finally has already drained the ship queue by the time we get here.
        print("\n[worker] interrupted -- stopped")
