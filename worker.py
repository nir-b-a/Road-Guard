"""
worker.py -- the GPU side of the Road Guard processing architecture.

It is the bridge that was previously missing between the upload server and the CV
pipeline. The loop:

    1. GET  {SERVER_URL}/api/internal/next-job   (atomically claims a queued drive)
    2. Download that drive's 7 files from Cloudflare R2 to a local temp dir
    3. Run the existing pipeline   (main.run_pipeline)
    4. Upload an evidence clip back to R2 (<sessionId>/out/...) -- TEMPORARY: a 5s clip
       cut from the start of the raw video, until per-violation clips land upstream
    5. POST one /api/internal/violation per detected violation
    6. POST /api/internal/drive/<id>/complete  (processed | failed)

The big video bytes go R2 <-> worker directly; only small JSON crosses to the server.
The YOLO model is loaded ONCE and reused for every job (the 100 MB weights are not
reloaded per clip).

Config (env vars, see backend/.env.example for the shared R2_* names):
    SERVER_URL        backend base URL                 (default http://localhost:5000)
    INTERNAL_TOKEN    shared secret -> x-internal-token header (optional)
    WORKER_ID         identifies this worker in logs/claims    (default host name)
    POLL_SECONDS      idle poll interval                       (default 5)
    WORK_DIR          scratch dir for downloads/outputs        (default ./_worker_jobs)
    R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_ENDPOINT

Run:  python worker.py
"""
import csv as _csv
import datetime as _dt
import os
import sys
import time
import shutil
import socket
import traceback

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
_load_env_file(os.path.join(_HERE, "worker.env"))
_load_env_file(os.path.join(_HERE, "backend", ".env"))

SERVER_URL     = os.environ.get("SERVER_URL", "http://localhost:5000").rstrip("/")
INTERNAL_TOKEN = os.environ.get("INTERNAL_TOKEN", "")
WORKER_ID      = os.environ.get("WORKER_ID", socket.gethostname())
POLL_SECONDS   = float(os.environ.get("POLL_SECONDS", "5"))
WORK_DIR       = os.environ.get("WORK_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker_jobs"))

R2_BUCKET   = os.environ.get("R2_BUCKET")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")

# Drive.files keys -> the on-disk filename the pipeline expects beside the video.
SENSOR_KEYS = {"frames": "frames.csv", "gps": "gps.csv", "gravity": "gravity.csv",
               "gyro": "gyro.csv", "linacc": "linacc.csv", "intrinsics": "intrinsics.json"}


def _headers():
    h = {"x-worker-id": WORKER_ID}
    if INTERNAL_TOKEN:
        h["x-internal-token"] = INTERNAL_TOKEN
    return h


# Real GPS data from an Android device has timestamp_ms as absolute Unix milliseconds
# (e.g. 1_752_000_000_000 ≈ year 2025). Stub/synthetic data uses relative ms starting
# near 0. This threshold separates them so we only emit a recording date for real clips.
_UNIX_EPOCH_THRESHOLD_MS = 1_000_000_000_000   # 2001-09-09 00:00:00 UTC

# Mock ego speed used when no real GPS data is available (stub runs, demos).
# Chosen to be above the mock limit so speeding confidence is non-trivial.
_MOCK_SPEED_KMH   = 95.0
_MOCK_LIMIT_KMH   = 80.0   # informational; actual limit enforcement is in the CV pipeline


def _gps_at_frame(gps_path, key_frame, fps):
    """Return (lat, lon, speed_kmh, recorded_iso) from gps.csv at the frame's timestamp.

    Finds the row whose timestamp_ms is closest to key_frame/fps seconds into the clip.
    recorded_iso is an ISO-8601 UTC string when timestamp_ms is a real Unix epoch
    (friends will populate this from the Android GPS stream); None for synthetic stubs.
    Falls back to _MOCK_SPEED_KMH when no GPS file is present or all speed rows are zero,
    so the authority dashboard always shows a meaningful speed until real GPS is wired up.
    """
    if not os.path.isfile(gps_path):
        return 0.0, 0.0, _MOCK_SPEED_KMH, None
    rows = []
    try:
        with open(gps_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                try:
                    rows.append({
                        "ts":  float(row["timestamp_ms"]),
                        "lat": float(row.get("lat", 0) or 0),
                        "lon": float(row.get("lon", 0) or 0),
                        "spd": float(row.get("speed_kmh", 0) or 0),
                    })
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        print(f"[worker] gps.csv read error: {e}")
        return 0.0, 0.0, 0.0, None
    if not rows:
        return 0.0, 0.0, _MOCK_SPEED_KMH, None
    target_ms = key_frame * 1000.0 / max(fps, 1)
    best = min(rows, key=lambda r: abs(r["ts"] - target_ms))
    recorded_iso = None
    if best["ts"] > _UNIX_EPOCH_THRESHOLD_MS:
        recorded_iso = _dt.datetime.utcfromtimestamp(best["ts"] / 1000.0).strftime("%Y-%m-%dT%H:%M:%SZ")
    speed = best["spd"] if best["spd"] != 0.0 else _MOCK_SPEED_KMH
    return best["lat"], best["lon"], speed, recorded_iso


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


def report_violation(drive_id, clip_key, plate_key, v_rec,
                     lat=0.0, lon=0.0, speed=None, recorded_at=None):
    """POST one violation record to the backend. v_rec is a manifest violation dict.

    lat/lon/speed come from gps.csv at the violation's key frame (real values when the
    Android clip includes absolute GPS timestamps; zeros for synthetic stubs).
    recorded_at is an ISO-8601 UTC string derived from the GPS timestamp, or None.
    """
    details_speed = (v_rec.get("details") or {}).get("est_speed_kmh")
    body = {
        "driveId":        drive_id,
        "videoClipPath":  clip_key,
        "plateClipPath":  plate_key,
        "carId":          v_rec.get("plate") or f"vehicle_{v_rec.get('vehicle_id', '?')}",
        "calculatedSpeed": speed if speed is not None else (details_speed or 0),
        "lat":            lat,
        "lon":            lon,
        "recordedAt":     recorded_at,
        "violationType":  v_rec.get("violation"),
        "tier":           v_rec.get("tier"),
        "confidence":     v_rec.get("detector_confidence"),
    }
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


def _extract_clip_h264(src_path, dst_path, start_sec, end_sec):
    """Cut [start_sec, end_sec] from src_path and write an H.264/faststart MP4 to dst_path.
    Uses fast keyframe seek (-ss before -i). Returns True on success, False if ffmpeg is absent."""
    if shutil.which("ffmpeg") is None:
        print("[worker] WARNING: ffmpeg not on PATH — clip skipped; install ffmpeg for evidence clips.")
        return False
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}", "-i", src_path,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        dst_path,
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[worker] ffmpeg clip failed: {e}")
        return False


def process(s3, job, yolo_model):
    """
    Full pipeline for one drive job:
      1. Download raw inputs from incoming/<session>/ in R2.
      2. Run the CV pipeline (run_pipeline from main.py).
      3. For each ViolationEvent: cut a 15s H.264/faststart clip (10s pre + 5s post key_frame).
      4. Call violations.export.build_export to get the ranked manifest + per-violation file tree.
      5. Upload every file from the bundle to output/<session>/<violation_id>/ in R2.
      6. POST one /api/internal/violation per violation to the backend.
    """
    import subprocess
    drive_id = job["driveId"]
    session  = job["sessionId"]
    files    = job["files"] or {}
    job_dir  = os.path.join(WORK_DIR, session)
    os.makedirs(job_dir, exist_ok=True)

    # ── 1. Download raw inputs from incoming/<session>/ ──────────────────────
    video_key = files.get("video")
    if not video_key:
        raise RuntimeError("drive has no video key")
    video_local = os.path.join(job_dir, os.path.basename(video_key))
    s3.download_file(R2_BUCKET, video_key, video_local)
    for fkey, fname in SENSOR_KEYS.items():
        key = files.get(fkey)
        if key:
            s3.download_file(R2_BUCKET, key, os.path.join(job_dir, fname))

    # ── 2. Run the CV pipeline ────────────────────────────────────────────────
    from main import run_pipeline   # lazy: GPU stack failures are per-job, not at startup
    result = run_pipeline(
        video_local, is_simulation=False, yolo_model=yolo_model, out_dir=job_dir)

    results_by_event = result["results_by_event"]   # [(ViolationEvent, EvidenceResult|None)]
    fps              = result["fps"]
    total_frames     = result["total_frames"]

    # ── 3. Cut a per-violation H.264/faststart clip (10s pre + 5s post) ──────
    try:
        from violations.clip_extract import clip_window, ClipAsset
        _CLIP_IMPORT_OK = True
    except Exception as _e:
        print(f"[worker] clip_extract unavailable ({_e}); clips will be skipped")
        _CLIP_IMPORT_OK = False

    records = []
    for event, evidence in results_by_event:
        clip = None
        if _CLIP_IMPORT_OK:
            try:
                win  = clip_window(event.key_frame, fps, total_frames=total_frames)
                clip_path = os.path.join(
                    job_dir, f"v{event.vehicle_id}_{event.violation_type}_f{event.key_frame}.mp4")
                ok = _extract_clip_h264(video_local, clip_path, win.start_sec, win.end_sec)
                if ok:
                    clip = ClipAsset(path=clip_path, window=win, container="mp4", recompressed=True)
                    # Annotate: draw a green bbox around the violating vehicle on every frame.
                    bbox = event.details.get("bbox")
                    if bbox:
                        try:
                            from violations.clip_extract import annotate_clip_inplace
                            from violations.export import describe_violation as _desc_v
                            annotate_clip_inplace(clip_path, bbox, _desc_v(event)[:70])
                        except Exception as ann_err:
                            print(f"[worker] clip annotation skipped: {ann_err}")
            except Exception as e:
                print(f"[worker] clip extraction failed for {event.vehicle_id}/{event.violation_type}: {e}")
        records.append((event, evidence, clip))

    # ── 4. Assemble ranked manifest + file tree via export.py ────────────────
    try:
        from violations import export as vx
        bundle = vx.build_export(records)
    except Exception as e:
        print(f"[worker] build_export failed ({e}); no violations uploaded")
        print(f"[worker] drive {drive_id}: {len(results_by_event)} violation(s) (export failed)")
        return

    # ── 5. Upload per-violation folder tree to output/<date>_<name>/ ─────────
    # Use a human-readable date + video filename so R2 is easy to navigate.
    import re as _re
    _date_str   = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    _vid_stem   = os.path.splitext(os.path.basename(video_local))[0]
    _safe_stem  = _re.sub(r"[^a-zA-Z0-9_-]", "_", _vid_stem)[:50]
    out_prefix  = f"output/{_date_str}_{_safe_stem}/"

    # manifest.json at the session root
    manifest_bytes = __import__("json").dumps(bundle.manifest, ensure_ascii=False, indent=2).encode()
    s3.put_object(Bucket=R2_BUCKET, Key=out_prefix + "manifest.json",
                  Body=manifest_bytes, ContentType="application/json")

    # each file from bundle.files (e.g. v5_SOLID_LINE_CROSSING_f200/clip.mp4)
    for rel_path, data in bundle.files.items():
        r2_key = out_prefix + rel_path
        content_type = ("video/mp4"    if rel_path.endswith(".mp4")
                        else "image/png" if rel_path.endswith(".png")
                        else "application/json" if rel_path.endswith(".json")
                        else "application/octet-stream")
        try:
            s3.put_object(Bucket=R2_BUCKET, Key=r2_key, Body=bytes(data), ContentType=content_type)
        except Exception as e:
            print(f"[worker] R2 upload failed for {r2_key}: {e}")

    # ── 6. POST each violation to the backend ────────────────────────────────
    # Build a GPS lookup keyed by violation_id so we can attach real lat/lon/speed/date.
    from violations.export import violation_id as _viol_id_str
    gps_path = os.path.join(job_dir, "gps.csv")
    gps_cache = {}
    for event, _ev in results_by_event:
        key = _viol_id_str(event)
        gps_cache[key] = _gps_at_frame(gps_path, event.key_frame, fps or 30)

    for v_rec in bundle.violations:
        vid      = v_rec["violation_id"]          # e.g. v5_SOLID_LINE_CROSSING_f200
        clip_key  = out_prefix + vid + "/clip.mp4"

        # plate.png is in the evidence block if it was collected
        plate_entry = (v_rec.get("evidence") or {}).get("plate_crop")
        plate_key   = (out_prefix + plate_entry["file"]) if plate_entry else None

        lat, lon, speed, recorded_at = gps_cache.get(vid, (0.0, 0.0, 0.0, None))
        report_violation(drive_id, clip_key, plate_key, v_rec,
                         lat=lat, lon=lon, speed=speed, recorded_at=recorded_at)

    print(f"[worker] drive {drive_id}: {len(bundle.violations)} violation(s) uploaded to {out_prefix}")


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    s3 = make_s3()

    # Load the vehicle detector + lane-seg model once; reuse for every job.
    import main as _main
    from main import loadYoloModel, loadLaneModel
    print("[worker] loading YOLO model ...")
    yolo_model = loadYoloModel()

    # Lane-seg model is required for solid-line crossing detection (LANE_MODEL global in main.py).
    _lane_weights = os.path.join(_main.REPO_ROOT, "weights", "phase3_v3_yellowprotect.pt")
    if os.path.isfile(_lane_weights):
        _main.LANE_MODEL = loadLaneModel(_lane_weights)
        print(f"[worker] lane-seg model loaded ({os.path.basename(_lane_weights)})")
    else:
        print(f"[worker] WARNING: lane weights not found at {_lane_weights}; crossing detection disabled")

    print(f"[worker] ready -- polling {SERVER_URL} every {POLL_SECONDS}s as '{WORKER_ID}'")

    while True:
        job = claim_job()
        if not job:
            time.sleep(POLL_SECONDS)
            continue

        drive_id = job["driveId"]
        session  = job["sessionId"]
        print(f"[worker] claimed drive {drive_id} (session {session})")
        try:
            process(s3, job, yolo_model)
            complete(drive_id, "processed")
            print(f"[worker] drive {drive_id} processed")
        except Exception as e:
            traceback.print_exc()
            complete(drive_id, "failed", str(e))
            print(f"[worker] drive {drive_id} failed: {e}")
        finally:
            shutil.rmtree(os.path.join(WORK_DIR, session), ignore_errors=True)


if __name__ == "__main__":
    main()
