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


def report_violation(s3, drive_id, evidence_key, v):
    body = {
        "driveId": drive_id,
        "videoClipPath": evidence_key,
        "carId": v["carId"],
        "calculatedSpeed": v["calculatedSpeed"],
        "lat": v["lat"],
        "lon": v["lon"],
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


def clip_first_seconds(src_path, dst_path, seconds=5):
    """TEMPORARY evidence stand-in: write the first `seconds` of `src_path` to `dst_path`
    with OpenCV (already a pipeline dependency -> no new install, no ffmpeg-on-PATH
    requirement). Real per-violation clips are produced upstream later; this only exists
    to exercise the upload -> serve path end to end. Returns the frame count written."""
    import cv2
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video to clip: {src_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = max(1, int(round(fps * seconds)))
    writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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


def process(s3, job, yolo_model):
    drive_id  = job["driveId"]
    session   = job["sessionId"]
    files     = job["files"] or {}
    job_dir   = os.path.join(WORK_DIR, session)
    os.makedirs(job_dir, exist_ok=True)

    # 2. Download every artifact to its expected local filename.
    video_key = files.get("video")
    if not video_key:
        raise RuntimeError("drive has no video key")
    video_local = os.path.join(job_dir, os.path.basename(video_key))
    s3.download_file(R2_BUCKET, video_key, video_local)
    for fkey, fname in SENSOR_KEYS.items():
        key = files.get(fkey)
        if key:
            s3.download_file(R2_BUCKET, key, os.path.join(job_dir, fname))

    # 3. Run the pipeline (Android clip: ego pose reconstructed from the sensor CSVs).
    from main import run_pipeline   # imported lazily so a missing GPU stack fails per-job, not at startup
    result = run_pipeline(video_local, is_simulation=False, yolo_model=yolo_model)

    violations = result.get("violations", [])

    # 4. Evidence clip. TEMPORARY: real per-violation clips will be produced upstream
    # later (a teammate owns that). For now upload a single 5-second clip cut from the
    # START of the raw video, so the full download -> process -> upload -> serve path is
    # exercised end to end. It's uploaded unconditionally (even with zero violations) so
    # the R2 .../out/ path can be verified; every violation references this one key.
    clip_local = os.path.join(job_dir, "evidence_5s.mp4")
    clip_first_seconds(video_local, clip_local, seconds=5)
    evidence_key = f"{session}/out/{os.path.basename(clip_local)}"
    s3.upload_file(clip_local, R2_BUCKET, evidence_key, ExtraArgs={"ContentType": "video/mp4"})

    # 5. Report each violation (all referencing the temporary evidence clip above).
    for v in violations:
        report_violation(s3, drive_id, evidence_key, v)

    print(f"[worker] drive {drive_id}: {len(violations)} violation(s)")


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    s3 = make_s3()

    # Load the detector once, reuse for every job.
    from main import loadYoloModel
    print("[worker] loading YOLO model ...")
    yolo_model = loadYoloModel()
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
