"""fake_session.py -- push a synthetic drive through the REAL online path.

This is the Android app, replayed from the desktop. It does exactly what
PostDriveActivity does, in the same order, with the same request bodies:

    register / login (driver)
      -> POST /api/driver/upload/init      -> 7 presigned R2 PUT URLs
      -> PUT each of the 7 files DIRECTLY to Cloudflare R2
      -> POST /api/driver/upload/complete  -> server HEADs all 7, drive := queued
      -> (worker.py claims it, downloads from R2, runs main.py, uploads evidence)
      -> poll until the drive reaches a terminal state
      -> print the violations the worker wrote back to Mongo

Nothing is faked past the client boundary. The bytes really land in R2, the server
really presigns and validates, the worker really processes and reports. The only
"fake" part is that the session comes from a folder on disk instead of a phone --
which is precisely what makes it a usable test: no phone, no network path to the
phone, no 100 MB upload over Wi-Fi.

The source folder must be a real session: a video plus the six fixed-name artifacts
(frames.csv, gps.csv, gravity.csv, gyro.csv, linacc.csv, intrinsics.json). That is
the "7-file contract" the whole system is built on, and real_vids/ is full of them.

Examples
--------
  # the whole loop, on the committed sample session (trimmed to keep it quick)
  python tools/fake_session.py real_vids/fast_hw/test_session --trim-seconds 20

  # full-length video, no trim
  python tools/fake_session.py real_vids/fast_hw/test_session

  # just upload and queue it; don't wait for the worker
  python tools/fake_session.py real_vids/fast_hw/test_session --no-watch

  # against a remote backend (e.g. an ngrok tunnel)
  python tools/fake_session.py <dir> --server https://xxx.ngrok-free.app/api
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The six fixed-name artifacts; the video's name is free-form. Same list, same order
# as driverController.SENSOR_FILES / PostDriveActivity.DATA_FILES.
SENSOR_FILES = ["frames.csv", "gps.csv", "gravity.csv", "gyro.csv", "linacc.csv", "intrinsics.json"]
MIME = {".csv": "text/csv", ".json": "application/json", ".mp4": "video/mp4",
        ".mov": "video/quicktime", ".m4v": "video/x-m4v", ".mpeg": "video/mpeg"}

# A per-violation clip left behind by an earlier run (v100_SPEEDING_f1200.mp4) is an
# OUTPUT, never the drive video. Same rule as worker_common.VIOLATION_CLIP_RE.
CLIP_RE = re.compile(r"^v\d+_.+_f\d+\.(mp4|mov|m4v)$", re.I)
DERIVED_RE = re.compile(r"(_annotated|_web|_undistorted)\.(mp4|mov|m4v)$", re.I)

TERMINAL = {"processed", "failed", "rejected"}


def log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


def die(msg: str) -> "NoReturn":
    print(f"\n!! {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.token = None

    def _headers(self):
        h = {"Content-Type": "application/json",
             # The app sends this so an ngrok free tunnel doesn't serve its HTML
             # interstitial instead of our JSON. Harmless everywhere else.
             "ngrok-skip-browser-warning": "true"}
        if self.token:
            h["Authorization"] = "Bearer " + self.token
        return h

    def post(self, path, body):
        return requests.post(self.base + path, data=json.dumps(body),
                             headers=self._headers(), timeout=60)

    def get(self, path):
        return requests.get(self.base + path, headers=self._headers(), timeout=60)


def ensure_account(api: Api, email: str, password: str, role: str, invite: str) -> str:
    """Register (idempotently) and log in. -> JWT. The `users` collection starts empty
    on a fresh database, so registering first is what makes this runnable from scratch."""
    body = {"name": f"Fake {role}", "email": email, "password": password, "role": role}
    if role == "authority":
        body["inviteCode"] = invite
    r = api.post("/auth/register", body)
    if r.status_code == 201:
        log("auth", f"registered {role} {email}")
    elif r.status_code == 400 and "already registered" in r.text:
        log("auth", f"{role} {email} already exists")
    else:
        die(f"register({role}) failed ({r.status_code}): {r.text}")

    r = api.post("/auth/login", {"email": email, "password": password})
    if r.status_code != 200:
        die(f"login({role}) failed ({r.status_code}): {r.text}")
    return r.json()["data"]["token"]


# --------------------------------------------------------------------------- #
# the session on disk
# --------------------------------------------------------------------------- #
def pick_video(session_dir: str, explicit: str | None) -> str:
    """The drive video: the largest .mp4 that is not an output of a previous run."""
    if explicit:
        p = explicit if os.path.isabs(explicit) else os.path.join(session_dir, explicit)
        if not os.path.isfile(p):
            die(f"--video not found: {p}")
        return p
    cands = [f for f in os.listdir(session_dir)
             if f.lower().endswith((".mp4", ".mov", ".m4v", ".mpeg"))
             and not CLIP_RE.match(f) and not DERIVED_RE.search(f)]
    if not cands:
        die(f"no drive video in {session_dir} (only evidence clips / derived files?)")
    cands.sort(key=lambda f: os.path.getsize(os.path.join(session_dir, f)), reverse=True)
    return os.path.join(session_dir, cands[0])


def check_session(session_dir: str) -> None:
    missing = [n for n in SENSOR_FILES
               if not os.path.isfile(os.path.join(session_dir, n))
               or os.path.getsize(os.path.join(session_dir, n)) == 0]
    if missing:
        die(f"{session_dir} is not a complete session -- missing/empty: {', '.join(missing)}\n"
            f"   The server enforces all 7 files; pick a folder from real_vids/ that has them.")


def trim_video(src: str, seconds: float, work_dir: str) -> str:
    """Head-cut the first `seconds` into a new mp4 so the upload + pipeline run take
    minutes, not an hour. A HEAD cut keeps frames.csv aligned: frame 0 is still frame 0,
    the pipeline just sees fewer of them."""
    import cv2
    os.makedirs(work_dir, exist_ok=True)
    dst = os.path.join(work_dir, "fake_" + os.path.basename(src))
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        die(f"cannot open video to trim: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    want = max(1, int(round(fps * seconds)))
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while n < want:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    if n == 0:
        die(f"trim produced no frames from {src}")
    log("trim", f"{n} frames ({n/fps:.1f}s @ {fps:.0f}fps, {w}x{h}) -> "
                f"{os.path.basename(dst)}  {os.path.getsize(dst)/1e6:.1f} MB")
    return dst


# --------------------------------------------------------------------------- #
# the upload, exactly as PostDriveActivity does it
# --------------------------------------------------------------------------- #
def do_init(api: Api, session_id: str, video: str) -> dict:
    size = os.path.getsize(video)
    log("init", f"POST /driver/upload/init  session={session_id} "
                f"video={os.path.basename(video)} ({size/1e6:.1f} MB)")
    r = api.post("/driver/upload/init", {"sessionId": session_id,
                                         "videoName": os.path.basename(video),
                                         "videoSize": size})
    if r.status_code != 201:
        die(f"init rejected ({r.status_code}): {r.text}\n"
            f"   507 = bucket full | 400 = size out of range (VIDEO_MIN_BYTES/MAX) "
            f"| 409 = sessionId already used")
    data = r.json()["data"]
    log("init", f"drive {data['driveId']} created; {len(data['uploads'])} presigned PUT URLs")
    return data


def do_puts(uploads: dict, session_dir: str, video: str) -> None:
    """PUT every file straight to R2. The server never sees these bytes."""
    sensor_keys = set(SENSOR_FILES)
    video_key = next((k for k in uploads if k not in sensor_keys), None)
    if video_key is None:
        die(f"init returned no video upload URL; keys were {list(uploads)}")

    plan = [(video_key, video)] + [(n, os.path.join(session_dir, n)) for n in SENSOR_FILES
                                   if n in uploads]
    total = sum(os.path.getsize(p) for _, p in plan)
    log("put", f"uploading {len(plan)} objects, {total/1e6:.1f} MB total, direct to R2")

    sent = 0
    for key, path in plan:
        url = uploads[key]
        if url.startswith("local-placeholder://"):
            die("the server handed back a placeholder URL -- R2 is NOT configured in "
                "backend/.env (R2_BUCKET / R2_ENDPOINT / keys). Nothing was uploaded.")
        mime = MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        with open(path, "rb") as fh:
            r = requests.put(url, data=fh, headers={"Content-Type": mime}, timeout=900)
        if r.status_code not in (200, 201):
            extra = ""
            if r.status_code == 403:
                extra = ("\n   403 from R2 means the presigned URL was signed with credentials "
                         "the bucket rejects.\n   Check R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / "
                         "R2_BUCKET in backend/.env,\n   then RESTART the backend (it reads .env "
                         "once at startup).")
            die(f"PUT {key} failed ({r.status_code}): {r.text[:300]}{extra}")
        sent += os.path.getsize(path)
        log("put", f"  {key:<20} {os.path.getsize(path)/1e6:>8.2f} MB   "
                   f"[{100*sent//total:>3}%]")


def do_complete(api: Api, session_id: str) -> None:
    log("done", "POST /driver/upload/complete  (server HEADs all 7 objects)")
    r = api.post("/driver/upload/complete", {"sessionId": session_id})
    if r.status_code != 202:
        die(f"complete rejected ({r.status_code}): {r.text}\n"
            f"   The server could not verify the uploaded objects -- see the API console.")
    log("done", "drive queued -- worker.py will claim it within POLL_SECONDS (default 5s)")


# --------------------------------------------------------------------------- #
# --fake-worker: the last leg only, with no R2 and no GPU
# --------------------------------------------------------------------------- #
def load_internal_token() -> str:
    """/api/internal/* is gated by INTERNAL_TOKEN when backend/.env sets one. Same
    minimal reader worker.py uses, so the secret stays in exactly one place."""
    path = os.path.join(REPO_ROOT, "backend", ".env")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("INTERNAL_TOKEN") and "=" in line:
                return line.partition("=")[2].strip()
    return ""


def fake_worker(api: Api, drive_id: str, session_id: str, n: int) -> None:
    """Post what a worker WOULD have posted, straight to /api/internal/*.

    This skips R2 and the CV pipeline entirely -- the violations are invented, not
    detected. It exists to exercise the second half of the loop (worker -> API ->
    Mongo -> dashboard) when R2 is down or no GPU is free. Never use it to claim the
    detector found something.
    """
    token = load_internal_token()
    headers = {"Content-Type": "application/json", "x-worker-id": "fake-worker"}
    if token:
        headers["x-internal-token"] = token
    log("fake", f"posting {n} invented violation(s) as a worker would "
                f"({'with' if token else 'no'} INTERNAL_TOKEN)")

    for i in range(n):
        body = {
            "driveId": drive_id,
            # The shape a real worker uploads: <sessionId>/out/<violation_id>.mp4
            "videoClipPath": f"{session_id}/out/v{100+i}_SPEEDING_f{1200*(i+1)}.mp4",
            "carId": f"{12+i}-345-6{i}",
            "calculatedSpeed": 96.0 + 7 * i,
            "lat": 32.0513233, "lon": 34.9047184,
            "violationType": "speeding" if i % 2 == 0 else "lane_crossing",
        }
        r = requests.post(api.base + "/internal/violation", data=json.dumps(body),
                          headers=headers, timeout=30)
        if r.status_code != 201:
            die(f"internal/violation failed ({r.status_code}): {r.text}")
        log("fake", f"  violation {r.json()['data']['violationId']} "
                    f"{body['violationType']} {body['carId']}")

    r = requests.post(api.base + f"/internal/drive/{drive_id}/complete",
                      data=json.dumps({"status": "processed"}), headers=headers, timeout=30)
    if r.status_code != 200:
        die(f"internal/drive/complete failed ({r.status_code}): {r.text}")
    log("fake", "drive marked processed (the API's R2 retention sweep will log warnings "
                "for raw keys that were never uploaded -- expected in this mode)")


# --------------------------------------------------------------------------- #
# watch the worker do its half
# --------------------------------------------------------------------------- #
def find_drive(auth: Api, session_id: str) -> dict | None:
    r = auth.get("/authority/drives")
    if r.status_code != 200:
        return None
    for d in r.json().get("data", []):
        if d.get("sessionId") == session_id:
            return d
    return None


def watch(auth: Api, session_id: str, timeout: float, interval: float) -> dict | None:
    log("watch", f"polling drive status every {interval:g}s (Ctrl+C to stop watching; "
                 f"the worker keeps going)")
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        drive = find_drive(auth, session_id)
        if drive is None:
            die("cannot read /authority/drives -- is the authority account valid?")
        status = drive.get("status")
        if status != last:
            elapsed = time.time() - t0
            who = f" by {drive.get('workerId')}" if drive.get("workerId") else ""
            log("watch", f"{last or 'created'} -> {status}{who}   (+{elapsed:.0f}s)")
            last = status
        if status in TERMINAL:
            return drive
        time.sleep(interval)
    log("watch", f"timed out after {timeout:.0f}s in state '{last}'")
    return None


def report(auth: Api, drive: dict) -> int:
    status = drive.get("status")
    if status != "processed":
        print(f"\n=== drive {status} ===")
        if drive.get("error"):
            print(f"  error: {drive['error']}")
        if status == "failed":
            print("  The worker claimed it but the pipeline threw -- see the worker console.")
        return 1

    r = auth.get("/authority/violations")
    if r.status_code != 200:
        die(f"could not fetch violations ({r.status_code}): {r.text}")
    mine = [v for v in r.json().get("data", []) if str(v.get("driveId")) == str(drive["_id"])]

    print(f"\n=== drive processed: {len(mine)} violation(s) written back to Mongo ===")
    for v in mine:
        loc = v.get("location") or {}
        print(f"  {v['_id']}  {v.get('violationType','?'):<13} plate={v.get('carId','?'):<12} "
              f"{v.get('calculatedSpeed','?')} km/h  "
              f"@({loc.get('lat')},{loc.get('lon')})  status={v.get('status')}")
        print(f"      clip: {v.get('videoClipPath')}")
    if not mine:
        print("  (the pipeline found nothing in this clip -- that is a valid result; "
              "try a longer --trim-seconds or a clip with a real violation)")
    print("\nOpen the dashboard to review them: http://localhost:5173")
    return 0


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Replay a session folder through the real "
                                             "upload -> R2 -> worker -> Mongo path")
    ap.add_argument("session_dir", help="folder holding the video + the 6 sensor files")
    ap.add_argument("--server", default="http://localhost:5000/api", help="API base URL")
    ap.add_argument("--video", default=None, help="which file in the folder is the drive video")
    ap.add_argument("--trim-seconds", type=float, default=0,
                    help="upload only the first N seconds (0 = the whole video). A head cut "
                         "keeps frames.csv aligned and makes the round trip minutes, not hours")
    ap.add_argument("--session-id", default=None, help="default: session_<epoch ms>")
    ap.add_argument("--email", default="fakedriver@roadguard.local")
    ap.add_argument("--password", default="password123")
    ap.add_argument("--authority-email", default="fakeofficer@roadguard.local")
    ap.add_argument("--invite-code", default=os.environ.get("INVITE_CODE", "ROADGUARD-2026"))
    ap.add_argument("--no-watch", action="store_true", help="queue it and exit")
    ap.add_argument("--fake-worker", type=int, default=0, metavar="N",
                    help="skip R2 and the pipeline: create the drive, then POST N INVENTED "
                         "violations to /api/internal/* as a worker would. Exercises "
                         "API -> Mongo -> dashboard when R2 is down or no GPU is free. "
                         "These are not detections")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds to wait for the worker")
    ap.add_argument("--interval", type=float, default=5, help="status poll interval")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = os.path.abspath(args.session_dir)
    if not os.path.isdir(session_dir):
        die(f"not a directory: {session_dir}")
    check_session(session_dir)

    api = Api(args.server)
    try:
        h = requests.get(api.base + "/health", timeout=10)
        log("api", f"{api.base}/health -> {h.status_code} {h.text.strip()[:80]}")
    except requests.RequestException as e:
        die(f"cannot reach the API at {api.base}: {e}\n   Start it:  cd backend && npm run dev")

    video = pick_video(session_dir, args.video)
    log("src", f"session {session_dir}")
    log("src", f"video   {os.path.basename(video)}  {os.path.getsize(video)/1e6:.1f} MB")
    # Trimming is pointless in --fake-worker mode: nothing is uploaded or processed.
    if args.trim_seconds and not args.fake_worker:
        video = trim_video(video, args.trim_seconds,
                           os.path.join(REPO_ROOT, "outputs", "fake_session"))

    session_id = args.session_id or f"session_{int(time.time()*1000)}"

    api.token = ensure_account(api, args.email, args.password, "driver", args.invite_code)
    data = do_init(api, session_id, video)

    auth = Api(args.server)
    auth.token = ensure_account(auth, args.authority_email, args.password,
                                "authority", args.invite_code)

    if args.fake_worker:
        fake_worker(api, data["driveId"], session_id, args.fake_worker)
        drive = find_drive(auth, session_id)
        print("\n*** --fake-worker: R2 and the CV pipeline were BYPASSED. The violations "
              "below\n    were invented by this script, not detected. ***")
        return report(auth, drive) if drive else 1

    do_puts(data["uploads"], session_dir, video)
    do_complete(api, session_id)

    if args.no_watch:
        log("watch", f"skipped (--no-watch). Session: {session_id}")
        return 0

    drive = watch(auth, session_id, args.timeout, args.interval)
    if drive is None:
        return 1
    return report(auth, drive)


if __name__ == "__main__":
    sys.exit(main())
