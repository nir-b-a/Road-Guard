"""
synth_session.py -- turn a bare drive video into a complete 7-file session folder.

A session recorded by the phone is a video PLUS six sensor/calibration files, and
that contract is enforced in three places: PostDriveActivity refuses to upload if
any of the six is missing or zero-length, /upload/init requires all seven to be
declared, and /upload/complete HEADs all seven in R2 before queueing the drive.
A video on its own therefore cannot enter the pipeline at any point.

This script synthesises the six missing files so an ordinary .mp4 becomes a
session the app will happily upload:

    frames.csv      REAL   - frame index -> timestamp, from the video's own fps
    gps.csv         SYNTH  - straight line at a constant speed from a start point
    gyro.csv        SYNTH  - all zeros (driving straight, no yaw)
    gravity.csv     SYNTH  - constant vector for a landscape dashcam mount
    linacc.csv      SYNTH  - all zeros (constant speed => no acceleration)
    intrinsics.json SYNTH  - frame size + horizontal FOV

Only frames.csv is genuinely derived from the video. The rest describe a car
driving perfectly straight at a constant speed, which is a *consistent* story
rather than a true one -- see "What the numbers mean" at the bottom of this file
before trusting any speed the pipeline reports for a synthesised session.

Usage
    python tools/synth_session.py DRIVE.mp4
    python tools/synth_session.py DRIVE.mp4 --speed-kmh 75 --push

    --push       adb-push the finished folder to the phone, where the app's
                 "Unfinished Upload" dialog will offer it on next launch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Constants  # noqa: E402

# The app's own package + storage layout (AndroidManifest applicationId, and
# getExternalFilesDir(null)/sessions/ per SessionStore.sessionsRoot).
ANDROID_PACKAGE = "com.example.roadgaurd"
ANDROID_SESSIONS = f"/sdcard/Android/data/{ANDROID_PACKAGE}/files/sessions"

# PostDriveActivity.MIN_VIDEO_BYTES -- the app rejects anything smaller before it
# even calls the server, and SessionStore hides such folders from the retry dialog.
MIN_VIDEO_BYTES = 5 * 1024 * 1024
MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024

# A phone in a landscape dashcam mount, camera looking forward and level. With the
# gyro at zero this vector cannot affect the result (it is only used to project the
# gyro onto the vertical axis, and the projection of a zero vector is zero), so it
# exists to keep the file well-formed and to stay correct if the gyro ever becomes
# non-zero. ground_distance -- the one consumer that would care -- is currently
# commented out in main.py.
DEFAULT_GRAVITY = (9.81, 0.0, 0.0)

# Metres per degree of latitude; good to ~0.1% anywhere outside the poles, which is
# far tighter than a synthetic track warrants.
M_PER_DEG_LAT = 111_320.0


def probe_video(path: str, exact: bool) -> tuple[int, float, int, int]:
    """Return (n_frames, fps, width, height) for the clip."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"[synth] cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Container metadata is occasionally missing or wrong. Decoding every frame is
    # the only way to be certain, so do it when asked or when the header is bogus.
    if exact or n_frames <= 0:
        n_frames = 0
        while cap.grab():
            n_frames += 1
        print(f"[synth] counted {n_frames} frames by decoding")

    cap.release()

    if fps <= 0:
        fps = 30.0
        print(f"[synth] video reports no fps -- assuming {fps}")
    if n_frames <= 0:
        sys.exit("[synth] video appears to have no frames")
    return n_frames, fps, width, height


def write_frames(path: str, n_frames: int, fps: float, t0_ns: int) -> int:
    """frame,timestamp_ns at a uniform cadence. Returns the last timestamp."""
    step_ns = int(round(1e9 / fps))
    with open(path, "w", newline="") as f:
        f.write("frame,timestamp_ns\n")
        for i in range(n_frames):
            f.write(f"{i},{t0_ns + i * step_ns}\n")
    return t0_ns + (n_frames - 1) * step_ns


def write_gps(path: str, t0_ns: int, t1_ns: int, hz: float,
              lat: float, lon: float, bearing_deg: float, speed_mps: float) -> None:
    """A straight constant-speed track, dead-reckoned from the start point.

    Real GPS arrives at ~1 Hz while frames arrive at ~30 Hz; the pipeline
    interpolates speed onto frame timestamps, so matching that cadence keeps the
    interpolation path exercised exactly as it is for a real drive.
    """
    step_ns = int(1e9 / hz)
    theta = math.radians(bearing_deg)
    with open(path, "w", newline="") as f:
        f.write("timestamp_ns,lat,lon,speed_mps,bearing_deg,accuracy_m\n")
        t = t0_ns
        while t <= t1_ns:
            dt_s = (t - t0_ns) / 1e9
            dist = speed_mps * dt_s
            cur_lat = lat + (dist * math.cos(theta)) / M_PER_DEG_LAT
            denom = M_PER_DEG_LAT * math.cos(math.radians(cur_lat))
            cur_lon = lon + (dist * math.sin(theta)) / (denom if abs(denom) > 1e-9 else 1e-9)
            f.write(f"{t},{cur_lat:.7f},{cur_lon:.7f},{speed_mps:.3f},{bearing_deg:.1f},5.0\n")
            t += step_ns


def write_xyz(path: str, header: str, t0_ns: int, t1_ns: int, hz: float,
              vec: tuple[float, float, float]) -> None:
    """A constant 3-axis sensor stream -- used for both gyro and gravity."""
    step_ns = int(1e9 / hz)
    x, y, z = vec
    with open(path, "w", newline="") as f:
        f.write(header + "\n")
        t = t0_ns
        while t <= t1_ns:
            f.write(f"{t},{x},{y},{z}\n")
            t += step_ns


def write_intrinsics(path: str, width: int, height: int, fov: float) -> None:
    """Frame size + FOV rather than invented fx/fy.

    ego_yaw.load_intrinsics accepts either explicit fx/fy/cx/cy or this form, and
    guessing a focal length we have not calibrated would be a fabrication dressed
    up as a measurement. This says plainly "assume a FOV" -- which is what it does.
    """
    with open(path, "w") as f:
        json.dump({"image_width": width, "image_height": height,
                   "fov_horizontal_deg": fov}, f, indent=2)


def place_video(src: str, dst: str, hardlink: bool) -> None:
    if hardlink:
        try:
            os.link(src, dst)
            print("[synth] hardlinked video (no extra disk used)")
            return
        except OSError as e:
            print(f"[synth] hardlink failed ({e}); copying instead")
    shutil.copy2(src, dst)


def adb_push(session_dir: str, session_id: str, serial: str | None) -> None:
    base = ["adb"] + (["-s", serial] if serial else [])

    devices = subprocess.run(base + ["devices"], capture_output=True, text=True)
    if devices.returncode != 0:
        sys.exit("[synth] adb not found on PATH -- install platform-tools or push by hand")
    attached = [ln for ln in devices.stdout.splitlines()[1:] if ln.strip().endswith("device")]
    if not attached:
        sys.exit("[synth] no device attached (check the USB cable and USB debugging)")

    dest = f"{ANDROID_SESSIONS}/"
    print(f"[synth] adb push -> {dest}{session_id}/")
    rc = subprocess.run(base + ["push", session_dir, dest]).returncode
    if rc != 0:
        sys.exit(f"[synth] adb push failed (exit {rc})")
    print("[synth] pushed. Open the app -- the Unfinished Upload dialog will offer it.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a complete 7-file session folder around a bare drive video.")
    ap.add_argument("video", help="the .mp4 to wrap")
    ap.add_argument("--out-dir", default=None,
                    help="parent folder for the session (default: <video dir>/synth_sessions)")
    ap.add_argument("--session-id", default=None,
                    help="session folder name (default: session_<epoch ms>, matching the app)")

    ap.add_argument("--speed-kmh", type=float, default=90.0,
                    help="constant ego speed written to gps.csv (default: 90)")
    ap.add_argument("--lat", type=float, default=32.0513233, help="track start latitude")
    ap.add_argument("--lon", type=float, default=34.9047184, help="track start longitude")
    ap.add_argument("--bearing", type=float, default=0.0,
                    help="constant heading in degrees, 0 = north (default: 0)")

    ap.add_argument("--gps-hz", type=float, default=1.0, help="gps.csv sample rate (default: 1)")
    ap.add_argument("--imu-hz", type=float, default=100.0,
                    help="gyro/gravity/linacc sample rate (default: 100)")
    ap.add_argument("--fov", type=float, default=Constants.FOV_HORIZONTAL_DEG,
                    help=f"horizontal FOV for intrinsics.json (default: {Constants.FOV_HORIZONTAL_DEG})")
    ap.add_argument("--gravity", default=",".join(str(v) for v in DEFAULT_GRAVITY),
                    help="gravity vector 'grx,gry,grz' (default: landscape mount)")

    ap.add_argument("--exact-frames", action="store_true",
                    help="decode the whole video to count frames instead of trusting the header")
    ap.add_argument("--hardlink", action="store_true",
                    help="hardlink the video instead of copying (same volume only)")
    ap.add_argument("--push", action="store_true", help="adb-push the folder to the phone")
    ap.add_argument("--serial", default=None, help="adb device serial (for multiple devices)")
    args = ap.parse_args()

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        sys.exit(f"[synth] no such video: {video}")

    size = os.path.getsize(video)
    if size < MIN_VIDEO_BYTES:
        sys.exit(f"[synth] video is {size / 1e6:.1f} MB -- the app rejects anything under "
                 f"{MIN_VIDEO_BYTES // (1024 * 1024)} MB (PostDriveActivity.MIN_VIDEO_BYTES), "
                 f"and SessionStore will not even list it. Use a longer clip.")
    if size > MAX_VIDEO_BYTES:
        sys.exit(f"[synth] video is {size / 1e9:.1f} GB -- over the 4 GB server limit.")

    try:
        gx, gy, gz = (float(v) for v in args.gravity.split(","))
    except ValueError:
        sys.exit("[synth] --gravity must be three comma-separated numbers, e.g. '9.81,0,0'")

    n_frames, fps, width, height = probe_video(video, args.exact_frames)
    duration_s = n_frames / fps

    session_id = args.session_id or f"session_{int(time.time() * 1000)}"
    parent = args.out_dir or os.path.join(os.path.dirname(video), "synth_sessions")
    session_dir = os.path.join(parent, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Sensors must bracket the frame timestamps: the yaw-rate step bins samples into
    # per-frame windows and the GPS step interpolates, so a stream that stops short
    # of the last frame silently degrades to zeros at the tail.
    t0_ns = int(time.time() * 1e9)
    pad_ns = 500_000_000
    t_last = write_frames(os.path.join(session_dir, "frames.csv"), n_frames, fps, t0_ns)

    write_gps(os.path.join(session_dir, "gps.csv"), t0_ns - pad_ns, t_last + pad_ns,
              args.gps_hz, args.lat, args.lon, args.bearing, args.speed_kmh / 3.6)
    write_xyz(os.path.join(session_dir, "gyro.csv"), "timestamp_ns,gx,gy,gz",
              t0_ns - pad_ns, t_last + pad_ns, args.imu_hz, (0.0, 0.0, 0.0))
    write_xyz(os.path.join(session_dir, "gravity.csv"), "timestamp_ns,grx,gry,grz",
              t0_ns - pad_ns, t_last + pad_ns, args.imu_hz, (gx, gy, gz))
    write_xyz(os.path.join(session_dir, "linacc.csv"), "timestamp_ns,ax,ay,az",
              t0_ns - pad_ns, t_last + pad_ns, args.imu_hz, (0.0, 0.0, 0.0))
    write_intrinsics(os.path.join(session_dir, "intrinsics.json"), width, height, args.fov)

    # Exactly one .mp4 must sit in the folder: SessionStore.findPendingSessions takes
    # the FIRST match it finds, so a stray second video would be picked at random.
    dst_video = os.path.join(session_dir, os.path.basename(video))
    if not os.path.exists(dst_video):
        place_video(video, dst_video, args.hardlink)

    print()
    print(f"[synth] session : {session_id}")
    print(f"[synth] folder  : {session_dir}")
    print(f"[synth] video   : {width}x{height}, {n_frames} frames @ {fps:.2f} fps "
          f"({duration_s:.1f}s, {size / 1e6:.0f} MB)")
    print(f"[synth] ego     : {args.speed_kmh:.0f} km/h straight, bearing {args.bearing:.0f} deg, "
          f"from {args.lat:.5f},{args.lon:.5f}")
    print()
    for name in ("frames.csv", "gps.csv", "gyro.csv", "gravity.csv", "linacc.csv",
                 "intrinsics.json", os.path.basename(video)):
        p = os.path.join(session_dir, name)
        print(f"          {name:<16} {os.path.getsize(p):>12,} bytes")

    if args.push:
        print()
        adb_push(session_dir, session_id, args.serial)
    else:
        print()
        print("[synth] to put it on the phone:")
        print(f"          adb push \"{session_dir}\" {ANDROID_SESSIONS}/")


if __name__ == "__main__":
    main()


# ── What the numbers mean ────────────────────────────────────────────────────
#
# The pipeline reconstructs a tracked vehicle's WORLD speed as roughly
# "ego speed + closing speed measured from the video". Only the second term comes
# from the footage; the first is whatever --speed-kmh says. So:
#
#   * Detection, tracking, distance, plate reading, clip cutting, the violation
#     records and the whole upload/queue/worker/dashboard chain are all real and
#     exercised end to end.
#   * The absolute km/h on each violation is real-relative-speed offset by a
#     number you invented. If you set --speed-kmh 90 for a clip filmed at 60, every
#     vehicle reads ~30 km/h too fast, and whether anything trips the speed
#     threshold is partly your choice of flag.
#   * Ego heading is a straight line because the gyro is zeros. On footage that
#     actually turns, world-frame reconstruction will drift through the corner.
#
# That is fine for demonstrating that the system works, and misleading if quoted
# as an accuracy result. Set --speed-kmh to your honest best guess of how fast the
# car in the footage was going, and say the sensors were synthesised whenever you
# show the output.
