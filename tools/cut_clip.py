"""
cut_clip.py -- cut an Android dashcam capture (video + synchronized sensor CSVs)
down to a shorter time window, keeping EVERY stream consistent.

A capture folder (see ego_yaw.py / the Android data spec) holds:
    <video>.mp4        the footage
    frames.csv         frame, timestamp_ns         (decoded frame i -> capture time)
    gyro.csv           timestamp_ns, gx, gy, gz                    (rad/s)
    gravity.csv        timestamp_ns, grx, gry, grz                 (m/s^2)
    linacc.csv         timestamp_ns, ax, ay, az        (optional;  m/s^2)
    gps.csv            timestamp_ns, lat, lon, speed_mps, bearing_deg, accuracy_m
    intrinsics.json    camera intrinsics            (copied verbatim)
    tags.json          (copied verbatim)

Everything is aligned by timestamp_ns on ONE monotonic clock, and the speed
pipeline pairs decoded video frame i with frames.csv row i. So a correct cut must:

  * pick a frame window [N0, N1] from frames.csv for the requested time span;
  * extract EXACTLY those video frames, in order (frame-accurate, no dup/drop);
  * RE-INDEX frames.csv so `frame` restarts at 0 (the cut video decodes from 0),
    while keeping the ORIGINAL absolute timestamp_ns -- the sensor streams are
    matched against those same absolute values, so rebasing them would break
    alignment unless every stream were shifted by the identical offset;
  * keep the sensor rows whose timestamp_ns lands in the window (+/- a guard, so
    the edge frames still have neighbours to interpolate / average from);
  * copy intrinsics.json / tags.json unchanged.

Usage:
    python tools/cut_clip.py SRC_DIR OUT_DIR
           [--start-sec 0] [--dur-sec 60] [--guard-sec 0.5] [--video name.mp4]
           [--crf 18]

start-sec is measured from the FIRST frame's timestamp. The video is re-encoded
(H.264) because stream-copy can only cut on keyframes, which would not be
frame-accurate against frames.csv.
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys


# Sensor streams: filename -> timestamp column. Each is filtered by timestamp.
SENSOR_FILES = {
    "gyro.csv":    "timestamp_ns",
    "gravity.csv": "timestamp_ns",
    "linacc.csv":  "timestamp_ns",
    "gps.csv":     "timestamp_ns",
}
COPY_VERBATIM = ("intrinsics.json", "tags.json")


def find_source_video(src_dir: str, override: str | None) -> str:
    """Return the source .mp4. Prefers an explicit --video; otherwise the single
    .mp4 in the folder, ignoring pipeline outputs (*_annotated.mp4)."""
    if override:
        p = override if os.path.isabs(override) else os.path.join(src_dir, override)
        if not os.path.exists(p):
            sys.exit(f"[cut_clip] --video not found: {p}")
        return p
    cands = [p for p in glob.glob(os.path.join(src_dir, "*.mp4"))
             if not p.endswith("_annotated.mp4")]
    if not cands:
        sys.exit(f"[cut_clip] no source .mp4 in {src_dir}")
    if len(cands) > 1:
        sys.exit(f"[cut_clip] multiple .mp4 in {src_dir}; pass --video <name.mp4>:\n  "
                 + "\n  ".join(os.path.basename(c) for c in cands))
    return cands[0]


def read_frames(frames_csv: str) -> list[tuple[int, int]]:
    """[(frame, timestamp_ns), ...] sorted by frame index."""
    rows: list[tuple[int, int]] = []
    with open(frames_csv, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["frame"]), int(r["timestamp_ns"])))
            except (KeyError, ValueError):
                continue
    rows.sort()
    if not rows:
        sys.exit(f"[cut_clip] no usable rows in {frames_csv}")
    return rows


def write_reindexed_frames(out_csv: str, kept: list[tuple[int, int]]) -> None:
    """frame re-indexed 0..len-1; timestamp_ns kept ABSOLUTE (sensor clock)."""
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_ns"])
        for new_i, (_, ts) in enumerate(kept):
            w.writerow([new_i, ts])


def filter_sensor(src: str, dst: str, t_col: str, lo: int, hi: int) -> tuple[int, int]:
    """Copy header + rows with lo <= timestamp_ns <= hi. Returns (kept, total)."""
    kept = total = 0
    with open(src, newline="") as fi, open(dst, "w", newline="") as fo:
        reader = csv.reader(fi)
        writer = csv.writer(fo)
        header = next(reader, None)
        if header is None:
            return 0, 0
        writer.writerow(header)
        try:
            tidx = header.index(t_col)
        except ValueError:
            sys.exit(f"[cut_clip] {os.path.basename(src)} has no '{t_col}' column")
        for row in reader:
            if not row:
                continue
            total += 1
            try:
                ts = int(row[tidx])
            except (IndexError, ValueError):
                continue
            if lo <= ts <= hi:
                writer.writerow(row)
                kept += 1
    return kept, total


def cut_video(src_video: str, dst_video: str, n0: int, n1: int, crf: int) -> None:
    """Extract original decoded frames [n0, n1] inclusive, frame-accurate.

    `select=between(n,n0,n1)` keys off the decoder's frame counter (0-based from
    the start, so NO input -ss -- that would reset the counter). `fps_mode
    passthrough` forbids frame dup/drop, and `setpts=PTS-STARTPTS` rebases the
    output clock to 0. Re-encoded so the cut is exact rather than keyframe-snapped.
    """
    vf = r"select=between(n\,%d\,%d),setpts=PTS-STARTPTS" % (n0, n1)
    cmd = [
        "ffmpeg", "-y", "-i", src_video,
        "-vf", vf,
        "-fps_mode", "passthrough",
        "-an",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        dst_video,
    ]
    print("[cut_clip] ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def count_video_frames(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        check=True, capture_output=True, text=True)
    return int(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description="Cut an Android capture (video + sensors) to a time window.")
    ap.add_argument("src_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--start-sec", type=float, default=0.0, help="window start, seconds from first frame")
    ap.add_argument("--dur-sec", type=float, default=60.0, help="window length in seconds")
    ap.add_argument("--guard-sec", type=float, default=0.5, help="extra sensor margin each edge")
    ap.add_argument("--video", default=None, help="source mp4 name (auto-detected if omitted)")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality (lower=better, 18~visually lossless)")
    args = ap.parse_args()

    src_dir = os.path.abspath(args.src_dir)
    out_dir = os.path.abspath(args.out_dir)
    if os.path.abspath(out_dir) == src_dir:
        sys.exit("[cut_clip] refuse to write into the source directory; choose a new OUT_DIR")
    os.makedirs(out_dir, exist_ok=True)

    src_video = find_source_video(src_dir, args.video)
    frames_csv = os.path.join(src_dir, "frames.csv")
    if not os.path.exists(frames_csv):
        sys.exit(f"[cut_clip] missing {frames_csv}")

    # --- pick the frame window from frames.csv ------------------------------
    rows = read_frames(frames_csv)
    t_first = rows[0][1]
    win_start = t_first + int(args.start_sec * 1e9)
    win_end = win_start + int(args.dur_sec * 1e9)
    kept = [(fr, ts) for fr, ts in rows if win_start <= ts < win_end]
    if not kept:
        sys.exit(f"[cut_clip] no frames in [{args.start_sec}, "
                 f"{args.start_sec + args.dur_sec}] s of this clip")
    n0, n1 = kept[0][0], kept[-1][0]
    count = n1 - n0 + 1
    span_s = (kept[-1][1] - kept[0][1]) / 1e9
    guard = int(args.guard_sec * 1e9)
    sens_lo, sens_hi = win_start - guard, kept[-1][1] + guard

    print(f"[cut_clip] source video : {os.path.basename(src_video)}")
    print(f"[cut_clip] frame window : {n0}..{n1}  ({count} frames, {span_s:.3f} s)")
    print(f"[cut_clip] sensor window: [{sens_lo}, {sens_hi}] ns "
          f"(+/-{args.guard_sec}s guard)")

    # --- frames.csv (re-indexed, absolute timestamps) -----------------------
    write_reindexed_frames(os.path.join(out_dir, "frames.csv"), kept)
    print(f"[cut_clip] frames.csv   : {count} rows re-indexed 0..{count - 1}")

    # --- sensor streams -----------------------------------------------------
    for name, t_col in SENSOR_FILES.items():
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            print(f"[cut_clip] {name:12s}: (absent, skipped)")
            continue
        k, tot = filter_sensor(src, os.path.join(out_dir, name), t_col, sens_lo, sens_hi)
        print(f"[cut_clip] {name:12s}: kept {k}/{tot} rows")

    # --- verbatim json ------------------------------------------------------
    for name in COPY_VERBATIM:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
            print(f"[cut_clip] {name:14s}: copied")

    # --- video (frame-accurate) ---------------------------------------------
    dst_video = os.path.join(out_dir, os.path.basename(src_video))
    cut_video(src_video, dst_video, n0, n1, args.crf)

    # --- verify frame counts agree ------------------------------------------
    got = count_video_frames(dst_video)
    status = "OK" if got == count else "MISMATCH"
    print(f"[cut_clip] verify       : video has {got} frames, frames.csv has {count} -> {status}")
    if got != count:
        sys.exit("[cut_clip] FRAME COUNT MISMATCH -- the cut video and frames.csv "
                 "are not aligned; do not use this output.")
    size_mb = os.path.getsize(dst_video) / 1e6
    print(f"[cut_clip] done -> {out_dir}  (video {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
