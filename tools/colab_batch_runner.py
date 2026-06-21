"""
Colab batch runner -- "hit Play and walk away".

Runs the integrated main.py over EVERY test video, once per detector (v8m, y11x),
with the CORRECT main.py command line. This exists because hand-written batch cells
kept guessing the wrong flags (main.py takes the video as the FIRST POSITIONAL arg and
`--model <weights-file>`, NOT `--video` / `--detector <key>`), so every run failed with
returncode 1 and the real error was swallowed by capture_output.

What it does per video:
  1. (optional) normalize the clip with ffmpeg to local NVMe (/content) -- fixes the
     odd-codec / VFR dashcam files that cv2 can't open straight off Drive.
  2. run main.py <clip> --model <weights> --colab [--benchmark] --out-dir <per-video/det>
  3. on failure, PRINT the tail of the captured log so you see the real traceback
     instead of a bare "[ERROR]".
  4. skip a (video, detector) that already produced its CSV, so a re-run resumes.

Paste into a Colab cell:

    !cd /content/drive/MyDrive/malshinon_master && python tools/colab_batch_runner.py

Toggle the CONFIG block below (BENCHMARK for CSV-only fast runs, NORMALIZE off if your
clips already open fine, MAX_FRAMES for a quick smoke test).
"""
from __future__ import annotations

import os
import sys
import glob
import time
import shutil
import subprocess


# ============================================================================ #
# CONFIG
# ============================================================================ #
DRIVE_ROOT = "/content/drive/MyDrive/malshinon_master"
VIDEOS_BASE = os.path.join(DRIVE_ROOT, "tests_videos")        # searched recursively for *.mp4
OUTPUT_DIR = os.path.join(DRIVE_ROOT, "batch_outputs")        # per-video/detector results land here
LANE_WEIGHTS = os.path.join(DRIVE_ROOT, "weights", "phase3_v3_yellowprotect.pt")

# detector key -> the weights file main.py's --model expects (both auto-download).
DETECTORS = {"v8m": "yolov8m.pt", "y11x": "yolo11x.pt"}

BENCHMARK = True        # True -> --benchmark (CSV/text only, skips the slow annotated render)
NORMALIZE = False       # VideoHandler now decodes exotic codecs itself (cv2 -> ffmpeg stream),
                        # so the slow upfront transcode is off by default. Flip True only if a
                        # specific clip still won't open.
IMGSZ = 1280            # YOLO inference size (1984 = main default but slow); set None to use default
MAX_FRAMES = 0          # 0 = whole video; e.g. 300 for a quick smoke test
SUBPROCESS_TIMEOUT = 60 * 45    # per (video, detector) hard cap, seconds
LOG_TAIL_LINES = 25     # how many trailing log lines to print when a run fails


# ============================================================================ #
def find_videos() -> list[str]:
    vids: list[str] = []
    for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv"):
        vids += glob.glob(os.path.join(VIDEOS_BASE, "**", ext), recursive=True)
    # drop our own annotated outputs so we never reprocess a result as if it were input
    vids = [v for v in vids if "_annotated" not in os.path.basename(v)]
    return sorted(set(vids))


def normalize(video: str) -> str:
    """Transcode to a clean H.264 mp4 on local NVMe; return that path (or the original on failure)."""
    if not NORMALIZE:
        return video
    local = "/content/_rg_input.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-c:v", "libx264", "-preset", "ultrafast",
         "-an", local],
        capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(local):
        print(f"    [warn] ffmpeg normalize failed; using original. "
              f"stderr tail:\n      {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '(none)'}")
        return video
    return local


def already_done(det_out: str, name: str) -> bool:
    # main.py always writes <name>_vehicles.csv; treat its presence as "this run finished".
    return os.path.isfile(os.path.join(det_out, f"{name}_vehicles.csv"))


def run_one(clip: str, name: str, det_key: str, weights: str) -> bool:
    det_out = os.path.join(OUTPUT_DIR, name, det_key)
    os.makedirs(det_out, exist_ok=True)
    if already_done(det_out, name):
        print(f"    [{det_key}] skip (already done)")
        return True

    cmd = [sys.executable, os.path.join(DRIVE_ROOT, "main.py"),
           clip,                                  # FIRST POSITIONAL = the video (this was the bug)
           "--model", weights,                    # NOT --detector; --model wants a weights file
           "--colab",                             # headless + smart Drive staging
           "--out-dir", det_out]
    if os.path.isfile(LANE_WEIGHTS):
        cmd += ["--lane-weights", LANE_WEIGHTS]   # absent -> main.py just disables the yellow rule
    if BENCHMARK:
        cmd += ["--benchmark"]
    if IMGSZ:
        cmd += ["--imgsz", str(IMGSZ)]
    if MAX_FRAMES:
        cmd += ["--max-frames", str(MAX_FRAMES)]

    log_path = os.path.join(det_out, "run.log")
    print(f"    [{det_key}] main.py -> {det_out}")
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            r = subprocess.run(cmd, cwd=DRIVE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=SUBPROCESS_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"    [{det_key}] TIMEOUT after {SUBPROCESS_TIMEOUT}s (see {log_path})")
            return False
    if r.returncode != 0:
        print(f"    [{det_key}] FAILED rc={r.returncode} -- last {LOG_TAIL_LINES} log lines:")
        try:
            with open(log_path, encoding="utf-8") as fh:
                for line in fh.readlines()[-LOG_TAIL_LINES:]:
                    print("      " + line.rstrip())
        except OSError:
            pass
        return False
    return True


def main() -> None:
    if not os.path.isfile(os.path.join(DRIVE_ROOT, "main.py")):
        print(f"[abort] no main.py in {DRIVE_ROOT} -- is DRIVE_ROOT correct / Drive mounted?")
        return
    if not os.path.isfile(LANE_WEIGHTS):
        print(f"[warn] lane weights not found ({LANE_WEIGHTS}); yellow-line rule will be disabled")

    videos = find_videos()
    if not videos:
        print(f"[abort] no videos found under {VIDEOS_BASE}")
        return
    print(f"[run] {len(videos)} videos x {len(DETECTORS)} detectors -> {OUTPUT_DIR}")
    print(f"[run] benchmark={BENCHMARK} normalize={NORMALIZE} imgsz={IMGSZ} max_frames={MAX_FRAMES}\n")

    t0 = time.time()
    ok = fail = 0
    for i, video in enumerate(videos, 1):
        name = os.path.splitext(os.path.basename(video))[0]
        print(f"[{i}/{len(videos)}] {name}")
        clip = normalize(video)
        for det_key, weights in DETECTORS.items():
            if run_one(clip, name, det_key, weights):
                ok += 1
            else:
                fail += 1

    if NORMALIZE and os.path.isfile("/content/_rg_input.mp4"):
        os.remove("/content/_rg_input.mp4")
    print(f"\n[done] {ok} ok, {fail} failed, in {(time.time()-t0)/60:.1f} min -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
