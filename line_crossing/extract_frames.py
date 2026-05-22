"""
Extract representative frames from each crossing_solid_line video.

Picks frames at 10%, 40%, and 70% through the video to sample
different road conditions (early/mid/late scene variety).

Usage (from project root):
    python line_crossing/extract_frames.py

Output: test_frames/<video_stem>_f<N>.jpg
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_DIR = Path(__file__).parent.parent / "tests_videos" / "raw_videos" / "crossing_solid_line"
OUT_DIR   = Path(__file__).parent.parent / "test_frames"
SAMPLE_POSITIONS = [0.10, 0.40, 0.70]   # fraction through the video


def extract(video_path: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        print(f"  [SKIP] {video_path.name} — could not read frame count")
        cap.release()
        return

    stem = video_path.stem
    saved = 0
    for frac in SAMPLE_POSITIONS:
        frame_idx = int(total * frac)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  [WARN] {stem}: could not read frame {frame_idx}")
            continue

        # Tag brightness so we know which are dark/bright
        gray_mean = int(frame.mean())
        out_name = f"{stem}_f{frame_idx}_bright{gray_mean}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), frame)
        print(f"  {out_name}  (mean brightness={gray_mean})")
        saved += 1

    cap.release()
    print(f"  -> {saved} frames saved from {video_path.name}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    videos = [v for v in videos if "_annotated" not in v.name]

    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        sys.exit(1)

    print(f"Extracting frames from {len(videos)} videos -> {OUT_DIR}\n")
    for v in videos:
        print(f"[{v.name}]")
        extract(v, OUT_DIR)

    all_frames = list(OUT_DIR.glob("*.jpg"))
    print(f"\nDone. {len(all_frames)} frames total in {OUT_DIR}/")


if __name__ == "__main__":
    main()
