"""
Run main.py on every downloaded video and produce annotated output files.
Run from the project root: python tests_videos/run_all.py [category]

Optional category filter: crossing_solid_line | red_light | normal_driving
If omitted, processes all categories.

Outputs per video (saved next to the source video):
  <stem>_annotated.mp4   — video with YOLO bounding boxes
  <stem>_yolo.jsonl      — per-frame YOLO log
  <stem>_world.jsonl     — per-frame world-state log
"""

import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw_videos"
MAIN_PY = Path(__file__).parent.parent / "main.py"

ALL_CATEGORIES = ["crossing_solid_line", "red_light", "normal_driving"]


def run_on_category(category: str):
    cat_dir = RAW_DIR / category
    if not cat_dir.exists():
        print(f"[SKIP] {category} — folder not found (run download_videos.py first)")
        return

    videos = sorted(cat_dir.glob("*.mp4"))
    videos = [v for v in videos if not v.name.endswith("_annotated.mp4")]

    if not videos:
        print(f"[SKIP] {category} — no .mp4 files found")
        return

    print(f"\n=== {category} ({len(videos)} videos) ===")
    for video in videos:
        annotated = video.parent / f"{video.stem}_annotated.mp4"
        if annotated.exists():
            print(f"  [SKIP] {video.name} — annotated already exists")
            continue
        print(f"  Processing: {video.name}")
        result = subprocess.run([sys.executable, str(MAIN_PY), str(video)])
        if result.returncode != 0:
            print(f"  [FAIL] {video.name}")
        else:
            print(f"  [OK]   -> {annotated.name}")


def main():
    categories = sys.argv[1:] if len(sys.argv) > 1 else ALL_CATEGORIES
    unknown = [c for c in categories if c not in ALL_CATEGORIES]
    if unknown:
        print(f"Unknown categories: {unknown}. Valid: {ALL_CATEGORIES}")
        sys.exit(1)

    for cat in categories:
        run_on_category(cat)

    print("\nDone.")


if __name__ == "__main__":
    main()
