"""
Download categorized dashcam videos from YouTube for YOLOv8 training.
Saves to raw_videos/<category>/ — this folder is gitignored.
Requires: pip install yt-dlp  (and ffmpeg on PATH for merged audio+video)
"""

import subprocess
import sys
from pathlib import Path

URLS = {
    "crossing_solid_line": [
        "https://www.youtube.com/watch?v=B7-3EuQFAZM",
        "https://www.youtube.com/watch?v=0SmdindPVEY",
        "https://www.youtube.com/watch?v=JqDMXT4Eue4",
        "https://www.youtube.com/watch?v=Rj2yWbv1934",
        "https://www.youtube.com/watch?v=NFTpZcN4UsU",
    ],
    "red_light": [
        "https://www.youtube.com/watch?v=zkjYAH3a6Ck",
        "https://www.youtube.com/watch?v=Q4a11adqIMk",
        "https://www.youtube.com/watch?v=TTHsdKvzK-U",
        "https://www.youtube.com/watch?v=Oo5-TVQcIqc",
        "https://www.youtube.com/watch?v=U8RA2mnVc6E",
        "https://www.youtube.com/watch?v=iw6Iv02DOig",
        "https://www.youtube.com/watch?v=iAsfdfzuSTY",
        "https://www.youtube.com/watch?v=WBftrM1TyF8",
        "https://www.youtube.com/watch?v=Tq-3_lLRBxQ",
        "https://www.youtube.com/watch?v=8kVnM5HzbFQ",
    ],
    "normal_driving": [
        "https://www.youtube.com/watch?v=3KpV5GRLkJc",
        "https://www.youtube.com/watch?v=jErBH3onWZs",
        "https://www.youtube.com/watch?v=JaZW2e6lTpk",
    ],
}

BASE_DIR = Path(__file__).parent / "raw_videos"


def download_category(category: str, urls: list[str]) -> list[tuple[str, bool]]:
    output_dir = BASE_DIR / category
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for url in urls:
        print(f"\n[{category}] {url}")
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--restrict-filenames",
            "--output", str(output_dir / "%(id)s_%(title)s.%(ext)s"),
            "--no-playlist",
            url,
        ]
        ok = subprocess.run(cmd).returncode == 0
        results.append((url, ok))
        if not ok:
            print(f"  WARNING: failed")
    return results


def main():
    results = []
    for category, urls in URLS.items():
        results.extend(download_category(category, urls))

    failed = [u for u, ok in results if not ok]
    print(f"\n{'='*50}")
    print(f"Done: {len(results)-len(failed)}/{len(results)} succeeded.")
    if failed:
        print("Failed:")
        for u in failed:
            print(f"  {u}")


if __name__ == "__main__":
    main()
