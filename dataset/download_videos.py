"""
Download categorized dashcam videos from YouTube for YOLOv8 training dataset.
Uses yt-dlp to download MP4s with sanitized filenames into category subfolders.
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
    results = []

    for url in urls:
        print(f"\n[{category}] Downloading: {url}")
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "--restrict-filenames",          # sanitize filenames (no spaces/special chars)
            "--output", str(output_dir / "%(title)s.%(ext)s"),
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=False)
        success = result.returncode == 0
        results.append((url, success))
        if not success:
            print(f"  WARNING: Failed to download {url}")

    return results


def check_large_files(threshold_mb: float = 95.0) -> list[tuple[Path, float]]:
    large = []
    for f in BASE_DIR.rglob("*.mp4"):
        size_mb = f.stat().st_size / (1024 * 1024)
        if size_mb > threshold_mb:
            large.append((f, size_mb))
    return large


def main():
    all_results = []
    for category, urls in URLS.items():
        results = download_category(category, urls)
        all_results.extend(results)

    failed = [(url, ok) for url, ok in all_results if not ok]

    print("\n" + "=" * 60)
    print(f"Download complete. {len(all_results) - len(failed)}/{len(all_results)} succeeded.")

    if failed:
        print("\nFailed URLs:")
        for url, _ in failed:
            print(f"  {url}")

    large_files = check_large_files()
    if large_files:
        print("\n*** FILES EXCEEDING 95 MB (GitHub 100 MB hard limit) ***")
        for path, size_mb in sorted(large_files, key=lambda x: x[1], reverse=True):
            print(f"  {path.relative_to(BASE_DIR.parent)}  ({size_mb:.1f} MB)")
        print("\nThese files CANNOT be pushed to GitHub without Git LFS.")
        print("Options: (1) enable Git LFS, (2) trim videos, (3) exclude from git.")
    else:
        print("\nAll files are under 95 MB — safe to push without Git LFS.")


if __name__ == "__main__":
    main()
