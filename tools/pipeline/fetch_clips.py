"""
Phase 1 -- fetch the curated "violation + visible plate" clips via yt-dlp.

Saves to tests_videos/raw_videos/solid_line_crossing_with_pl/<video_id>.mp4 so each file's stem
is the clip prefix the rest of the pipeline keys on. (Repo uses tests_videos, not test_videos.)
"""
from __future__ import annotations

import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEST = os.path.join(_REPO, "tests_videos", "raw_videos", "solid_line_crossing_with_pl")

CLIPS = [
    "https://www.youtube.com/shorts/0fr98nSc3CA",
    "https://www.youtube.com/shorts/7E35VSQbAH8",
    "https://www.youtube.com/watch?v=8JerQB_w8D4",
    "https://www.youtube.com/watch?v=loXvNaf7plw",
    "https://www.youtube.com/watch?v=-0Jmlzkie9s",
]


def fetch(urls: list[str] = CLIPS, dest: str = DEST) -> list[str]:
    """Download each URL as <id>.mp4 into dest. Returns the list of resulting mp4 filenames."""
    import yt_dlp

    os.makedirs(dest, exist_ok=True)
    opts = {
        # prefer a single progressive mp4; else merge best video+audio to mp4
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(dest, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "ignoreerrors": True,          # one bad URL must not abort the rest
        "retries": 3,
        "quiet": False,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download(urls)

    got = sorted(f for f in os.listdir(dest) if f.lower().endswith(".mp4"))
    print(f"[fetch] {len(got)}/{len(urls)} mp4(s) in {dest}:")
    for f in got:
        print(f"   {f}")
    return got


if __name__ == "__main__":
    fetch()
