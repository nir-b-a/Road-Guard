# Road Guard — Final Project

Traffic violation detection system using YOLOv8 and dashcam footage.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Video sources and download instructions are in [tests_videos/dataset_sources.md](tests_videos/dataset_sources.md).

To download all videos locally:
```bash
cd tests_videos/
python download_videos.py
```

Videos are saved to `tests_videos/raw_videos/` (gitignored — never committed to the repo).
