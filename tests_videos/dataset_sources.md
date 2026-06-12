# Dataset Video Sources

Categorized YouTube dashcam videos for YOLOv8 violation detection training.  
All 18 URLs were verified live as of 2026-05-22.

---

## crossing_solid_line (5 videos)

| Duration | URL |
|----------|-----|
| 0:16 | https://www.youtube.com/watch?v=B7-3EuQFAZM |
| 0:23 | https://www.youtube.com/watch?v=0SmdindPVEY |
| 0:28 | https://www.youtube.com/watch?v=JqDMXT4Eue4 |
| 0:22 | https://www.youtube.com/watch?v=Rj2yWbv1934 |
| 0:16 | https://www.youtube.com/watch?v=NFTpZcN4UsU |

## red_light (10 videos)

| Duration | URL |
|----------|-----|
| 0:20 | https://www.youtube.com/watch?v=zkjYAH3a6Ck |
| 0:21 | https://www.youtube.com/watch?v=Q4a11adqIMk |
| 0:38 | https://www.youtube.com/watch?v=TTHsdKvzK-U |
| 0:40 | https://www.youtube.com/watch?v=Oo5-TVQcIqc |
| 0:10 | https://www.youtube.com/watch?v=U8RA2mnVc6E |
| 0:16 | https://www.youtube.com/watch?v=iw6Iv02DOig |
| 0:10 | https://www.youtube.com/watch?v=iAsfdfzuSTY |
| 0:16 | https://www.youtube.com/watch?v=WBftrM1TyF8 |
| 0:04 | https://www.youtube.com/watch?v=Tq-3_lLRBxQ |
| 0:14 | https://www.youtube.com/watch?v=8kVnM5HzbFQ |

## normal_driving (3 videos)

| Duration | URL |
|----------|-----|
| 10:01 | https://www.youtube.com/watch?v=3KpV5GRLkJc |
| 10:01 | https://www.youtube.com/watch?v=jErBH3onWZs |
| 10:01 | https://www.youtube.com/watch?v=JaZW2e6lTpk |

---

## How to Download

### Requirements

Install dependencies:
```bash
pip install yt-dlp
```

For merged video+audio (recommended), also install **ffmpeg**:
- Windows: `winget install ffmpeg` or download from https://ffmpeg.org/download.html
- Make sure `ffmpeg` is on your PATH

### Run the download script

```bash
cd tests_videos/
python download_videos.py
```

Videos will be saved to:
```
tests_videos/
└── raw_videos/
    ├── crossing_solid_line/
    ├── red_light/
    └── normal_driving/
```

> The `raw_videos/` folder is in `.gitignore` — videos stay local only and are never committed to the repo.
