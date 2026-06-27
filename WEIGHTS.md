# Model Weights — Download Guide

All weights are gitignored (too large for git). Download them manually before running the pipeline.

---

## Weights required to run the pipeline

| File | Purpose | Auto-downloaded? |
|------|---------|-----------------|
| `yolo11x.pt` | Vehicle + traffic light detector (main tracker) | Yes — Ultralytics downloads on first run |
| `weights/phase3_v3_yellowprotect.pt` | **Committed in git** — lane-seg model (yellow + solid line) | Already in repo |
| `israeli_plates.pt` | **Committed in git** — Israeli license plate detector (FastALPR) | Already in repo |
| `models/tire_yolo11n.pt` | Tire detector for Stage-2 cascade (solid-line FP filter) | **Manual — see below** |

---

## Downloading `models/tire_yolo11n.pt`

This is our custom-trained YOLOv11n tire detector. It is stored on **Roboflow** under the Road Guard workspace.

```bash
# Option A — download from Roboflow (requires API key in .env)
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
model = rf.workspace("road-guard").project("tire-detection").version(1).model
model.download("yolov8", location="models/")
# rename downloaded best.pt -> models/tire_yolo11n.pt

# Option B — copy from a teammate's machine
# The file lives at models/tire_yolo11n.pt (5.3 MB)
# Transfer via USB / shared Drive / Slack
```

> If the cascade is disabled (not yet wired into main.py), this weight is not needed for a standard run.

---

## Notes

- `yolo11x.pt` is auto-downloaded by Ultralytics to the current directory on first run — no manual step needed.
- The lane-seg model (`weights/phase3_v3_yellowprotect.pt`) and plate detector (`israeli_plates.pt`) are committed and require no download.
- Never commit `.pt` / `.pth` files — they are all gitignored except the two small re-included weights above.
