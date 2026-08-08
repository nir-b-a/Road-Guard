# Model Weights — Download Guide

> **Just run `python tools/fetch_models.py`.** It downloads whatever is missing and verifies
> every weight against a known SHA-256. `--check` verifies without touching the network;
> `--list` prints the inventory. The rest of this file explains what it does.

Large weights are gitignored; the three small custom ones are committed, so a fresh clone can
run the full pipeline without any private credentials.

---

## Weights required to run the pipeline

| File | Purpose | Auto-downloaded? |
|------|---------|-----------------|
| `yolo11x.pt` | Vehicle + traffic light detector (main tracker) | Yes — Ultralytics downloads on first run |
| `weights/phase3_v3_yellowprotect.pt` | **Committed in git** — lane-seg model (yellow + solid line) | Already in repo |
| `israeli_plates.pt` | **Committed in git** — Israeli license plate detector (FastALPR) | Already in repo |
| `models/tire_yolo11n.pt` | **Committed in git** — tire detector for the Stage-2 cascade (solid-line FP filter) | Already in repo |
| FastALPR detector + OCR | Plate location and text recognition (two ONNX models) | Yes — `fast_alpr` downloads on first use |

---

## `models/tire_yolo11n.pt`

Our custom-trained YOLO11n tire detector, used by the Stage-2 cascade that filters solid-line
crossing false positives. At 5.5 MB it is **committed to the repo** (see the re-include rules in
`.gitignore`), so no Roboflow API key is needed to run the pipeline.

The Stage-2 cascade **is** wired into `main.py`. Without this weight the cascade degrades to
Stage-1 only, which means noticeably more false positives on solid-line crossings — so treat a
`fetch_models.py` failure on it as a real problem, not a warning.

The original training data lives on **Roboflow** under the Road Guard workspace; that is only
needed to *retrain*, never to run.

---

## Notes

- `yolo11x.pt` is auto-downloaded by Ultralytics to the current directory on first run — no manual step needed.
- The lane-seg model (`weights/phase3_v3_yellowprotect.pt`), plate detector (`israeli_plates.pt`) and tire detector (`models/tire_yolo11n.pt`) are committed and require no download.
- Never commit `.pt` / `.pth` files — they are all gitignored except the three small re-included weights above.
- Provenance of every model (who trained it, on what data, under which licence) is documented in [`docs/EXTERNAL_COMPONENTS.md`](docs/EXTERNAL_COMPONENTS.md).
