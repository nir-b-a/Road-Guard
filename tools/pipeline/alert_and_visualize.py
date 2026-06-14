"""
Downstream consumer: alert + visual proof for finished violations.

Takes the original video, the per-frame cache JSON (for bboxes), and the final violations JSON
(from run_pipeline.py). For each violation it:
  1. TRIGGERS AN ALERT -- prints a structured JSON payload to the console (simulating an API send)
     with the plate, the LR confidence, and the (bilingual) reason.
  2. RENDERS VISUAL PROOF -- cuts the [start_frame, end_frame] snippet, draws a thick RED box on
     the violating track each frame, overlays plate + confidence, and writes
     output/pl_review/<prefix>_violation_<track_id>.mp4.

cv2 is imported lazily inside the renderer so the alert/lookup logic stays importable on a
CPU/CI box. NOTE: cv2.putText cannot draw Hebrew, so the bilingual reason lives in the JSON
alert only; the on-video overlay uses an ASCII reason tag.

Consumes the unchanged cache schema (frame/vehicles[track_id/bbox]) and the run_pipeline payload.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_PIPELINE_DIR))

REASON_FULL = "Solid white line crossing / חציית קו הפרדה רציף"
REASON_ASCII = "SOLID WHITE LINE CROSSING"           # cv2 cannot render Hebrew -> ASCII on video
PLATE_UNKNOWN = "PLATE_UNKNOWN"

DEFAULT_OUT_DIR = os.path.join(_REPO, "output", "pl_review")


# --------------------------------------------------------------------------- #
# IO + lookups (pure, cv2-free)
# --------------------------------------------------------------------------- #
def load_json(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[alert_and_visualize] missing input: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def track_boxes(cache: dict, track_id: int) -> dict:
    """frame_index -> bbox for one track, scanned from the cache."""
    boxes = {}
    for fr in cache["frames"]:
        for v in fr.get("vehicles", []):
            if v["track_id"] == track_id:
                boxes[fr["frame"]] = v["bbox"]
                break
    return boxes


def alert_payload(violation: dict) -> dict:
    """Build the structured alert payload (plate falls back to PLATE_UNKNOWN)."""
    plate = violation.get("plate_candidate") or PLATE_UNKNOWN
    return {
        "event": "TRAFFIC_VIOLATION",
        "plate_number": plate,
        "violation_confidence": round(float(violation.get("confidence", 0.0)), 4),
        "violation_reason": REASON_FULL,
        "track_id": violation.get("track_id"),
        "start_frame": violation.get("start_frame"),
        "end_frame": violation.get("end_frame"),
        "plate_confidence_score": round(float(violation.get("plate_confidence_score", 0.0)), 4),
        "plate_source": violation.get("plate_source", "none"),
    }


def trigger_alert(violation: dict) -> dict:
    payload = alert_payload(violation)
    print("\n[ALERT] POST /api/violations ->")
    print(json.dumps(payload, ensure_ascii=False, indent=2))   # ensure_ascii=False -> Hebrew shows
    return payload


# --------------------------------------------------------------------------- #
# visual proof (cv2 here only)
# --------------------------------------------------------------------------- #
def render_snippet(video_path: str, cache: dict, violation: dict, out_dir: str,
                   prefix: str, pad: int = 0) -> str | None:
    """Write a red-boxed proof clip for one violation; returns the output path or None."""
    import cv2  # lazy: keeps this module importable without OpenCV

    tid = violation["track_id"]
    s, e = int(violation["start_frame"]), int(violation["end_frame"])
    plate = violation.get("plate_candidate") or PLATE_UNKNOWN
    conf = float(violation.get("confidence", 0.0))
    boxes = track_boxes(cache, tid)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[warn] cannot open video {video_path!r}; skipping track {tid}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or float(cache.get("fps", 25.0))
    win_s, win_e = max(0, s - pad), e + pad

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_violation_{tid}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    font = cv2.FONT_HERSHEY_SIMPLEX

    cap.set(cv2.CAP_PROP_POS_FRAMES, win_s)
    writer = None
    idx, written = win_s, 0
    while idx <= win_e:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        thick = max(3, round(h / 240.0))
        fscale = max(0.8, h / 700.0)
        red = (0, 0, 255)
        white = (255, 255, 255)

        bbox = boxes.get(idx)
        if bbox is not None:
            x1, y1, x2, y2 = (int(round(c)) for c in bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), red, thick)
            label = f"{plate}  conf={conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, font, fscale, thick)
            ly = max(y1 - 10, th + 14)
            cv2.rectangle(frame, (x1, ly - th - 12), (x1 + tw + 14, ly + 6), red, -1)
            cv2.putText(frame, label, (x1 + 7, ly), font, fscale, white,
                        max(2, thick - 1), cv2.LINE_AA)

        # corner banner (always) + red border while inside the event window
        banner = f"VIOLATION  {REASON_ASCII}  |  {plate}  conf={conf:.2f}"
        (bw, bh), _ = cv2.getTextSize(banner, font, fscale * 0.8, max(2, thick - 1))
        cv2.rectangle(frame, (0, 0), (bw + 24, bh + 20), (0, 0, 0), -1)
        cv2.putText(frame, banner, (12, bh + 10), font, fscale * 0.8, white,
                    max(2, thick - 1), cv2.LINE_AA)
        if s <= idx <= e:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), red, thick + 2)

        writer.write(frame)
        written += 1
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if written == 0:
        print(f"[warn] no frames written for track {tid} (window {win_s}-{win_e}); "
              f"check start/end vs video length")
        if os.path.isfile(out_path):
            os.remove(out_path)
        return None
    print(f"   proof -> {out_path}  ({written} frames @ {fps:.1f}fps)")
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Alert + visual proof for finished violations.")
    ap.add_argument("prefix", help="clip prefix, e.g. DeNnDugXxP0")
    ap.add_argument("--video", default=None, help="video path (default: the cache's recorded path)")
    ap.add_argument("--cache", default=None,
                    help="cache JSON (default: outputs/violation_cache/<prefix>.json)")
    ap.add_argument("--violations", default=None,
                    help="violations JSON (default: outputs/pipeline/<prefix>_violations.json)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--pad", type=int, default=0,
                    help="extra frames before/after the event window (0 = exact, per spec)")
    args = ap.parse_args()

    cache_path = args.cache or os.path.join(_REPO, "outputs", "violation_cache", f"{args.prefix}.json")
    viol_path = args.violations or os.path.join(_REPO, "outputs", "pipeline",
                                                f"{args.prefix}_violations.json")
    cache = load_json(cache_path)
    payload = load_json(viol_path)
    violations = payload.get("violations", [])
    video_path = args.video or cache.get("path")

    print(f"[consume] {args.prefix}: {len(violations)} violation(s) from {viol_path}")
    if not violations:
        print("[done] no violations to alert/visualize.")
        return

    written = []
    for v in violations:
        trigger_alert(v)
        out = render_snippet(video_path, cache, v, args.out_dir, args.prefix, pad=args.pad)
        if out:
            written.append(out)

    print(f"\n[done] {len(violations)} alert(s) sent, {len(written)} proof clip(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
