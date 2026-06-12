"""
Phase 3 qualitative test: run the trained YOLOv8-seg lane-type model on a dashcam
video and write an annotated output (boxes + masks + class labels) for visual review.

Loads the 3-class model (solid_white_lane / yellow_solid_lane / dashed_lane) and
draws ultralytics' standard overlay per frame.

Usage (roadguard-dl env):
    python tools/infer_lane_video.py
    python tools/infer_lane_video.py --video tests_videos/high_way_drive/route_241_western_negev.mp4 --frames 600
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import cv2
from ultralytics import YOLO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "models", "phase3_israeli_head.pt")
# Unseen highway (training frames came from highway_5_trans_samaria), so this is
# an honest generalization check rather than a replay of training data.
DEFAULT_VIDEO = os.path.join(REPO_ROOT, "tests_videos", "high_way_drive", "route_241_western_negev.mp4")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "outputs", "phase3_infer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=600, help="max frames to process (0 = all)")
    ap.add_argument("--minutes", type=float, default=0.0,
                    help="cap processing to this many minutes of footage (overrides --frames if >0)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0", help="'0' for CUDA:0, or 'cpu'")
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit(f"[error] weights not found: {args.weights}")
    if not os.path.isfile(args.video):
        raise SystemExit(f"[error] video not found: {args.video}")

    os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
    out = args.out or os.path.join(
        DEFAULT_OUT_DIR, os.path.splitext(os.path.basename(args.video))[0] + "_pred.mp4"
    )

    model = YOLO(args.weights, task="segment")
    print(f"[model] {os.path.basename(args.weights)}  classes: {model.names}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[error] could not open video: {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.minutes > 0:
        n = min(int(args.minutes * 60 * fps), total)
    else:
        n = total if args.frames <= 0 else min(args.frames, total)
    print(f"[video] {os.path.basename(args.video)} {w}x{h}@{fps:.1f}fps -> processing {n}/{total} frames")

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"[error] could not open VideoWriter: {out}")

    counts: Counter = Counter()
    frames_with_det = 0
    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=args.conf, verbose=False, device=args.device)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            frames_with_det += 1
            for c in res.boxes.cls.tolist():
                counts[res.names[int(c)]] += 1
        writer.write(res.plot())  # ultralytics overlay: boxes + masks + class labels
        i += 1
        if i % 100 == 0:
            print(f"  {i}/{n} frames...")

    cap.release()
    writer.release()
    print(f"\n[done] wrote {i} annotated frames -> {out}")
    print(f"[done] frames with >=1 detection: {frames_with_det}/{i}")
    print(f"[done] detections by class: {dict(counts)}")


if __name__ == "__main__":
    main()
