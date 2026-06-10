"""
Side-by-side dual-model comparison video: LEFT = previous/baseline weights,
RIGHT = the new mistake-driven weights. Both panels overlay boxes + instance
masks + class labels (Ultralytics result.plot()), perfectly frame-synced, with a
labelled banner on each side. One output .mp4 per input clip.

Usage (GPU conda interpreter; run from the repo root):
  python tools/generate_comparison_video.py \
      --videos tests_videos/high_way_drive/highway_5_trans_samaria.mp4 \
               tests_videos/high_way_drive/mitzpe_ramon_to_petah_tikva.mp4 \
      --left-weights weights/phase3_israeli_head.pt \
      --right-weights weights/phase3_v3_hardmined.pt \
      --minutes 2
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from class_stabilizer import ClassStabilizer  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_LEFT = os.path.join(REPO, "weights", "phase3_israeli_head.pt")
DEF_RIGHT = os.path.join(REPO, "weights", "phase3_v3_hardmined.pt")
DEF_OUT = os.path.join(REPO, "outputs", "phase3_compare")
BANNER_H = 34


def banner(panel: np.ndarray, text: str, color) -> np.ndarray:
    """Draw a labelled banner strip across the top of a panel."""
    cv2.rectangle(panel, (0, 0), (panel.shape[1], BANNER_H), (0, 0, 0), -1)
    cv2.putText(panel, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return panel


def process_video(video, left_model, right_model, left_label, right_label,
                  conf, minutes, panel_w, out_dir) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[skip] cannot open {video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(int(minutes * 60 * fps), total) if minutes > 0 else total

    panel_h = int(panel_w * src_h / src_w)
    out_w, out_h = panel_w * 2, panel_h
    name = os.path.splitext(os.path.basename(video))[0]
    out_path = os.path.join(out_dir, f"{name}_compare.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    print(f"[video] {name} {src_w}x{src_h}@{fps:.0f}fps -> {n} frames -> {out_w}x{out_h}")
    lc, rc = Counter(), Counter()
    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        lres = left_model.predict(frame, conf=conf, verbose=False)[0]
        rres = right_model.predict(frame, conf=conf, verbose=False)[0]
        for c in (lres.boxes.cls.tolist() if lres.boxes is not None else []):
            lc[lres.names[int(c)]] += 1
        for c in (rres.boxes.cls.tolist() if rres.boxes is not None else []):
            rc[rres.names[int(c)]] += 1
        left = cv2.resize(lres.plot(), (panel_w, panel_h))
        right = cv2.resize(rres.plot(), (panel_w, panel_h))
        banner(left, left_label, (0, 200, 255))    # amber
        banner(right, right_label, (0, 255, 120))   # green
        writer.write(np.hstack([left, right]))
        i += 1
        if i % 200 == 0:
            print(f"  {i}/{n} frames...")

    cap.release()
    writer.release()
    print(f"[done] {name}: {i} frames -> {out_path}")
    print(f"        LEFT  ({left_label}) detections: {dict(lc)}")
    print(f"        RIGHT ({right_label}) detections: {dict(rc)}")


def process_video_stab(video, model, stab_params, left_label, right_label,
                       conf, minutes, panel_w, out_dir) -> None:
    """SINGLE-model mode: LEFT = raw per-frame labels, RIGHT = temporally
    stabilized labels (same weights, same detections). Masks are kept from the
    model; only each detection's class index is remapped to the stabilizer's
    decision, so the box+mask+label all recolor to the stable class."""
    import torch
    from ultralytics.engine.results import Boxes

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[skip] cannot open {video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(int(minutes * 60 * fps), total) if minutes > 0 else total

    panel_h = int(panel_w * src_h / src_w)
    name = os.path.splitext(os.path.basename(video))[0]
    out_path = os.path.join(out_dir, f"{name}_stabilized.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (panel_w * 2, panel_h))

    stab = ClassStabilizer(**stab_params)
    name2idx = None
    print(f"[video] {name} {src_w}x{src_h}@{fps:.0f}fps -> {n} frames (stabilize mode)")
    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=conf, verbose=False)[0]
        left = res.plot()                                   # raw per-frame labels

        dets = []
        if res.boxes is not None and len(res.boxes):
            if name2idx is None:
                name2idx = {v: k for k, v in res.names.items()}
            for c, cf, b in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist(),
                                res.boxes.xyxy.tolist()):
                dets.append((res.names[int(c)], float(cf), tuple(b)))
        stable = stab.update(dets)
        # Match stabilizer output back to detections by box, remap class index.
        bmap = {tuple(round(v, 1) for v in sd.box): sd.stable_class for sd in stable}
        if res.boxes is not None and len(res.boxes):
            new_idx = []
            for c, b in zip(res.boxes.cls.tolist(), res.boxes.xyxy.tolist()):
                sc = bmap.get(tuple(round(v, 1) for v in b))
                new_idx.append(name2idx[sc] if sc in name2idx else int(c))
            new_data = res.boxes.data.clone()               # inference tensor is read-only
            new_data[:, 5] = torch.tensor(new_idx, dtype=new_data.dtype, device=new_data.device)
            res.boxes = Boxes(new_data, res.boxes.orig_shape)
        right = res.plot()                                  # stabilized labels (masks recolor too)

        left = cv2.resize(left, (panel_w, panel_h))
        right = cv2.resize(right, (panel_w, panel_h))
        banner(left, left_label, (0, 200, 255))
        banner(right, right_label + " (stabilized)", (0, 255, 120))
        writer.write(np.hstack([left, right]))
        i += 1
        if i % 200 == 0:
            print(f"  {i}/{n} frames...")

    cap.release()
    writer.release()
    st = stab.flicker_stats()
    print(f"[done] {name}: {i} frames -> {out_path}")
    print(f"        flicker (class-changes/track): RAW {st['raw_per_track']:.3f}  ->  "
          f"STABILIZED {st['stable_per_track']:.3f}  "
          f"({(1 - st['stable_per_track'] / st['raw_per_track']) * 100 if st['raw_per_track'] else 0:.1f}% less)"
          f"  over {st['n_tracks']} tracks")


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side dual-model lane/island comparison video.")
    ap.add_argument("--videos", nargs="+", default=[
        os.path.join(REPO, "tests_videos", "high_way_drive", "highway_5_trans_samaria.mp4"),
        os.path.join(REPO, "tests_videos", "high_way_drive", "mitzpe_ramon_to_petah_tikva.mp4"),
    ])
    ap.add_argument("--left-weights", default=DEF_LEFT)
    ap.add_argument("--right-weights", default=DEF_RIGHT)
    ap.add_argument("--left-label", default="LEFT: Previous Model")
    ap.add_argument("--right-label", default="RIGHT: Phase 3 Hard-Mined Model")
    ap.add_argument("--minutes", type=float, default=2.0, help="minutes per clip (0 = full)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--panel-width", type=int, default=960, help="each side's width (output = 2x)")
    ap.add_argument("--out", default=DEF_OUT)
    # --- temporal class-stabilizer mode (single model: LEFT raw vs RIGHT stabilized) ---
    ap.add_argument("--stabilize", action="store_true",
                    help="single-model mode: LEFT=raw per-frame, RIGHT=temporally stabilized labels")
    ap.add_argument("--stab-weights", default=os.path.join(REPO, "weights", "phase3_v3_yellowprotect.pt"),
                    help="weights used on BOTH panels in --stabilize mode")
    ap.add_argument("--stab-window", type=int, default=15)
    ap.add_argument("--stab-margin", type=float, default=0.20)
    ap.add_argument("--stab-k", type=int, default=5)
    ap.add_argument("--stab-lam", type=float, default=0.85)
    ap.add_argument("--stab-iou", type=float, default=0.30)
    ap.add_argument("--stab-conf-gate", type=float, default=0.40)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.stabilize:
        if not os.path.isfile(args.stab_weights):
            raise SystemExit(f"[error] weights not found: {args.stab_weights}")
        stab_params = dict(window=args.stab_window, switch_margin=args.stab_margin,
                           switch_k=args.stab_k, lam=args.stab_lam, iou_assoc=args.stab_iou,
                           conf_gate=args.stab_conf_gate)
        print("=" * 64)
        print(f"STABILIZE mode  weights: {args.stab_weights}")
        print(f"stabilizer: {stab_params}")
        print(f"clips : {len(args.videos)}   minutes/clip: {args.minutes}   conf: {args.conf}")
        print("=" * 64)
        model = YOLO(args.stab_weights, task="segment")
        print(f"[model] classes: {model.names}")
        right_label = f"RIGHT: Stabilized (N={args.stab_window}, m={args.stab_margin})"
        for v in args.videos:
            if os.path.isfile(v):
                process_video_stab(v, model, stab_params, "LEFT: Raw per-frame",
                                   right_label, args.conf, args.minutes,
                                   args.panel_width, args.out)
            else:
                print(f"[skip] not found: {v}")
        return

    for w in (args.left_weights, args.right_weights):
        if not os.path.isfile(w):
            raise SystemExit(f"[error] weights not found: {w}")

    print("=" * 64)
    print(f"LEFT  : {args.left_weights}")
    print(f"RIGHT : {args.right_weights}")
    print(f"clips : {len(args.videos)}   minutes/clip: {args.minutes}   conf: {args.conf}")
    print("=" * 64)

    left_model = YOLO(args.left_weights, task="segment")
    right_model = YOLO(args.right_weights, task="segment")
    print(f"[left]  classes: {left_model.names}")
    print(f"[right] classes: {right_model.names}")

    for v in args.videos:
        if os.path.isfile(v):
            process_video(v, left_model, right_model, args.left_label, args.right_label,
                          args.conf, args.minutes, args.panel_width, args.out)
        else:
            print(f"[skip] not found: {v}")


if __name__ == "__main__":
    main()
