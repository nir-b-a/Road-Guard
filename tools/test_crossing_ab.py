"""A/B runner: legacy CrossingMonitor vs crossing2, same clip, same models, one video.

Pass 1 runs the models once and collects BOTH detectors' inputs:
  * the legacy path exactly as tools/test_corun_video.py wires it - stabilized solid
    mask + `CrossingMonitor.update_from_mask_image` on the bbox bottom-centre anchor;
  * a `FrameObservation` per frame (lane polylines + tracked boxes) for crossing2.

crossing2 then runs offline over the whole clip (it has to: centered smoothing,
two-pass lane association, bidirectional confirmation). Pass 2 re-reads the video and
renders one annotated output carrying both verdicts, so a disagreement can be watched
rather than inferred: a red border is crossing2, an "OLD" badge under the box is the
legacy detector.

Nothing in line_crossing/crossing_detector.py, dl_lane_detector.py, mask_stabilizer.py
or lane_types.py is modified or imported differently than the existing tool does.

Examples
--------
  python tools/test_crossing_ab.py --frames 600
  python tools/test_crossing_ab.py --video clip.mp4 --detector new --out out.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from Constants import DetectClass  # noqa: E402
from line_crossing.crossing2 import (  # noqa: E402
    CrossingConfig, FrameObservation, VehicleBox, detect_crossings, format_events,
)
from line_crossing.crossing2.overlay import CrossingOverlay  # noqa: E402
from line_crossing.crossing_detector import CrossingMonitor  # noqa: E402
from line_crossing.dl_lane_detector import DLLaneDetector  # noqa: E402
from line_crossing.mask_stabilizer import SolidMaskStabilizer  # noqa: E402

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "UnLanedet", "config", "clrernet", "resnet34_culane.py")
DEFAULT_CKPT = os.path.join(REPO_ROOT, "weights", "clrernet_model_best_culane.pth")
DEFAULT_VIDEO = os.path.join(
    REPO_ROOT, "tests_videos", "raw_videos", "crossing_solid_line",
    "0SmdindPVEY_Crossing_solid_white_line_-_Dashcam.f299.mp4",
)
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "phase4_ab", "crossing_ab.mp4")
DEFAULT_TYPE_WEIGHTS = os.path.join(REPO_ROOT, "models", "phase3_israeli_head.pth")
YOLO_WEIGHTS = os.path.join(REPO_ROOT, "yolov8m.pt")
YOLO_CONF = 0.5


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--type-weights", default=DEFAULT_TYPE_WEIGHTS)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--detector", choices=("new", "old", "both"), default="both",
                    help="which verdict(s) to draw; both always run so the table is complete")
    ap.add_argument("--ego-min-y", type=int, default=0,
                    help="drop boxes whose bottom y exceeds this (ego hood filter); 0 = off")
    ap.add_argument("--max-age", type=int, default=4, help="legacy stabilizer K")
    ap.add_argument("--t-high", type=float, default=0.40)
    ap.add_argument("--t-low", type=float, default=0.25)
    # crossing2 knobs worth sweeping by hand; everything else lives in CrossingConfig.
    ap.add_argument("--traverse-margin", type=float, default=None,
                    help="hysteresis band in LANE WIDTHS (default 0.15 ~ 0.5 m)")
    ap.add_argument("--traverse-hold", type=float, default=None, help="seconds held per side")
    ap.add_argument("--straddle-frac", type=float, default=None,
                    help="footprint fraction across the line for the strafe channel")
    ap.add_argument("--near-gate", type=float, default=None,
                    help="min bbox width as a fraction of the local lane width")
    ap.add_argument("--no-render", action="store_true", help="table only, skip pass 2")
    return ap.parse_args()


def build_config(args) -> CrossingConfig:
    cfg = CrossingConfig()
    if args.traverse_margin is not None:
        cfg.traverse_margin = args.traverse_margin
    if args.traverse_hold is not None:
        cfg.traverse_hold_sec = args.traverse_hold
    if args.straddle_frac is not None:
        cfg.straddle_min_frac = args.straddle_frac
    if args.near_gate is not None:
        cfg.min_bbox_lane_frac = args.near_gate
    return cfg


def collect(args, cap, lane_det, yolo, stabilizer, monitor, n, run_legacy: bool):
    """Pass 1: run the models once; return (observations, legacy per-frame flags)."""
    obs: list[FrameObservation] = []
    legacy: dict[int, set[int]] = {}
    legacy_new = 0
    t_yolo, t_lane = [], []

    for k in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frame_id = args.start + k

        t0 = time.perf_counter()
        res = yolo.track(frame, persist=True, verbose=False,
                         classes=DetectClass.Vehicle_Classes, conf=YOLO_CONF,
                         device=0 if args.device == "cuda" else "cpu")
        boxes = res[0].boxes if res else None
        names = res[0].names if res else {}
        vehicles: list[VehicleBox] = []
        if boxes is not None:
            for b in boxes:
                if b.id is None:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                if args.ego_min_y and y2 > args.ego_min_y:
                    continue
                vehicles.append(VehicleBox(int(b.id.item()), (x1, y1, x2, y2),
                                           names.get(int(b.cls.item()))))
        t1 = time.perf_counter()

        lanes = lane_det.detect(frame)
        raw_solid = lane_det.solid_lane_masks()
        t2 = time.perf_counter()

        if run_legacy:
            mask = stabilizer.update(raw_solid, frame_shape=frame.shape[:2])
            for v in vehicles:
                if monitor.update_from_mask_image(v.track_id, v.bbox, mask):
                    legacy_new += 1
                    print(f"[legacy] vehicle {v.track_id} latched at frame {frame_id}")
            legacy[frame_id] = {v.track_id for v in vehicles
                                if monitor.is_violator(v.track_id)}

        obs.append(FrameObservation(frame_id, lanes, vehicles))
        if k:
            t_yolo.append(t1 - t0)
            t_lane.append(t2 - t1)
        if k % 50 == 0:
            print(f"  frame {frame_id:>5}: veh={len(vehicles)} lanes={len(lanes)} "
                  f"raw_masks={len(raw_solid)}")

    if t_yolo:
        print(f"[time] yolo {1000*sum(t_yolo)/len(t_yolo):.1f} ms/f   "
              f"lane {1000*sum(t_lane)/len(t_lane):.1f} ms/f")
    return obs, legacy, legacy_new


def render(args, obs, result, legacy, src_fps, src_w, src_h):
    """Pass 2: re-read the clip and draw both verdicts onto one video."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[render] could not reopen {args.video}; skipping render")
        return
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h))
    if not writer.isOpened():
        print(f"[render] could not open writer {args.out}; skipping render")
        cap.release()
        return

    ov = CrossingOverlay(result)
    show_new = args.detector in ("new", "both")
    show_old = args.detector in ("old", "both")
    written = 0
    for ob in obs:
        ok, frame = cap.read()
        if not ok:
            break
        canvas = frame
        if show_new:
            ov.draw_lanes(canvas, ob.frame_id)
        states = []
        for v in ob.vehicles:
            if show_new:
                states.append(ov.draw_vehicle(canvas, ob.frame_id, v.track_id, v.bbox))
            else:
                x1, y1, x2, y2 = v.bbox
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
            if show_old:
                ov.draw_legacy_badge(canvas, v.bbox,
                                     v.track_id in legacy.get(ob.frame_id, ()))
        ov.draw_banner(canvas, ob.frame_id, states,
                       extra=f"old_latched={len(legacy.get(ob.frame_id, ()))}")
        writer.write(canvas)
        written += 1

    cap.release()
    writer.release()
    print(f"[render] wrote {written} frames -> {args.out}")


def main() -> None:
    args = parse_args()
    for label, path in [("config", args.config), ("ckpt", args.ckpt),
                        ("type-weights", args.type_weights),
                        ("video", args.video), ("yolo", YOLO_WEIGHTS)]:
        if not os.path.exists(path):
            raise SystemExit(f"Missing {label}: {path}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    n = min(args.frames, max(0, total - args.start)) if total > 0 else args.frames
    print(f"[video] {os.path.basename(args.video)} {src_w}x{src_h} @ {src_fps:.1f}fps  "
          f"{n} frames from {args.start}/{total}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    print("[load] YOLOv8m + CLRerNet + lane-type seg head ...")
    yolo = YOLO(YOLO_WEIGHTS)
    lane_det = DLLaneDetector(config_path=args.config, ckpt_path=args.ckpt,
                              device=args.device, src_size=(src_w, src_h),
                              type_weights=args.type_weights, type_every_n=1)
    monitor = CrossingMonitor()
    stabilizer = SolidMaskStabilizer(max_age=args.max_age, t_high=args.t_high,
                                     t_low=args.t_low, frame_size=(src_w, src_h))

    run_legacy = args.detector in ("old", "both")
    obs, legacy, legacy_new = collect(args, cap, lane_det, yolo, stabilizer,
                                      monitor, n, run_legacy)
    cap.release()

    cfg = build_config(args)
    t0 = time.perf_counter()
    result = detect_crossings(obs, src_w, src_h, src_fps, cfg)
    solve_ms = 1000 * (time.perf_counter() - t0)

    crossings = [e for e in result.events if e.kind == "crossing"]
    strafes = [e for e in result.events if e.kind == "strafe"]
    old_ids = set(monitor._violators)
    new_ids = result.violator_ids()

    print("\n========== LANE TRACKS (stage 0) ==========")
    for t in result.tracks:
        print(f"  L{t.track_id:<3} {t.lane_type:<13} obs={t.n_obs:<5} "
              f"{'SUPPRESSED ' if t.suppressed else ''}"
              f"{'DOUBLE ' if t.is_double else ''}{'GORE' if t.is_gore else ''}"
              f"  votes={ {k: round(v, 1) for k, v in t.votes.items()} }")

    print("\n========== crossing2 EVENTS ==========")
    print(f"  {len(crossings)} crossing(s), {len(strafes)} strafe(s)  "
          f"[{len(result.offsets)} vehicle-line pairs scored, solve {solve_ms:.0f} ms]")
    print(format_events(result.events))

    print("\n========== A/B ==========")
    print(f"  legacy latched  : {sorted(old_ids)}  ({legacy_new} first-hits)")
    print(f"  crossing2 crossing: {sorted(new_ids)}")
    print(f"  both            : {sorted(old_ids & new_ids)}")
    print(f"  legacy only     : {sorted(old_ids - new_ids)}")
    print(f"  crossing2 only  : {sorted(new_ids - old_ids)}")
    print(f"  crossing2 strafe: {sorted({e.vehicle_id for e in strafes})}")

    if not args.no_render:
        render(args, obs, result, legacy, src_fps, src_w, src_h)


if __name__ == "__main__":
    main()
