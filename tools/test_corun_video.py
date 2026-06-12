"""
Phase 3 co-run + LIVE VIOLATION test: YOLOv8 (vehicle tracking) + DLLaneDetector
(CLRerNet geometry + YOLOv8-seg lane-type masks) on a dashcam clip, written to an
annotated video.

Mask-based solid-line crossing
------------------------------
  * the seg head emits solid_white / solid_yellow lane MASKS per frame (via
    DLLaneDetector._classify_types, exposed by `.solid_lane_masks()`);
  * for every tracked vehicle we take its bottom-center anchor (x_center, y_max)
    - where the tyres meet the road - and test it against those solid masks
    (CrossingMonitor.update_from_masks -> point-in-polygon);
  * a hit flags that vehicle ID as a solid-line violator (state persists), turns
    its box red, and raises an on-frame alert banner.

Also reports throughput (avg/warm FPS, per-stage timing) and GPU VRAM headroom so
we can confirm the 6 GB GTX 1060 survives running all three models per frame.

Run inside the roadguard-dl env with the CUDA 11.8 bin on PATH.
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
from line_crossing.crossing_detector import CrossingMonitor, bbox_bottom_center  # noqa: E402
from line_crossing.dl_lane_detector import DLLaneDetector  # noqa: E402
from line_crossing.mask_stabilizer import SolidMaskStabilizer  # noqa: E402

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "UnLanedet", "config", "clrernet", "resnet34_culane.py")
DEFAULT_CKPT = os.path.join(REPO_ROOT, "weights", "clrernet_model_best_culane.pth")
# A clip where a car actually crosses a solid white line, so the violation logic
# has something to fire on (swap with --video for highway/no-violation footage).
DEFAULT_VIDEO = os.path.join(
    REPO_ROOT, "tests_videos", "raw_videos", "crossing_solid_line",
    "0SmdindPVEY_Crossing_solid_white_line_-_Dashcam.f299.mp4",
)
DEFAULT_OUT = os.path.join(REPO_ROOT, "outputs", "phase3_infer", "corun_violation_output.mp4")
DEFAULT_TYPE_WEIGHTS = os.path.join(REPO_ROOT, "models", "phase3_israeli_head.pth")

YOLO_WEIGHTS = os.path.join(REPO_ROOT, "yolov8m.pt")
YOLO_CONF = 0.5

LANE_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
BOX_COLOR = (0, 255, 0)            # normal tracked vehicle (green)
VIOLATION_COLOR = (0, 0, 255)      # flagged vehicle (red)
ANCHOR_COLOR = (0, 165, 255)       # bottom-center anchor dot (orange)
SOLID_MASK_COLOR = (0, 0, 255)     # stabilized solid-lane overlay (red)


def gb(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def draw_solid_mask(canvas: np.ndarray, solid_mask) -> None:
    """Translucent overlay of the STABILIZED binary solid-lane mask (the violation
    surface the anchor test actually sees)."""
    if solid_mask is None or not solid_mask.any():
        return
    overlay = canvas.copy()
    overlay[solid_mask > 0] = SOLID_MASK_COLOR
    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)


def draw_lanes(canvas: np.ndarray, lanes) -> None:
    for i, lane in enumerate(lanes):
        color = LANE_COLORS[i % len(LANE_COLORS)]
        pts = np.array(lane.points, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, color, 3)
        bx, by = lane.bottom
        cv2.putText(canvas, f"{lane.position} {lane.lane_type}", (int(bx) + 6, int(by) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_vehicles(canvas: np.ndarray, vehicles, monitor: CrossingMonitor) -> bool:
    """Draw each vehicle box (red if flagged) + its bottom-center anchor.
    Returns True if any visible vehicle is currently a violator."""
    any_violator = False
    for vid, (x1, y1, x2, y2) in vehicles:
        violator = monitor.is_violator(vid)
        any_violator = any_violator or violator
        color = VIOLATION_COLOR if violator else BOX_COLOR
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 6 if violator else 2)
        label = f"VIOLATION ID {vid}" if violator else f"ID {vid}"
        cv2.putText(canvas, label, (x1, max(y1 - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3 if violator else 2, cv2.LINE_AA)
        # The exact anchor point fed into the point-in-mask test.
        ax, ay = bbox_bottom_center((x1, y1, x2, y2))
        cv2.circle(canvas, (ax, ay), 6, ANCHOR_COLOR, -1)
    return any_violator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--type-weights", default=DEFAULT_TYPE_WEIGHTS,
                    help="Phase 3 YOLOv8-seg lane-type weights (.pth)")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start", type=int, default=0, help="first frame to process (seek)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ego-min-y", type=int, default=0,
                    help="drop vehicle boxes whose bottom y exceeds this (ego hood filter); "
                         "0 = disabled")
    ap.add_argument("--max-age", type=int, default=4,
                    help="SolidMaskStabilizer K: frames a vanished mask coasts (slow-decay)")
    ap.add_argument("--t-high", type=float, default=0.40, help="hysteresis ADMIT threshold")
    ap.add_argument("--t-low", type=float, default=0.25, help="hysteresis KEEP threshold")
    args = ap.parse_args()

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
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    n = min(args.frames, total - args.start)
    print(f"[video] {os.path.basename(args.video)}  {src_w}x{src_h} @ {src_fps:.1f}fps  "
          f"processing {n} frames from idx {args.start}/{total}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = torch.device("cuda", torch.cuda.current_device()) if args.device == "cuda" \
        else torch.device(args.device)
    free0, totalvram = torch.cuda.mem_get_info(dev) if args.device == "cuda" else (0, 0)
    print(f"[vram] device total={gb(totalvram):.2f} GB  free-at-start={gb(free0):.2f} GB")

    print("[load] YOLOv8m + CLRerNet + YOLOv8-seg lane-type head ...")
    yolo = YOLO(YOLO_WEIGHTS)
    lane_det = DLLaneDetector(
        config_path=args.config, ckpt_path=args.ckpt, device=args.device,
        src_size=(src_w, src_h),
        type_weights=args.type_weights,
        type_every_n=1,   # fresh masks every frame for a crisp live violation test
    )
    monitor = CrossingMonitor()
    stabilizer = SolidMaskStabilizer(
        max_age=args.max_age, t_high=args.t_high, t_low=args.t_low,
        frame_size=(src_w, src_h),
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter: {args.out}")

    yolo_t, lane_t, frame_t = [], [], []
    min_free = free0
    written = 0
    new_violations = 0

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        canvas = frame.copy()

        f0 = time.perf_counter()
        # --- YOLOv8 vehicle tracking ---
        t0 = time.perf_counter()
        results = yolo.track(frame, persist=True, verbose=False,
                             classes=DetectClass.Vehicle_Classes, conf=YOLO_CONF,
                             device=0 if args.device == "cuda" else "cpu")
        vehicles = []
        boxes = results[0].boxes if results else None
        if boxes is not None:
            for box in boxes:
                if box.id is None:
                    continue
                vid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if args.ego_min_y and y2 > args.ego_min_y:
                    continue  # ego dashboard/hood region
                vehicles.append((vid, (x1, y1, x2, y2)))
        t1 = time.perf_counter()

        # --- DL lane detection (geometry) + seg lane-type masks -> stabilized raster ---
        lanes = lane_det.detect(frame)
        raw_solid = lane_det.solid_lane_masks()                          # [(type, poly, conf), ...]
        solid_mask = stabilizer.update(raw_solid, frame_shape=frame.shape[:2])  # uint8 HxW 0/255
        t2 = time.perf_counter()

        # --- mask-based solid-line crossing test (O(1) raster lookup) ---
        for vid, bbox in vehicles:
            if monitor.update_from_mask_image(vid, bbox, solid_mask):
                new_violations += 1
                print(f"[VIOLATION] vehicle {vid} on solid line at frame {args.start + i}")

        # --- render ---
        draw_solid_mask(canvas, solid_mask)
        draw_lanes(canvas, lanes)
        any_violator = draw_vehicles(canvas, vehicles, monitor)
        cv2.putText(canvas, f"frame {i}  veh={len(vehicles)}  raw_masks={len(raw_solid)}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        if any_violator:
            cv2.putText(canvas, "!!! SOLID LINE VIOLATION !!!", (60, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, VIOLATION_COLOR, 6, cv2.LINE_AA)
        writer.write(canvas)
        written += 1

        if i > 0:
            yolo_t.append(t1 - t0)
            lane_t.append(t2 - t1)
            frame_t.append(time.perf_counter() - f0)
        if args.device == "cuda":
            free_now, _ = torch.cuda.mem_get_info(dev)
            min_free = min(min_free, free_now)

        if i % 50 == 0:
            print(f"  frame {i:>4}: veh={len(vehicles)} lanes={len(lanes)} "
                  f"raw_masks={len(raw_solid)} violators={len(monitor._violators)}")

    cap.release()
    writer.release()

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n========== RESULTS ==========")
    print(f"wrote {written} frames -> {args.out}")
    print(f"unique vehicles flagged for solid-line crossing: {new_violations}")
    if frame_t:
        print(f"avg per-frame: {avg(frame_t)*1000:.1f} ms  ->  {1.0/avg(frame_t):.1f} FPS  (combined, warm)")
        print(f"  YOLOv8  stage: {avg(yolo_t)*1000:6.1f} ms ({1.0/avg(yolo_t):5.1f} FPS solo)")
        print(f"  DL lane stage: {avg(lane_t)*1000:6.1f} ms ({1.0/avg(lane_t):5.1f} FPS solo)  "
              f"(CLRerNet + seg type head)")
    if args.device == "cuda":
        torch.cuda.synchronize(dev)
        peak_reserved = torch.cuda.max_memory_reserved(dev)
        print(f"[vram] torch peak reserved={gb(peak_reserved):.2f} GB")
        print(f"[vram] device min-free during loop={gb(min_free):.2f} GB / {gb(totalvram):.2f} GB "
              f"(peak usage ~{gb(totalvram - min_free):.2f} GB)")
        headroom = gb(min_free)
        print(f"[verdict] {'OK - headroom ' + f'{headroom:.2f} GB' if headroom > 0.2 else 'TIGHT - low headroom'}")


if __name__ == "__main__":
    main()
