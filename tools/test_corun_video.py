"""
Phase 2 co-run visual check: YOLOv8 (vehicles) + DLLaneDetector (CLRerNet lanes)
on a real dashcam clip, written to an annotated video.

Also reports throughput (avg/warm FPS, per-stage timing) and GPU VRAM headroom
so we can confirm the 6 GB GTX 1060 survives running both models per frame.

Run inside the roadguard-dl env with the CUDA 11.8 bin on PATH (the compiled
UnLanedet ops link against cudart).
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
from line_crossing.dl_lane_detector import DLLaneDetector  # noqa: E402

DEFAULT_CONFIG = os.path.join(REPO_ROOT, "UnLanedet", "config", "clrernet", "resnet34_culane.py")
DEFAULT_CKPT = os.path.join(REPO_ROOT, "weights", "clrernet_model_best_culane.pth")
DEFAULT_VIDEO = os.path.join(REPO_ROOT, "tests_videos", "high_way_drive", "highway_5_trans_samaria.mp4")
DEFAULT_OUT = os.path.join(REPO_ROOT, "weights", "corun_test_output.mp4")

YOLO_WEIGHTS = os.path.join(REPO_ROOT, "yolov8m.pt")
YOLO_CONF = 0.5

LANE_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
BOX_COLOR = (255, 200, 0)


def gb(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def draw_lanes(canvas: np.ndarray, lanes) -> None:
    for i, lane in enumerate(lanes):
        color = LANE_COLORS[i % len(LANE_COLORS)]
        pts = np.array(lane.points, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, color, 3)
        for p in pts:
            cv2.circle(canvas, (int(p[0]), int(p[1])), 4, color, -1)
        bx, by = lane.bottom
        cv2.putText(canvas, f"lane {lane.position}", (int(bx) + 6, int(by) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_boxes(canvas: np.ndarray, vehicles) -> None:
    for vid, (x1, y1, x2, y2) in vehicles:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"ID {vid}" if vid is not None else "veh"
        cv2.putText(canvas, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BOX_COLOR, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--start", type=int, default=0, help="first frame to process (seek)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for label, path in [("config", args.config), ("ckpt", args.ckpt),
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
    # mem_get_info / max_memory_* need a device with an explicit index.
    dev = torch.device("cuda", torch.cuda.current_device()) if args.device == "cuda" \
        else torch.device(args.device)
    free0, totalvram = torch.cuda.mem_get_info(dev) if args.device == "cuda" else (0, 0)
    print(f"[vram] device total={gb(totalvram):.2f} GB  free-at-start={gb(free0):.2f} GB")

    print("[load] YOLOv8m + CLRerNet ...")
    yolo = YOLO(YOLO_WEIGHTS)
    lane_det = DLLaneDetector(config_path=args.config, ckpt_path=args.ckpt,
                              device=args.device, src_size=(src_w, src_h))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open VideoWriter: {args.out}")

    yolo_t, lane_t, frame_t = [], [], []
    min_free = free0
    written = 0

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        canvas = frame.copy()

        f0 = time.perf_counter()
        # --- YOLOv8 vehicle tracking (same settings as main.py) ---
        t0 = time.perf_counter()
        results = yolo.track(frame, persist=True, verbose=False,
                             classes=DetectClass.Vehicle_Classes, conf=YOLO_CONF,
                             device=0 if args.device == "cuda" else "cpu")
        vehicles = []
        boxes = results[0].boxes if results else None
        if boxes is not None:
            for box in boxes:
                vid = int(box.id.item()) if box.id is not None else None
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                vehicles.append((vid, (x1, y1, x2, y2)))
        t1 = time.perf_counter()

        # --- DL lane detection ---
        lanes = lane_det.detect(frame)
        t2 = time.perf_counter()

        draw_lanes(canvas, lanes)
        draw_boxes(canvas, vehicles)
        cv2.putText(canvas, f"frame {i}  veh={len(vehicles)}  lanes={len(lanes)}",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
        written += 1

        # Skip frame 0 in timing (model warmup / first-call lazy load).
        if i > 0:
            yolo_t.append(t1 - t0)
            lane_t.append(t2 - t1)
            frame_t.append(time.perf_counter() - f0)
        if args.device == "cuda":
            free_now, _ = torch.cuda.mem_get_info(dev)
            min_free = min(min_free, free_now)

        if i % 50 == 0:
            print(f"  frame {i:>4}: veh={len(vehicles)} lanes={len(lanes)}")

    cap.release()
    writer.release()

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n========== RESULTS ==========")
    print(f"wrote {written} frames -> {args.out}")
    if frame_t:
        print(f"avg per-frame: {avg(frame_t)*1000:.1f} ms  ->  {1.0/avg(frame_t):.1f} FPS  (combined, warm)")
        print(f"  YOLOv8  stage: {avg(yolo_t)*1000:6.1f} ms ({1.0/avg(yolo_t):5.1f} FPS solo)")
        print(f"  DL lane stage: {avg(lane_t)*1000:6.1f} ms ({1.0/avg(lane_t):5.1f} FPS solo)")
    if args.device == "cuda":
        torch.cuda.synchronize(dev)
        peak_reserved = torch.cuda.max_memory_reserved(dev)
        peak_alloc = torch.cuda.max_memory_allocated(dev)
        print(f"[vram] torch peak reserved={gb(peak_reserved):.2f} GB  "
              f"peak allocated={gb(peak_alloc):.2f} GB")
        print(f"[vram] device min-free during loop={gb(min_free):.2f} GB / {gb(totalvram):.2f} GB "
              f"(peak usage ~{gb(totalvram - min_free):.2f} GB)")
        headroom = gb(min_free)
        print(f"[verdict] {'OK - headroom remained ' + f'{headroom:.2f} GB' if headroom > 0.2 else 'TIGHT - low headroom'}")


if __name__ == "__main__":
    main()
