"""
End-to-end smoke test for the Phase 2 DL lane detector (CLRerNet via UnLanedet).

Grabs a real frame from a highway dashcam video, runs DLLaneDetector.detect()
(model load + UnLanedet preprocess + forward + decode + coord remap), prints a
summary, and writes an annotated image so the geometry can be eyeballed.

Run inside the roadguard-dl env with the CUDA 11.8 bin on PATH (the compiled
ops link against cudart). No MSVC needed at runtime.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Make `line_crossing` importable when run from the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from line_crossing.dl_lane_detector import DLLaneDetector  # noqa: E402

DEFAULT_CONFIG = os.path.join(
    REPO_ROOT, "UnLanedet", "config", "clrernet", "resnet34_culane.py"
)
DEFAULT_CKPT = os.path.join(REPO_ROOT, "weights", "clrernet_model_best_culane.pth")
DEFAULT_VIDEO = os.path.join(
    REPO_ROOT, "tests_videos", "high_way_drive", "highway_5_trans_samaria.mp4"
)

_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0),
]


def grab_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(frame_index, max(0, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit(f"Could not read frame {idx} from {video_path}")
    print(f"[frame] {os.path.basename(video_path)} idx={idx}/{total} "
          f"shape={frame.shape[1]}x{frame.shape[0]}")
    return frame


def annotate(frame: np.ndarray, lanes) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.line(out, (w // 2, 0), (w // 2, h), (180, 180, 180), 1)  # image center
    for i, lane in enumerate(lanes):
        color = _COLORS[i % len(_COLORS)]
        pts = np.array(lane.points, dtype=np.int32)
        for p in pts:
            cv2.circle(out, (int(p[0]), int(p[1])), 4, color, -1)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, 2)
        bx, by = lane.bottom
        cv2.putText(out, f"pos={lane.position} {lane.lane_type}",
                    (int(bx) + 6, int(by)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "weights", "dl_inference_test.jpg"))
    args = ap.parse_args()

    for label, path in [("config", args.config), ("ckpt", args.ckpt), ("video", args.video)]:
        if not os.path.exists(path):
            raise SystemExit(f"Missing {label}: {path}")

    frame = grab_frame(args.video, args.frame)
    h, w = frame.shape[:2]

    print("[load] building CLRerNet + loading checkpoint (first call is slow)...")
    det = DLLaneDetector(
        config_path=args.config,
        ckpt_path=args.ckpt,
        device=args.device,
        src_size=(w, h),
    )

    # Two calls: first includes lazy model load, second is steady-state latency.
    import time
    t0 = time.time()
    lanes = det.detect(frame)
    t1 = time.time()
    lanes = det.detect(frame)
    t2 = time.time()

    print(f"[time] first detect (incl. load) = {t1 - t0:.2f}s | "
          f"warm detect = {t2 - t1:.3f}s")
    print(f"[result] {len(lanes)} lane(s) detected")
    for ln in sorted(lanes, key=lambda l: (l.position is None, l.position or 0)):
        xs = [p[0] for p in ln.points]
        ys = [p[1] for p in ln.points]
        print(f"  pos={ln.position!s:>4}  type={ln.lane_type:<7} "
              f"pts={len(ln.points):>2}  "
              f"bottom={ln.bottom}  top={ln.top}  "
              f"x-range=[{min(xs):.0f},{max(xs):.0f}] y-range=[{min(ys):.0f},{max(ys):.0f}]")

    # Sanity assertions on the coordinate remap (should be within frame bounds).
    in_bounds = all(
        0 <= p[0] <= w and 0 <= p[1] <= h for ln in lanes for p in ln.points
    )
    print(f"[check] all points within frame bounds ({w}x{h}): {in_bounds}")

    out_img = annotate(frame, lanes)
    cv2.imwrite(args.out, out_img)
    print(f"[saved] annotated frame -> {args.out}")


if __name__ == "__main__":
    main()
