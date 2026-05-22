"""
Visual parameter sweep for the HSV+fitLine lane-detection pipeline.

Reads all frames from test_frames/, runs every combination of the tunable
constants, draws detected solid lines, and saves annotated images to
output_tuning/ with filenames that encode the parameters used.

Usage (from project root):
    python line_crossing/tune_lanes.py

Output (per frame × combo):
    output_tuning/<frame_stem>__thr<N>__len<N>__gap<N>__ang<min>-<max>__cov<N>.jpg

Note: this generates many images (324 combos × 15 frames = 4 860 files).
Use score_lanes.py for the automated winner selection instead.
"""

import sys
import itertools
import cv2
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "test_frames"
OUT_DIR    = ROOT / "output_tuning"

SWEEP = {
    "hough_thr":      [25, 35, 50],
    "min_line_len":   [40, 60],
    "max_gap":        [15, 50, 100],
    "angle_min":      [10, 15, 20],
    "angle_max":      [80, 85],
    "solid_coverage": [0.50, 0.65, 0.80],
}

ROI_CROP_RATIO = 0.5


def _detect(frame, p):
    h, w = frame.shape[:2]
    roi_top = int(h * (1 - ROI_CROP_RATIO))
    roi = frame[roi_top:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lane_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0,  0,  180), (180, 40,  255)),
        cv2.inRange(hsv, (18, 80,  80), ( 35, 255, 255)),
    )

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(cv2.bitwise_and(gray, lane_mask), (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    raw = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=p["hough_thr"],
        minLineLength=p["min_line_len"],
        maxLineGap=p["max_gap"],
    )
    if raw is None:
        return {}

    ang_min, ang_max = p["angle_min"], p["angle_max"]
    filtered = []
    for seg in raw:
        x1, y1, x2, y2 = seg[0]
        if x1 == x2:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if (ang_min <= angle <= ang_max) or (180 - ang_max <= angle <= 180 - ang_min):
            filtered.append((x1, y1, x2, y2))

    if not filtered:
        return {}

    cx = w // 2
    result = {}
    for key, segs in (
        ("solid_line_left",  [s for s in filtered if (s[0] + s[2]) // 2 < cx]),
        ("solid_line_right", [s for s in filtered if (s[0] + s[2]) // 2 >= cx]),
    ):
        if not segs:
            continue
        all_y  = [y for _, y1, _, y2 in segs for y in (y1, y2)]
        y_span = max(all_y) - min(all_y)
        if y_span == 0:
            continue
        total_cov = sum(abs(y2 - y1) for _, y1, _, y2 in segs)
        if total_cov / y_span < p["solid_coverage"]:
            continue

        all_x  = [x for x1, _, x2, _ in segs for x in (x1, x2)]
        points = np.array(list(zip(all_x, all_y)), dtype=np.float32).reshape(-1, 1, 2)
        vx, vy, cx2, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten().tolist()
        if abs(vy) < 1e-6:
            continue
        y_min, y_max = min(all_y), max(all_y)
        t_top = (y_min - cy) / vy
        t_bot = (y_max - cy) / vy
        result[key] = (int(cx2 + t_top * vx), y_min + roi_top,
                       int(cx2 + t_bot * vx), y_max + roi_top)

    return result


def _tag(p):
    return (
        f"thr{p['hough_thr']}"
        f"__len{p['min_line_len']}"
        f"__gap{p['max_gap']}"
        f"__ang{p['angle_min']}-{p['angle_max']}"
        f"__cov{p['solid_coverage']}"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frames:
        print(f"No frames in {FRAMES_DIR}. Run extract_frames.py first.")
        sys.exit(1)

    keys   = list(SWEEP.keys())
    combos = list(itertools.product(*SWEEP.values()))
    total  = len(frames) * len(combos)
    print(f"Frames={len(frames)}  Combos={len(combos)}  Total={total}")

    done = 0
    for fp in frames:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        for combo in combos:
            p = dict(zip(keys, combo))
            out_name = f"{fp.stem}__{_tag(p)}.jpg"
            out_path = OUT_DIR / out_name
            if out_path.exists():
                done += 1
                continue

            detected = _detect(frame, p)
            out = frame.copy()
            cv2.line(out, (w // 2, 0), (w // 2, h), (0, 200, 0), 1)
            for label, (x1, y1, x2, y2) in detected.items():
                cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(out, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            note = f"{_tag(p)} | found: {list(detected.keys()) or 'none'}"
            cv2.putText(out, note, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, note, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_path), out)

            done += 1
            if done % 200 == 0:
                print(f"  {done}/{total} ...")

    print(f"\nDone. {len(list(OUT_DIR.glob('*.jpg')))} images in {OUT_DIR}/")


if __name__ == "__main__":
    main()
