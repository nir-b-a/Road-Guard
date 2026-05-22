"""
Automated geometric scorer for the HSV+fitLine lane-detection pipeline.

Sweeps over the tunable constants in line_detector.py (the parameters that
control HoughLinesP and solid/dashed classification), scores every combination
on the 15 representative test frames, applies the winner back to line_detector.py,
writes WINNING_PARAMETERS.md, and generates verification_best_params/.

Usage (from project root):
    python line_crossing/score_lanes.py
"""

import sys
import itertools
import re
import cv2
import numpy as np
from pathlib import Path

ROOT       = Path(__file__).parent.parent
FRAMES_DIR = ROOT / "test_frames"
VERIFY_DIR = ROOT / "verification_best_params"
DETECTOR   = ROOT / "line_crossing" / "line_detector.py"
SUMMARY_MD = ROOT / "WINNING_PARAMETERS.md"

# ---------------------------------------------------------------------------
# Parameter grid — sweeps only the constants that live in line_detector.py
# ---------------------------------------------------------------------------
SWEEP = {
    "hough_thr":      [25, 35, 50],
    "min_line_len":   [40, 60],
    "max_gap":        [15, 50, 100],
    "angle_min":      [10, 15, 20],
    "angle_max":      [80, 85],
    "solid_coverage": [0.50, 0.65, 0.80],
}

ROI_CROP_RATIO = 0.5


# ---------------------------------------------------------------------------
# Inline reimplementation of detect_solid_lines() that accepts params as args
# (avoids monkey-patching module-level constants between iterations)
# ---------------------------------------------------------------------------

def _detect(frame: np.ndarray, p: dict) -> dict:
    h, w = frame.shape[:2]
    roi_top = int(h * (1 - ROI_CROP_RATIO))
    roi = frame[roi_top:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask  = cv2.inRange(hsv, (0,  0,  180), (180, 40,  255))
    yellow_mask = cv2.inRange(hsv, (18, 80,  80), ( 35, 255, 255))
    lane_mask   = cv2.bitwise_or(white_mask, yellow_mask)

    gray    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    masked  = cv2.bitwise_and(gray, lane_mask)
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    raw = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
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
    left_segs  = [s for s in filtered if (s[0] + s[2]) // 2 < cx]
    right_segs = [s for s in filtered if (s[0] + s[2]) // 2 >= cx]

    result = {}
    for key, segs in (("solid_line_left", left_segs), ("solid_line_right", right_segs)):
        if not segs:
            continue
        is_solid, coords = _classify_and_merge(segs, roi_top, p["solid_coverage"])
        if is_solid and coords:
            result[key] = coords

    return result


def _classify_and_merge(segments, roi_top, solid_cov):
    all_y = [y for _, y1, _, y2 in segments for y in (y1, y2)]
    y_span = max(all_y) - min(all_y)
    if y_span == 0:
        return False, None

    total_covered = sum(abs(y2 - y1) for _, y1, _, y2 in segments)
    is_solid = (total_covered / y_span) >= solid_cov

    all_x = [x for x1, _, x2, _ in segments for x in (x1, x2)]
    points = np.array(list(zip(all_x, all_y)), dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten().tolist()

    if abs(vy) < 1e-6:
        return is_solid, None

    y_min, y_max = min(all_y), max(all_y)
    t_top = (y_min - cy) / vy
    t_bot = (y_max - cy) / vy
    coords = (int(cx + t_top * vx), y_min + roi_top,
              int(cx + t_bot * vx), y_max + roi_top)
    return is_solid, coords


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _line_length(coords):
    x1, y1, x2, y2 = coords
    return float(np.hypot(x2 - x1, y2 - y1))


def score_frame(detected: dict) -> float:
    """
    Score one (frame, params) pair.
    detect_solid_lines returns at most 2 entries (left + right).

      0 lines → 0   (blind)
      1 line  → length × 1.3
      2 lines → avg_length × 1.5  (ideal: both boundaries)

    Length from fitLine output rewards longer, more continuous solid coverage.
    """
    if not detected:
        return 0.0
    lengths = [_line_length(c) for c in detected.values()]
    avg_len = float(np.mean(lengths))
    return avg_len * (1.3 if len(lengths) == 1 else 1.5)


def score_combo(frames, params) -> float:
    return sum(score_frame(_detect(f, params)) for f in frames) / len(frames)


def _param_tag(p: dict) -> str:
    return (
        f"thr{p['hough_thr']}"
        f"__len{p['min_line_len']}"
        f"__gap{p['max_gap']}"
        f"__ang{p['angle_min']}-{p['angle_max']}"
        f"__cov{p['solid_coverage']}"
    )


# ---------------------------------------------------------------------------
# Apply winner back to line_detector.py
# ---------------------------------------------------------------------------

def apply_winner(winner: dict):
    src = DETECTOR.read_text(encoding="utf-8")

    pairs = [
        (r"^HOUGH_THRESHOLD\s*=.*$",
         f"HOUGH_THRESHOLD = {winner['hough_thr']}"),
        (r"^HOUGH_MIN_LINE_LENGTH\s*=.*$",
         f"HOUGH_MIN_LINE_LENGTH = {winner['min_line_len']}"),
        (r"^HOUGH_MAX_LINE_GAP\s*=.*$",
         f"HOUGH_MAX_LINE_GAP = {winner['max_gap']}"
         "    # validated by geometric sweep"),
        (r"^ANGLE_MIN_DEG\s*=.*$",
         f"ANGLE_MIN_DEG = {winner['angle_min']}         # steeper than this from horizontal = kept"),
        (r"^ANGLE_MAX_DEG\s*=.*$",
         f"ANGLE_MAX_DEG = {winner['angle_max']}         # shallower than this from horizontal = kept"),
        (r"^SOLID_Y_COVERAGE\s*=.*$",
         f"SOLID_Y_COVERAGE = {winner['solid_coverage']}"
         "    # y-span coverage ratio above which a group is classified as solid"),
    ]

    for pattern, replacement in pairs:
        src = re.sub(pattern, replacement, src, flags=re.MULTILINE)

    DETECTOR.write_text(src, encoding="utf-8")
    print(f"Updated {DETECTOR}")


# ---------------------------------------------------------------------------
# Verification images
# ---------------------------------------------------------------------------

def generate_verification(frame_paths, winner: dict):
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    for fp in frame_paths:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        detected = _detect(frame, winner)
        out = frame.copy()
        cv2.line(out, (w // 2, 0), (w // 2, h), (0, 200, 0), 1)

        for label, (x1, y1, x2, y2) in detected.items():
            cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        tag = _param_tag(winner)
        note = f"WINNER | {tag} | {list(detected.keys()) or 'none found'}"
        cv2.putText(out, note, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, note, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imwrite(str(VERIFY_DIR / fp.name), out)

    saved = len(list(VERIFY_DIR.glob("*.jpg")))
    print(f"Saved {saved} verification images -> {VERIFY_DIR}/")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_markdown(winner: dict, score: float, ranked: list):
    n_combos = len(ranked)
    n_frames = len(list(FRAMES_DIR.glob("*.jpg")))

    rows = [
        "# Lane Detector — Winning Parameters",
        "",
        "Generated by `line_crossing/score_lanes.py`.",
        "",
        "**Pipeline**: HSV color mask → gray mask → Canny edges → HoughLinesP",
        "→ angle filter → left/right grouping → coverage ratio → fitLine merge",
        "",
        f"**Sweep**: {n_combos} combinations × {n_frames} test frames",
        "(5 crossing-solid-line videos × 3 frames each; covers bright, medium, and night).",
        "",
        "## Scoring Method",
        "",
        "Each (frame, combo) pair is scored from the `detect_solid_lines()` output:",
        "",
        "- **0 solid lines** → score = 0  *(detector blind)*",
        "- **1 solid line** → score = `line_length × 1.3`",
        "- **2 solid lines** → score = `avg_length × 1.5`  *(ideal: left + right)*",
        "",
        "Line length is the span of the `fitLine`-merged representative line,",
        "so longer = more segment coverage = stronger confidence.",
        "Ranking uses the **average score across all 15 frames**.",
        "",
        "## Winner",
        "",
        f"**Average score: `{score:.2f}`**",
        "",
        "| Parameter | Constant in `line_detector.py` | Winning value |",
        "|---|---|---|",
        f"| HoughLinesP vote threshold | `HOUGH_THRESHOLD` | `{winner['hough_thr']}` |",
        f"| Minimum segment length | `HOUGH_MIN_LINE_LENGTH` | `{winner['min_line_len']}` px |",
        f"| Maximum in-line gap | `HOUGH_MAX_LINE_GAP` | `{winner['max_gap']}` px |",
        f"| Min angle from horizontal | `ANGLE_MIN_DEG` | `{winner['angle_min']}°` |",
        f"| Max angle from horizontal | `ANGLE_MAX_DEG` | `{winner['angle_max']}°` |",
        f"| Solid coverage threshold | `SOLID_Y_COVERAGE` | `{winner['solid_coverage']}` |",
        "",
        "## Full Leaderboard (top 20)",
        "",
        "| Rank | Avg Score | Parameters |",
        "|---|---|---|",
    ]
    for i, (tag, sc) in enumerate(ranked[:20], 1):
        rows.append(f"| {i} | {sc:.2f} | `{tag}` |")

    rows += [
        "",
        "## Files updated",
        "",
        "- `line_crossing/line_detector.py` — constants updated with winning values.",
        "- `verification_best_params/` — 15 annotated frames using the winning combo.",
    ]

    SUMMARY_MD.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {SUMMARY_MD}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    frame_paths = sorted(FRAMES_DIR.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames in {FRAMES_DIR}. Run extract_frames.py first.")
        sys.exit(1)

    frames = [cv2.imread(str(fp)) for fp in frame_paths]
    frames = [f for f in frames if f is not None]
    print(f"Loaded {len(frames)} frames.")

    keys   = list(SWEEP.keys())
    combos = list(itertools.product(*SWEEP.values()))
    print(f"Scoring {len(combos)} parameter combinations ...\n")

    scores = {}
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        avg    = score_combo(frames, params)
        scores[_param_tag(params)] = (avg, params)
        if (i + 1) % 36 == 0:
            print(f"  {i+1}/{len(combos)} scored ...")

    ranked_items = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    ranked_flat  = [(tag, sc) for tag, (sc, _) in ranked_items]

    print("\n=== Top 15 combos ===")
    for i, (tag, sc) in enumerate(ranked_flat[:15], 1):
        print(f"  #{i:2d}  score={sc:7.2f}  {tag}")

    best_tag, (best_score, best_params) = ranked_items[0]
    print(f"\nWINNER: {best_tag}  (score={best_score:.2f})")

    apply_winner(best_params)
    write_markdown(best_params, best_score, ranked_flat)

    print("\nGenerating verification images ...")
    generate_verification(frame_paths, best_params)
    print("\nAll done.")


if __name__ == "__main__":
    main()
