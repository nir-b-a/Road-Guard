import cv2
import numpy as np

# --- Pipeline configuration ---
ROI_CROP_RATIO = 0.5       # fraction of frame height to use (bottom portion)
BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 35
HOUGH_MIN_LINE_LENGTH = 60
HOUGH_MAX_LINE_GAP = 100    # validated by geometric sweep
ANGLE_MIN_DEG = 10         # steeper than this from horizontal = kept
ANGLE_MAX_DEG = 85         # shallower than this from horizontal = kept
SOLID_Y_COVERAGE = 0.5    # y-span coverage ratio above which a group is classified as solid


def detect_solid_lines(frame: np.ndarray, roi_crop_ratio: float = ROI_CROP_RATIO) -> dict[str, tuple]:
    """
    Detects solid lane lines in a dashcam frame.
    Returns a dict mapping line labels to (x1, y1, x2, y2) in full-frame coordinates.
    Returns an empty dict if no solid lines are found.
    """
    h, w = frame.shape[:2]
    roi_top = int(h * (1 - roi_crop_ratio))
    roi = frame[roi_top:, :]

    # TODO: Dynamic horizon detection — replace the fixed roi_top with a per-frame
    # vanishing point estimate so the ROI adapts when the camera tilts or the road curves.

    # Step 1: HSV color masking — isolate white and yellow lane markings
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    yellow_mask = cv2.inRange(hsv, (18, 80, 80), (35, 255, 255))
    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Step 2: Mask gray image, blur, then run Canny edge detection
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, lane_mask)
    blurred = cv2.GaussianBlur(masked, BLUR_KERNEL, 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    # Step 3: Probabilistic Hough transform
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )

    if raw_lines is None:
        return {}

    # Step 4: Filter by angle — remove near-horizontal noise and near-vertical walls
    filtered = _filter_by_angle(raw_lines)
    if not filtered:
        return {}

    # Step 5: Group into left/right by x-midpoint relative to frame center
    cx = w // 2
    left_segs = [s for s in filtered if (s[0] + s[2]) // 2 < cx]
    right_segs = [s for s in filtered if (s[0] + s[2]) // 2 >= cx]

    result = {}
    for key, segs in (("solid_line_left", left_segs), ("solid_line_right", right_segs)):
        if not segs:
            continue
        is_solid, coords = _classify_and_merge(segs, roi_top)
        if is_solid and coords:
            result[key] = coords

    # TODO: Israeli double-line law (קו הפרדה כפול) — if both "solid_line_left" and
    # "solid_line_right" are present, are close together (<~30px apart), and are parallel,
    # flag this as a double solid line with stricter crossing rules in both directions.

    return result


def _filter_by_angle(raw_lines: np.ndarray) -> list[tuple]:
    kept = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        # Accept lines diagonal enough to be lane markings (not road texture or curbs)
        if (ANGLE_MIN_DEG <= angle <= ANGLE_MAX_DEG) or (180 - ANGLE_MAX_DEG <= angle <= 180 - ANGLE_MIN_DEG):
            kept.append((x1, y1, x2, y2))
    return kept


def _classify_and_merge(segments: list[tuple], roi_top: int) -> tuple[bool, tuple | None]:
    """
    Classifies a group of segments as solid or dashed, then merges them into a single
    representative line via linear regression. Returns (is_solid, (x1, y1, x2, y2)).
    Coordinates are translated back to full-frame space via roi_top offset.
    """
    all_y = [y for _, y1, _, y2 in segments for y in (y1, y2)]
    y_span = max(all_y) - min(all_y)
    if y_span == 0:
        return False, None

    # Coverage ratio: total vertical span covered by segments vs total y-span of the group.
    # Solid lines have high coverage; dashed lines leave large gaps.
    # Note: overlapping segments can inflate this slightly, which is acceptable for V1.
    total_y_covered = sum(abs(y2 - y1) for _, y1, _, y2 in segments)
    is_solid = (total_y_covered / y_span) >= SOLID_Y_COVERAGE

    # Merge all segment endpoints into a single best-fit line via OpenCV's fitLine
    all_x = [x for x1, _, x2, _ in segments for x in (x1, x2)]
    points = np.array(list(zip(all_x, all_y)), dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, cx, cy = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten().tolist()

    if abs(vy) < 1e-6:
        return is_solid, None

    # Extrapolate the fitted line to the top and bottom y extent of the group
    y_min, y_max = min(all_y), max(all_y)
    t_top = (y_min - cy) / vy
    t_bot = (y_max - cy) / vy
    x_top = int(cx + t_top * vx)
    x_bot = int(cx + t_bot * vx)

    # Translate ROI-local coordinates back to full-frame coordinates
    coords = (x_top, y_min + roi_top, x_bot, y_max + roi_top)
    return is_solid, coords
