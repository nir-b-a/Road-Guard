"""
Classical OpenCV lane-detection pipeline tuned for the highway ODD.

Pipeline per frame:
  grayscale -> adaptive Canny (median-anchored, floor-clamped) ->
  dynamic VP-driven ROI -> Hough (calibrated to compressed ROI) ->
  slope filter -> stateful x-intercept partition -> per-side polyfit ->
  moving-average + patience-bounded dead reckoning.

Per-frame side outputs include a solid/dashed classification derived from
the vertical gap pattern of this frame's raw Hough segments.

Highway-ODD design choices:
  * No CLAHE: amplifies sand / dust / asphalt texture into spurious edges.
  * Floor-clamped adaptive Canny: tracks scene exposure, but never permissive
    enough to admit surface noise in bright environments.
  * Dynamic ROI: top edge tracks the smoothed vanishing point so the ROI
    breathes with road pitch (hills / camera tilt) instead of clipping the
    horizon on inclines or admitting sky on declines.
  * Stateful x-intercept tracker: lane lines crossing the screen center
    during a lane change keep their identity instead of swapping sides.
  * Per-frame solid/dashed classifier: looks at gaps between raw Hough
    segments per side and labels each line accordingly.
"""

from collections import Counter, deque

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# Canny
# -----------------------------------------------------------------------------
CANNY_SIGMA      = 0.33
CANNY_LOW_FLOOR  = 80
CANNY_HIGH_FLOOR = 160

# -----------------------------------------------------------------------------
# ROI geometry — calibrated for 1920x1080 dashcam mount
# -----------------------------------------------------------------------------
ROI_BOTTOM_Y      = 1080
DEFAULT_ROI_TOP_Y = 750     # fallback until vanishing point locks on

# Calibration edge slopes derived from the original trapezoid
#   left edge:  (0, 1080) -> (584, 750)    -> dx per unit drop = 584/330
#   right edge: (1450, 1080) -> (1154, 750) -> dx per unit drop = (1450-1154)/330
_LEFT_EDGE_DX_PER_DROP  = 584.0 / 330.0          # ~1.770
_RIGHT_EDGE_DX_PER_DROP = (1450.0 - 1154.0) / 330.0  # ~0.897
_ROI_X_BOTTOM_LEFT      = 0
_ROI_X_BOTTOM_RIGHT     = 1450


def _roi_vertices_for_top_y(top_y: int) -> np.ndarray:
    """
    Build a trapezoidal ROI whose top edge sits at top_y, preserving the
    calibration edge slopes anchored at the static bottom corners.
    """
    drop = ROI_BOTTOM_Y - top_y
    left_top_x  = int(round(_ROI_X_BOTTOM_LEFT  + drop * _LEFT_EDGE_DX_PER_DROP))
    right_top_x = int(round(_ROI_X_BOTTOM_RIGHT - drop * _RIGHT_EDGE_DX_PER_DROP))
    return np.array(
        [[
            (_ROI_X_BOTTOM_LEFT,  ROI_BOTTOM_Y),
            (left_top_x,          top_y),
            (right_top_x,         top_y),
            (_ROI_X_BOTTOM_RIGHT, ROI_BOTTOM_Y),
        ]],
        dtype=np.int32,
    )


# Exposed for callers that want to draw the ROI overlay (cold-start version).
ROI_VERTICES = _roi_vertices_for_top_y(DEFAULT_ROI_TOP_Y)

# -----------------------------------------------------------------------------
# Hough — calibrated to the compressed near-field ROI
# -----------------------------------------------------------------------------
HOUGH_RHO              = 2
HOUGH_THETA            = np.pi / 180
HOUGH_THRESHOLD        = 100
HOUGH_MIN_LINE_LENGTH  = 100
HOUGH_MAX_LINE_GAP     = 30

# Reject near-horizontal segments (chevrons, bumpers, shadow edges).
SLOPE_MIN_ABS = 0.5

# -----------------------------------------------------------------------------
# Temporal smoothing + dead reckoning
# -----------------------------------------------------------------------------
FIT_HISTORY_LEN   = 15
MAX_MISSED_FRAMES = 12

# Tracker association: raw Hough segments must extrapolate to within this many
# px of a tracked lane intercept (at y=1080) to be assigned to that lane.
INTERCEPT_MAX_DIST_PX = 400

# -----------------------------------------------------------------------------
# Tier 3 — Dynamic ROI from vanishing point
# -----------------------------------------------------------------------------
VP_HISTORY_LEN = 5   # rolling average of recent VP y-values
VP_Y_OFFSET    = 50  # ROI top = smoothed VP y + this offset
# Sanity bounds: reject VPs above the realistic horizon or so low the ROI
# would collapse. Tunable to the camera mount; safe for our 1080p dashcam.
VP_Y_MIN       = 200
VP_Y_MAX       = ROI_BOTTOM_Y - 200

# -----------------------------------------------------------------------------
# Tier 3 — solid vs dashed classifiers (A/B/C testbed)
# -----------------------------------------------------------------------------
CLASSIFIER_BASELINE = "BASELINE"   # Hough-segment vertical-gap (the original)
CLASSIFIER_STRIP    = "STRIP"      # 10-band pixel-density on masked edges
CLASSIFIER_GRADIENT = "GRADIENT"   # per-row max|grad| in a 60-px corridor

# Baseline (Hough-segment) classifier — max gap between segment tops/bottoms.
DASHED_GAP_THRESHOLD_PX = 80

# Strip classifier — split ROI into STRIP_N_BANDS vertical bands; count
# Canny edge pixels inside a 60-px corridor around the polyfit per band.
#
# Tuning (refined after the A/B/C test):
#   * Relative thresholding: per-band threshold is 0.6 x recent rolling mean
#     of total density, so the classifier compares this frame to the road's
#     own recent baseline instead of a hard-coded number. Robust to tunnels.
#   * Dual-threshold score hysteresis: per-side solid_score 0..100, +15 on
#     a "looks solid" frame, -10 on "looks dashed" frame, flips to SOLID at
#     >=80, DASHED at <=20, maintain in between. Eliminates beeping.
#   * Null-gate: when total edge density across all bands is < MIN_ROAD_SIGNAL
#     the classifier returns "unknown" -- the road has no detectable paint
#     signal so we refuse to guess.
STRIP_N_BANDS                = 10
STRIP_DENSITY_HIST_LEN       = 10
STRIP_DENSITY_THRESH_RATIO   = 0.60
STRIP_HIGH_BAND_FRAC         = 0.70
STRIP_MIN_ROAD_SIGNAL_TOTAL  = 80
STRIP_BAND_MIN_PX_FALLBACK   = 20   # cold-start before density history fills

# Dual-threshold solid-score hysteresis for STRIP mode.
SOLID_SCORE_HIT      = 15
SOLID_SCORE_MISS     = 10
SOLID_SCORE_CAP      = 100
SOLID_SCORE_HIGH     = 80
SOLID_SCORE_LOW      = 20

# Gradient-corridor classifier — per row, max|gradient| within a 60-px
# corridor. Solid if the fraction of "high-gradient" rows is >= the SOLID
# threshold; dashed if <= the DASHED threshold; ambiguous in between (hold
# previous verdict via hysteresis).
GRADIENT_CORRIDOR_PX  = 60
GRADIENT_HIGH_THRESH  = 50.0   # max|grad| at/above this counts as paint row
GRADIENT_FRAC_SOLID   = 0.60
GRADIENT_FRAC_DASHED  = 0.40

# Type hysteresis (applies to all three modes for fair comparison):
# the emitted label is the majority vote over the last N raw classifications.
TYPE_HYSTERESIS_LEN   = 10

# -----------------------------------------------------------------------------
# Stability — health-score hysteresis (anti-flicker)
# -----------------------------------------------------------------------------
# Each side has a 0..HEALTH_CAP health score. +HEALTH_HIT on a valid sanity-
# checked fit, -HEALTH_MISS on a miss (or sanity-rejected fit). The visible
# flag flips True when health crosses up through HIGH_THRESHOLD and back to
# False only when it falls through LOW_THRESHOLD — so a single noisy frame
# can't flicker the output line on/off.
HEALTH_HIT            = 20
HEALTH_MISS           = 10
HEALTH_CAP            = 100
HEALTH_HIGH_THRESHOLD = 80   # health >= this -> visible True
HEALTH_LOW_THRESHOLD  = 20   # health <= this -> visible False

# -----------------------------------------------------------------------------
# Stability — slope-variance sanity check
# -----------------------------------------------------------------------------
# Reject a fresh polyfit if its slope differs from the deque mean slope by
# more than this absolute delta. Catches Hough hallucinations on tunnel
# entries and harsh shadow transitions before they poison the moving average.
# Bypassed on cold start (empty deque) so the first valid fit can seed it.
MAX_SLOPE_DELTA = 0.3

# -----------------------------------------------------------------------------
# Diagnostic — perpendicular-corridor gradient audit
# -----------------------------------------------------------------------------
# Probe the raw grayscale around our smoothed polyfit to see how strong the
# paint-vs-asphalt gradient signal actually is per y-row. Used to decide
# whether the existing solid/dashed classifier just needs threshold tuning
# or whether the underlying signal is too noisy and a matched-filter
# replacement is warranted. Off by default; enable via LaneDetector(diag=True).
DIAG_CORRIDOR_PX        = 60   # full corridor width centered on x_center
DIAG_ROW_STEP           = 1    # sample every Nth y-row (1 = every row)
DIAG_PAINT_THRESHOLD    = 30   # max|grad| at or above this counts as "paint"
DIAG_DUMP_EVERY_N       = 60   # full per-row dump every Nth frame (~2s at 30fps)


# Per-segment type alias for clarity.
Segment = tuple[int, int, int, int]  # (x1, y1, x2, y2)


def _region_of_interest(edges: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Mask the single-channel edge image to the polygon defined by vertices."""
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(edges, mask)


def _classify_solidity(segments: list[Segment]) -> str:
    """
    Classify a side as 'solid' or 'dashed' from this frame's raw Hough
    segments. Sort by vertical position; if the largest vertical gap
    between consecutive segments exceeds DASHED_GAP_THRESHOLD_PX, it's dashed.

    Returns 'solid', 'dashed', or 'unknown' (no segments).
    """
    if not segments:
        return "unknown"
    if len(segments) == 1:
        return "solid"  # one continuous chunk implies no internal gap

    # Each segment occupies [min(y1,y2) .. max(y1,y2)] vertically.
    sorted_segs = sorted(segments, key=lambda s: min(s[1], s[3]))
    max_gap = 0
    for i in range(len(sorted_segs) - 1):
        curr_bottom = max(sorted_segs[i][1],   sorted_segs[i][3])
        next_top    = min(sorted_segs[i+1][1], sorted_segs[i+1][3])
        gap = next_top - curr_bottom
        if gap > max_gap:
            max_gap = gap
    return "dashed" if max_gap > DASHED_GAP_THRESHOLD_PX else "solid"


def _compute_vp_y(left_coefs: np.ndarray, right_coefs: np.ndarray) -> float | None:
    """
    Vanishing-point y from two linear fits in the form x = A*y + B (which is
    exactly what np.polyfit(y, x, 1) returns). Intersection:
        A_L * y + B_L = A_R * y + B_R
        => y_vp = (B_R - B_L) / (A_L - A_R)
    Returns None when the fits are near-parallel or the result is outside
    plausible image-space bounds.
    """
    A_left,  B_left  = float(left_coefs[0]),  float(left_coefs[1])
    A_right, B_right = float(right_coefs[0]), float(right_coefs[1])
    denom = A_left - A_right
    if abs(denom) < 1e-3:
        return None
    y_vp = (B_right - B_left) / denom
    if not (VP_Y_MIN < y_vp < VP_Y_MAX):
        return None
    return float(y_vp)


class LaneDetector:
    """
    Stateful classical lane detector. Hold one instance for the lifetime of
    a video stream so the moving-average history, x-intercept tracker, and
    vanishing-point tracker accumulate across frames.
    """

    def __init__(
        self,
        history_len: int = FIT_HISTORY_LEN,
        max_missed_frames: int = MAX_MISSED_FRAMES,
        diag: bool = False,
        classifier_mode: str = CLASSIFIER_BASELINE,
    ):
        self._diag = diag
        self._frame_idx: int = -1   # incremented at the start of each detect_lanes call
        if classifier_mode not in (CLASSIFIER_BASELINE, CLASSIFIER_STRIP, CLASSIFIER_GRADIENT):
            raise ValueError(f"unknown classifier_mode: {classifier_mode!r}")
        self.classifier_mode = classifier_mode
        self.left_fit_history:  deque[np.ndarray] = deque(maxlen=history_len)
        self.right_fit_history: deque[np.ndarray] = deque(maxlen=history_len)

        # Patience counters.
        self._max_missed   = max_missed_frames
        self._left_missed  = 0
        self._right_missed = 0

        # Per-side x-intercept trackers (None until first successful fit).
        self.left_intercept:  float | None = None
        self.right_intercept: float | None = None

        # Tier 3: vanishing-point tracker. Pushed to whenever both sides
        # produced a fresh per-frame polyfit; the rolling mean drives the
        # next frame's dynamic ROI top edge.
        self.vp_history: deque[float] = deque(maxlen=VP_HISTORY_LEN)
        self._smoothed_vp_y:     float | None = None
        self._current_roi_top_y: int = DEFAULT_ROI_TOP_Y

        # Tier 3: classification state. BASELINE / GRADIENT push raw verdicts
        # into a per-side deque; emitted label = 10-frame majority vote.
        # STRIP mode bypasses that and uses its own dual-threshold score
        # hysteresis (solid_score_*) + relative density history per side.
        self._type_history_left:  deque[str] = deque(maxlen=TYPE_HYSTERESIS_LEN)
        self._type_history_right: deque[str] = deque(maxlen=TYPE_HYSTERESIS_LEN)
        self._last_left_type:  str | None = None
        self._last_right_type: str | None = None

        # STRIP-mode dual-threshold score (0..100) and rolling per-frame
        # total density history (length 10) per side. Drive the relative
        # threshold + hysteresis logic in _classify_strip.
        self.solid_score_left:  int = 0
        self.solid_score_right: int = 0
        self._strip_density_hist_left:  deque[int] = deque(maxlen=STRIP_DENSITY_HIST_LEN)
        self._strip_density_hist_right: deque[int] = deque(maxlen=STRIP_DENSITY_HIST_LEN)

        # Stability: per-side health scores + hysteresis visibility flags.
        # detect_lanes only emits a line for a side when its visible flag
        # is True. The flags use dual thresholds (HIGH/LOW) so single-frame
        # noise can't flicker the output. Health is also decremented when
        # the slope-variance sanity check rejects a hallucinated polyfit.
        self.left_health:   int  = 0
        self.right_health:  int  = 0
        self.left_visible:  bool = False
        self.right_visible: bool = False

    def detect_lanes(self, frame: np.ndarray) -> dict[str, object]:
        """
        Returns a flat dict with up to four keys per side:
            "left_line":  (x_bottom, y_bottom, x_top, y_top)
            "left_type":  "solid" | "dashed"
            "right_line": (x_bottom, y_bottom, x_top, y_top)
            "right_type": "solid" | "dashed"
        Missing side -> both its _line and _type keys are absent.
        """
        self._frame_idx += 1

        # --- preprocessing -------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Expose for external classifier plug-ins (lab runner) — these point
        # at the per-frame intermediate images so plug-ins don't have to
        # reproduce the Canny/ROI chain.
        self.last_gray = gray
        self.last_bgr  = frame   # exposed for HSV-color plug-in classifiers

        v = float(np.median(gray))
        low  = max(CANNY_LOW_FLOOR,  int((1.0 - CANNY_SIGMA) * v))
        high = max(CANNY_HIGH_FLOOR, int((1.0 + CANNY_SIGMA) * v))
        high = min(255, high)

        edges = cv2.Canny(gray, low, high)

        # --- dynamic ROI (Tier 3) -----------------------------------------
        roi_top_y = self._current_roi_top_y
        masked    = _region_of_interest(edges, _roi_vertices_for_top_y(roi_top_y))
        # Expose for external plug-in classifiers.
        self.last_masked    = masked
        self.last_roi_top_y = roi_top_y

        # --- Hough ---------------------------------------------------------
        raw_lines = cv2.HoughLinesP(
            masked,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=HOUGH_THRESHOLD,
            lines=np.array([]),
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )

        # --- stateful partition (now returns segments per side) ------------
        left_segs, right_segs = self._partition(raw_lines, frame.shape[1])

        # --- Tier 3: per-frame solid/dashed classification ----------------
        # Compute raw verdict per the active classifier mode (or None when
        # the mode needs data we don't yet have). Hysteresis applied below
        # after the per-side polyfit, so STRIP/GRADIENT can use fresh fits.
        raw_left_type  = self._classify_baseline(left_segs)  if self.classifier_mode == CLASSIFIER_BASELINE else None
        raw_right_type = self._classify_baseline(right_segs) if self.classifier_mode == CLASSIFIER_BASELINE else None

        # --- per-side: sanity-checked polyfit + health hysteresis ---------
        # _step_side applies the slope-variance check, updates the deque,
        # adjusts the miss counter, bumps health up/down, and flips the
        # hysteresis visibility flag.
        fresh_left, self._left_missed, self.left_health, self.left_visible = (
            self._step_side(
                left_segs, self.left_fit_history,
                self._left_missed, self.left_health, self.left_visible,
            )
        )
        if self._left_missed > self._max_missed:
            self.left_fit_history.clear()
            self.left_intercept  = None
            self._last_left_type = None
            self._type_history_left.clear()
            self.left_health     = 0
            self.left_visible    = False
            self.solid_score_left = 0
            self._strip_density_hist_left.clear()

        fresh_right, self._right_missed, self.right_health, self.right_visible = (
            self._step_side(
                right_segs, self.right_fit_history,
                self._right_missed, self.right_health, self.right_visible,
            )
        )
        if self._right_missed > self._max_missed:
            self.right_fit_history.clear()
            self.right_intercept  = None
            self._last_right_type = None
            self._type_history_right.clear()
            self.right_health     = 0
            self.right_visible    = False
            self.solid_score_right = 0
            self._strip_density_hist_right.clear()

        # --- internal lines from history (always, ungated) -----------------
        # The x-intercept tracker and VP estimator need fresh estimates even
        # when the visibility hysteresis is suppressing output rendering.
        left_internal  = self._line_from_history(self.left_fit_history,  roi_top_y)
        right_internal = self._line_from_history(self.right_fit_history, roi_top_y)

        if left_internal is not None:
            self.left_intercept = float(left_internal[0])
        if right_internal is not None:
            self.right_intercept = float(right_internal[0])

        # --- mode-specific classification (STRIP / GRADIENT) --------------
        # STRIP runs its own internal score-based hysteresis and writes
        # directly to _last_*_type; GRADIENT and BASELINE go through the
        # 10-frame majority-vote deque below.
        if self.classifier_mode == CLASSIFIER_STRIP:
            if self.left_fit_history:
                coefs = np.mean(np.stack(self.left_fit_history, axis=0), axis=0)
                self._classify_strip("left", masked, coefs, roi_top_y)
            if self.right_fit_history:
                coefs = np.mean(np.stack(self.right_fit_history, axis=0), axis=0)
                self._classify_strip("right", masked, coefs, roi_top_y)
        elif self.classifier_mode == CLASSIFIER_GRADIENT:
            if self.left_fit_history:
                coefs = np.mean(np.stack(self.left_fit_history, axis=0), axis=0)
                raw_left_type = self._classify_gradient(gray, coefs, roi_top_y)
            if self.right_fit_history:
                coefs = np.mean(np.stack(self.right_fit_history, axis=0), axis=0)
                raw_right_type = self._classify_gradient(gray, coefs, roi_top_y)

        # --- 10-frame majority-vote hysteresis (skip for STRIP) ----------
        if self.classifier_mode != CLASSIFIER_STRIP:
            if raw_left_type is not None:
                self._type_history_left.append(raw_left_type)
            if self._type_history_left:
                self._last_left_type = Counter(self._type_history_left).most_common(1)[0][0]

            if raw_right_type is not None:
                self._type_history_right.append(raw_right_type)
            if self._type_history_right:
                self._last_right_type = Counter(self._type_history_right).most_common(1)[0][0]

        # --- diagnostic gradient audit (off unless diag=True) -------------
        if self._diag:
            if self.left_fit_history:
                self._gradient_audit(
                    gray, "left",
                    np.mean(np.stack(self.left_fit_history, axis=0), axis=0),
                    roi_top_y,
                )
            if self.right_fit_history:
                self._gradient_audit(
                    gray, "right",
                    np.mean(np.stack(self.right_fit_history, axis=0), axis=0),
                    roi_top_y,
                )

        # --- Tier 3: vanishing-point update + dynamic ROI advance ----------
        # Compute VP only when BOTH sides got a fresh per-frame fit. When
        # only one side is fresh we reuse the previous smoothed VP (per spec).
        if fresh_left and fresh_right:
            vp_y = _compute_vp_y(self.left_fit_history[-1], self.right_fit_history[-1])
            if vp_y is not None:
                self.vp_history.append(vp_y)
                self._smoothed_vp_y = float(np.mean(self.vp_history))

        if self._smoothed_vp_y is not None:
            candidate_top = int(round(self._smoothed_vp_y + VP_Y_OFFSET))
            # Clamp so the ROI never collapses or pushes above the horizon.
            candidate_top = max(VP_Y_MIN + VP_Y_OFFSET, candidate_top)
            candidate_top = min(ROI_BOTTOM_Y - 100,     candidate_top)
            self._current_roi_top_y = candidate_top

        # --- build flat result dict — visibility-gated --------------------
        # Internal trackers stayed alive above; here we only EMIT lines for
        # sides whose health-hysteresis flag is True. This is the anti-
        # flicker guarantee: a marginal frame won't toggle the rendered line.
        result: dict[str, object] = {}
        if self.left_visible and left_internal is not None:
            result["left_line"] = left_internal
            result["left_type"] = self._last_left_type or "unknown"
        if self.right_visible and right_internal is not None:
            result["right_line"] = right_internal
            result["right_type"] = self._last_right_type or "unknown"
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _step_side(
        segs: list[Segment],
        history: deque,
        missed: int,
        health: int,
        visible: bool,
    ) -> tuple[bool, int, int, bool]:
        """
        Process one side's raw Hough segments for the current frame.

        Behavior:
          * Computes np.polyfit(y, x, 1) when segs has >= 2 endpoint pairs.
          * Slope-variance sanity check: if the deque is non-empty and the
            new slope differs from the deque-mean slope by more than
            MAX_SLOPE_DELTA, the fit is rejected and the frame is treated
            as a miss. Bypassed on cold start (empty deque).
          * On acceptance: appends to deque, resets miss counter, +HEALTH_HIT.
          * On miss/reject: +1 miss counter, -HEALTH_MISS.
          * Hysteresis: health >= HIGH_THRESHOLD flips visible True;
            health <= LOW_THRESHOLD flips it False; in between, maintain.

        Returns the new (fresh, missed, health, visible).
        The history deque is mutated in place when a fit is accepted.
        """
        fresh = False
        if segs:
            xs = [c for seg in segs for c in (seg[0], seg[2])]
            ys = [c for seg in segs for c in (seg[1], seg[3])]
            if len(xs) >= 2:
                new_fit = np.polyfit(ys, xs, deg=1)
                # Sanity gate — bypass on cold start.
                if not history:
                    accept = True
                else:
                    mean_slope = float(np.mean([c[0] for c in history]))
                    accept = abs(float(new_fit[0]) - mean_slope) <= MAX_SLOPE_DELTA
                if accept:
                    history.append(new_fit)
                    missed = 0
                    health = min(HEALTH_CAP, health + HEALTH_HIT)
                    fresh  = True

        if not fresh:
            missed += 1
            health  = max(0, health - HEALTH_MISS)

        # Hysteresis gatekeeper
        if health >= HEALTH_HIGH_THRESHOLD:
            visible = True
        elif health <= HEALTH_LOW_THRESHOLD:
            visible = False
        # else: maintain current visible state

        return fresh, missed, health, visible

    def _partition(
        self, raw_lines: np.ndarray | None, frame_width: int,
    ) -> tuple[list[Segment], list[Segment]]:
        """
        Slope + x-intercept-tracker partition. Returns lists of raw Hough
        segments (4-tuples) per side, preserving segment identity so the
        downstream solid/dashed classifier can inspect vertical gaps.
        """
        left_segs:  list[Segment] = []
        right_segs: list[Segment] = []

        if raw_lines is None:
            return left_segs, right_segs

        mid_x         = frame_width // 2
        left_tracked  = self.left_intercept  is not None
        right_tracked = self.right_intercept is not None

        for line in raw_lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1 or y2 == y1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < SLOPE_MIN_ABS:
                continue

            x_at_bottom = x1 + (ROI_BOTTOM_Y - y1) * (x2 - x1) / (y2 - y1)

            if left_tracked and right_tracked:
                d_left  = abs(x_at_bottom - self.left_intercept)
                d_right = abs(x_at_bottom - self.right_intercept)
                if d_left <= d_right and d_left < INTERCEPT_MAX_DIST_PX:
                    left_segs.append((int(x1), int(y1), int(x2), int(y2)))
                elif d_right < INTERCEPT_MAX_DIST_PX:
                    right_segs.append((int(x1), int(y1), int(x2), int(y2)))
            elif left_tracked:
                if abs(x_at_bottom - self.left_intercept) < INTERCEPT_MAX_DIST_PX:
                    left_segs.append((int(x1), int(y1), int(x2), int(y2)))
                elif slope > 0 and x1 > mid_x and x2 > mid_x:
                    right_segs.append((int(x1), int(y1), int(x2), int(y2)))
            elif right_tracked:
                if abs(x_at_bottom - self.right_intercept) < INTERCEPT_MAX_DIST_PX:
                    right_segs.append((int(x1), int(y1), int(x2), int(y2)))
                elif slope < 0 and x1 < mid_x and x2 < mid_x:
                    left_segs.append((int(x1), int(y1), int(x2), int(y2)))
            else:
                # Cold start: slope sign + width/2 rule.
                if slope < 0 and x1 < mid_x and x2 < mid_x:
                    left_segs.append((int(x1), int(y1), int(x2), int(y2)))
                elif slope > 0 and x1 > mid_x and x2 > mid_x:
                    right_segs.append((int(x1), int(y1), int(x2), int(y2)))

        return left_segs, right_segs

    # ------------------------------------------------------------------
    # Classifier implementations — A / B / C
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_baseline(segments: list[Segment]) -> str | None:
        """A: original Hough-segment vertical-gap classifier."""
        if not segments:
            return None
        return _classify_solidity(segments)

    def _classify_strip(
        self,
        side: str,
        masked_edges: np.ndarray,
        coefs: np.ndarray,
        roi_top_y: int,
    ) -> None:
        """
        B (refined): per-band Canny edge density inside a 60-px corridor
        around the polyfit, fed into three stability layers:

          1. Null-gate: if total density across all bands < STRIP_MIN_ROAD_
             SIGNAL_TOTAL the road has no detectable paint; emit "unknown"
             via _last_*_type=None (renderer skips the label).
          2. Relative threshold: per-band threshold is
             STRIP_DENSITY_THRESH_RATIO * mean(recent total densities) /
             STRIP_N_BANDS. Tracks tunnel / dusk shifts automatically.
          3. Dual-threshold solid_score hysteresis: +HIT on "looks solid",
             -MISS on "looks dashed"; flip to SOLID at >=HIGH, DASHED at
             <=LOW, maintain in between.

        Writes the final label directly to self._last_left_type /
        self._last_right_type (no caller-side deque hysteresis applied).
        """
        poly = np.poly1d(coefs)
        H, W = masked_edges.shape[:2]
        band_h = max(1, (ROI_BOTTOM_Y - roi_top_y) // STRIP_N_BANDS)
        half_corridor = GRADIENT_CORRIDOR_PX // 2

        band_densities: list[int] = []
        for band in range(STRIP_N_BANDS):
            y0 = roi_top_y + band * band_h
            y1 = min(ROI_BOTTOM_Y, y0 + band_h)
            if y1 <= y0:
                continue
            x_center = int(poly((y0 + y1) // 2))
            x_lo = max(0, x_center - half_corridor)
            x_hi = min(W, x_center + half_corridor)
            band_slice = masked_edges[y0:y1, x_lo:x_hi]
            if band_slice.size == 0:
                continue
            band_densities.append(int(np.count_nonzero(band_slice)))

        if not band_densities:
            return  # no usable bands this frame — leave state unchanged

        total = sum(band_densities)

        # --- (1) Null-gate ------------------------------------------------
        # When the road shows no detectable paint at all (sparse / missing /
        # totally washed out) we refuse to guess. Don't push to history,
        # don't move the score; clear the emitted label.
        if total < STRIP_MIN_ROAD_SIGNAL_TOTAL:
            if side == "left":
                self._last_left_type = None
            else:
                self._last_right_type = None
            return

        # --- (2) Relative thresholding -----------------------------------
        # Per-side rolling history of recent per-frame total densities.
        # Per-band threshold = ratio * mean(history) / N_BANDS.
        if side == "left":
            hist = self._strip_density_hist_left
        else:
            hist = self._strip_density_hist_right
        hist.append(total)

        if len(hist) < 3:
            # Cold start: not enough baseline to derive a relative threshold.
            per_band_thresh = float(STRIP_BAND_MIN_PX_FALLBACK)
        else:
            per_band_thresh = (
                STRIP_DENSITY_THRESH_RATIO * float(np.mean(hist)) / STRIP_N_BANDS
            )

        high_band_count = sum(1 for d in band_densities if d > per_band_thresh)
        strip_density_is_high = (
            high_band_count / len(band_densities) >= STRIP_HIGH_BAND_FRAC
        )

        # --- (3) Dual-threshold solid_score hysteresis -------------------
        if side == "left":
            score = self.solid_score_left
        else:
            score = self.solid_score_right

        if strip_density_is_high:
            score = min(SOLID_SCORE_CAP, score + SOLID_SCORE_HIT)
        else:
            score = max(0, score - SOLID_SCORE_MISS)

        prev = self._last_left_type if side == "left" else self._last_right_type
        if score >= SOLID_SCORE_HIGH:
            label = "solid"
        elif score <= SOLID_SCORE_LOW:
            label = "dashed"
        else:
            # In the dead-band — maintain previous emitted label.
            label = prev if prev in ("solid", "dashed") else "dashed"

        if side == "left":
            self.solid_score_left = score
            self._last_left_type   = label
        else:
            self.solid_score_right = score
            self._last_right_type   = label

    @staticmethod
    def _classify_gradient(
        gray: np.ndarray,
        coefs: np.ndarray,
        roi_top_y: int,
    ) -> str | None:
        """
        C: per y-row, max|gradient| inside a 60-px corridor around the
        polyfit. Solid if >= GRADIENT_FRAC_SOLID of rows are above
        GRADIENT_HIGH_THRESH; dashed if <= GRADIENT_FRAC_DASHED; ambiguous
        in between (returns None so the hysteresis holds the previous label).
        """
        poly = np.poly1d(coefs)
        H, W = gray.shape[:2]
        half_corridor = GRADIENT_CORRIDOR_PX // 2
        high = 0
        total = 0
        for y in range(roi_top_y, ROI_BOTTOM_Y):
            x_center = int(poly(y))
            x_lo = max(0, x_center - half_corridor)
            x_hi = min(W, x_center + half_corridor)
            row_slice = gray[y, x_lo:x_hi]
            if row_slice.size < 4:
                continue
            grad = np.gradient(row_slice.astype(np.float32))
            if float(np.max(np.abs(grad))) > GRADIENT_HIGH_THRESH:
                high += 1
            total += 1
        if total == 0:
            return None
        frac = high / total
        if frac >= GRADIENT_FRAC_SOLID:
            return "solid"
        if frac <= GRADIENT_FRAC_DASHED:
            return "dashed"
        return None  # ambiguous — let hysteresis hold the previous verdict

    def _gradient_audit(
        self,
        gray: np.ndarray,
        side: str,
        coefs: np.ndarray,
        roi_top_y: int,
    ) -> None:
        """
        Walk every y-row in the ROI along the smoothed polyfit, extract a
        DIAG_CORRIDOR_PX-wide horizontal strip centered on x_center, and
        record max|gradient| inside it.

        Prints a one-line per-frame summary (always when diag on); also dumps
        full per-row max-gradient values every DIAG_DUMP_EVERY_N frames so
        the console doesn't drown in 10k lines per second.
        """
        poly = np.poly1d(coefs)
        H, W = gray.shape[:2]
        half = DIAG_CORRIDOR_PX // 2

        rows: list[tuple[int, float]] = []
        paint = gap = 0
        for y in range(roi_top_y, ROI_BOTTOM_Y, DIAG_ROW_STEP):
            x_center = int(poly(y))
            x_lo = max(0, x_center - half)
            x_hi = min(W, x_center + half)
            row_slice = gray[y, x_lo:x_hi]
            if row_slice.size < 4:
                continue
            grad = np.gradient(row_slice.astype(np.float32))
            max_grad = float(np.max(np.abs(grad)))
            rows.append((y, max_grad))
            if max_grad >= DIAG_PAINT_THRESHOLD:
                paint += 1
            else:
                gap += 1

        if not rows:
            return

        vals = [g for _, g in rows]
        gmin, gmed, gmax = min(vals), float(np.median(vals)), max(vals)
        print(
            f"[diag] frame {self._frame_idx:5d} {side:5s}  "
            f"rows={len(vals):3d}  paint={paint:3d}  gap={gap:3d}  "
            f"min={gmin:5.1f}  median={gmed:5.1f}  max={gmax:5.1f}"
        )

        if self._frame_idx % DIAG_DUMP_EVERY_N == 0:
            for y, g in rows:
                print(f"  [diag-row] frame {self._frame_idx} {side} y={y} max_grad={g:.1f}")

    @staticmethod
    def _line_from_history(
        history: deque[np.ndarray], top_y: int,
    ) -> tuple[int, int, int, int] | None:
        """
        Average the recent polyfit coefficients and evaluate at the ROI
        bottom (y=1080) and the current dynamic ROI top (top_y).
        """
        if not history:
            return None
        avg = np.mean(np.stack(history, axis=0), axis=0)  # [avg_slope, avg_intercept]
        poly = np.poly1d(avg)
        return (
            int(poly(ROI_BOTTOM_Y)), ROI_BOTTOM_Y,
            int(poly(top_y)),        top_y,
        )
