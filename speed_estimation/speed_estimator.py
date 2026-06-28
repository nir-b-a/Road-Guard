"""
Distance & speed estimation -- everything that turns a vehicle's bounding boxes
into a distance series and a speed series.

Layout (top to bottom):
  1. Camera intrinsics        -- focal length from FOV.
  2. Distance estimators      -- bbox -> (depth, lateral): height-based,
                                 ground-plane, combined; and _build_estimator
                                 which picks one and injects the vehicle height.
  3. Per-vehicle distance      -- estimateDistance (-> dist_per_frame) + smoothing.
  4. World-frame speed         -- build_world_runs + estimate_world_speeds: the
                                 production path (Kalman by default), persisted
                                 on the Vehicle objects.
  5. bboxes CSV I/O            -- load_bboxes, for the offline A/B CLI.

The actual differentiation (positions -> velocity) lives in smoothers.py; this
file only does the geometry (bbox -> distance) and the per-vehicle orchestration.

Distance and speed use the SAME combined depth (same estimator, focal length,
and per-class vehicle height), so the stored distance is exactly the geometry the
speed is built from -- a car gets 1.5 m, a bus/truck its own.
"""

import csv
import math
import numpy as np
from scipy.signal import savgol_filter

from Objects.World import World
import Constants
from Constants import (VEHICLE_HEIGHTS, DEFAULT_VEHICLE_HEIGHT_M, DEFAULT_CAMERA_HEIGHT_M,
                       LATERAL_REF_CENTER, LATERAL_REF_NEAR_EDGE)
from speed_estimation.smoothers import (Run, _odd,
                       kalman_speed_runs, savgol_speed_runs, theil_sen_speed_runs)


# ============================================================================
# 1. Camera intrinsics
# ============================================================================

def focal_length_from_fov(image_width: int, fov_horizontal_deg: float = 90.0) -> float:
    """Pinhole focal length in pixels for a given image width and horizontal FOV."""
    return (image_width / 2) / math.tan(math.radians(fov_horizontal_deg / 2))


# ============================================================================
# 2. Distance estimators: bbox -> (depth, lateral) in metres.
#   depth   = forward distance from the camera (always positive)
#   lateral = horizontal offset from the optical axis
#               positive -> target is to the right of the camera
# ============================================================================

class DistanceEstimator:
    """Base class. Subclasses must implement __call__(bbox) -> (depth, lateral) or None."""
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 lateral_ref: str = LATERAL_REF_CENTER):
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.lateral_ref = lateral_ref

    def _u_ref(self, x1: float, x2: float) -> float:
        """Horizontal pixel the lateral offset is measured from (the cross-track
        reference). Shared by every estimator so the choice is made in one place.

        "center" (legacy): the silhouette midpoint (x1+x2)/2. Simple, but NOT a
        fixed 3D point -- as a vehicle's aspect opens (you start seeing its flank)
        the silhouette grows sideways and its midpoint slides along the body by up
        to ~half a vehicle length. Differentiated, that slide reads as false
        lateral speed even for a parked car (the wide-angle blow-up).

        "near_edge": the vertical bbox edge CLOSEST to the optical axis (cx). That
        edge tracks the part of the vehicle nearest the camera's line of sight --
        the side LEAST disturbed by flank-reveal foreshortening, which grows the
        FAR edge -- so it stays much closer to a fixed 3D landmark than the
        midpoint. When the box straddles cx (vehicle ~dead ahead, aspect ~rear-on)
        the slide is negligible, so we fall back to the midpoint there.

        Caveat: an edge is a single bbox coordinate, so it carries more per-frame
        detector jitter than the averaged midpoint -- but that jitter is HIGH-freq
        and the downstream Hampel + Kalman smoothing removes it, whereas the
        midpoint slide is a LOW-freq bias that smoothing CANNOT remove. Trading
        removable noise for an unremovable bias is the win. It remains a heuristic,
        not a true fixed point: a very oblique view can still switch which corner
        is the extreme.
        """
        if self.lateral_ref == LATERAL_REF_NEAR_EDGE and not (x1 <= self.cx <= x2):
            return x1 if abs(x1 - self.cx) < abs(x2 - self.cx) else x2
        return (x1 + x2) / 2.0

    def __call__(self, bbox: tuple[int, int, int, int]) -> tuple[float, float] | None:
        raise NotImplementedError


class HeightBasedDistance(DistanceEstimator):
    """
    Depth from bounding-box height, assuming a fixed real-world vehicle height.

        depth = vehicle_height * fy / pixel_height

    Bias: if the actual vehicle is taller/shorter than vehicle_height_m, the
    distance is systematically wrong (an SUV reads closer than it is).
    """
    def __init__(self, fx, fy, cx, cy, vehicle_height_m: float = DEFAULT_VEHICLE_HEIGHT_M,
                 lateral_ref: str = LATERAL_REF_CENTER):
        super().__init__(fx, fy, cx, cy, lateral_ref)
        self.vehicle_height_m = vehicle_height_m

    def __call__(self, bbox):
        x1, y1, x2, y2 = bbox
        ph = y2 - y1
        if ph <= 0:
            return None
        depth = self.vehicle_height_m * self.fy / ph
        u_ref = self._u_ref(x1, x2)
        lateral = (u_ref - self.cx) * depth / self.fx
        return depth, lateral


class GroundPlaneDistance(DistanceEstimator):
    """
    Depth from the pixel position of the bbox's BOTTOM edge, assuming the
    vehicle sits on a flat road plane at a known camera height.

        depth = camera_height * fy / (y_bottom - cy)

    Pro: needs no assumption about the vehicle's size -- so no bias for SUVs,
         trucks, motorcycles.
    Con: sensitive to camera pitch (suspension bumps, road grade) because cy
         shifts by tan(pitch) * fy pixels.  Sensitive to occluded wheels.
    """
    def __init__(self, fx, fy, cx, cy, camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M,
                 lateral_ref: str = LATERAL_REF_CENTER):
        super().__init__(fx, fy, cx, cy, lateral_ref)
        self.camera_height_m = camera_height_m

    def __call__(self, bbox):
        x1, y1, x2, y2 = bbox
        v_below = y2 - self.cy
        if v_below <= 0:
            return None  # bbox bottom is above the horizon -> can't be on the road
        depth = self.camera_height_m * self.fy / v_below
        u_ref = self._u_ref(x1, x2)
        lateral = (u_ref - self.cx) * depth / self.fx
        return depth, lateral


class CombinedDistance(DistanceEstimator):
    """
    Average of two distance estimators.  Useful as a simple ensemble:
    if the two methods have partially independent errors (e.g.
    HeightBasedDistance is biased by vehicle size, GroundPlaneDistance is
    biased by camera pitch), the average is less wrong than either alone
    in the regimes where they disagree.

    When one estimator returns None for a frame (e.g. ground-plane bbox
    sits above the horizon), the other estimator's value is used directly.
    """
    def __init__(self, estimator_a: "DistanceEstimator",
                       estimator_b: "DistanceEstimator"):
        super().__init__(estimator_a.fx, estimator_a.fy,
                         estimator_a.cx, estimator_a.cy, estimator_a.lateral_ref)
        self.a = estimator_a
        self.b = estimator_b

    def __call__(self, bbox):
        ra = self.a(bbox)
        rb = self.b(bbox)
        if ra is None and rb is None:
            return None
        if ra is None:
            return rb
        if rb is None:
            return ra
        return ((ra[0] + rb[0]) * 0.5, (ra[1] + rb[1]) * 0.5)


def _build_estimator(method: str, fx: float, fy: float, cx: float, cy: float,
                     vehicle_height_m: float, camera_height_m: float,
                     lateral_ref: str = LATERAL_REF_CENTER) -> DistanceEstimator:
    """Pick a distance estimator by name and inject its heights.

    "height" / "ground" / anything-else ("combined") -> the matching estimator.
    Used by BOTH speed paths so they build estimators the same way (and so the
    per-class vehicle height reaches both). `lateral_ref` selects the cross-track
    reference point ("center" / "near_edge") -- it reaches both sub-estimators so
    the choice is consistent under "combined" too.
    """
    hb = HeightBasedDistance(fx, fy, cx, cy, vehicle_height_m=vehicle_height_m,
                             lateral_ref=lateral_ref)
    gp = GroundPlaneDistance(fx, fy, cx, cy, camera_height_m=camera_height_m,
                             lateral_ref=lateral_ref)
    if method == "height":
        return hb
    if method == "ground":
        return gp
    return CombinedDistance(hb, gp)


# ============================================================================
# 3. Per-vehicle distance series (-> vehicle.dist_per_frame)
# ============================================================================

def estimateDistance(world: World, fx: float, fy: float, cx: float, cy: float,
                     camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M):
    """
    Per-frame forward distance (m) for every vehicle, stored in dist_per_frame.

    Uses the SAME 'combined' estimator the world-frame speed path consumes
    (height-based + ground-plane average, with the vehicle's RESOLVED per-class
    height, at the real camera intrinsics) -- so the stored distance is the exact
    geometry the speed is built from, not a separate height-only estimate. The
    estimator's `depth` IS the forward distance; `lateral` is ignored here.

    Frames the estimator can't resolve (zero-height bbox, or bbox above the
    horizon for both methods) are stored as 0.0. This is the RAW depth;
    smooth_distances() smooths it afterwards for the distance output, while the
    speed path differentiates the same raw depth.
    """
    for vehicle in world.vehicles.values():
        height_m = VEHICLE_HEIGHTS.get(vehicle.vehicle_type, DEFAULT_VEHICLE_HEIGHT_M)
        estimator = _build_estimator("combined", fx, fy, cx, cy, height_m, camera_height_m)
        for frame, bbox in vehicle.bounding_box.items():
            r = estimator(bbox)
            vehicle.dist_per_frame[frame] = r[0] if r is not None else 0.0


def _contiguous_runs(frames: list[int]) -> list[list[int]]:
    """Split a sorted list of frame indices into runs of consecutive frames."""
    runs: list[list[int]] = []
    current: list[int] = []
    for f in frames:
        if not current or f == current[-1] + 1:
            current.append(f)
        else:
            runs.append(current)
            current = [f]
    if current:
        runs.append(current)
    return runs


def smooth_distances(world: World,
                     smooth_window: int = Constants.SMOOTH_WINDOW,
                     polyorder: int = Constants.POLYORDER):
    """
    In-place Savitzky-Golay smoothing of vehicle.dist_per_frame.

    Operates per vehicle, per contiguous run of valid (positive) distances.
    Zero / placeholder distances are left alone; they split runs.

    This affects the exported distance CSV and the distance plot, NOT the
    speed estimation (which uses its own smoothing).
    """
    smooth_window = _odd(smooth_window)

    for vehicle in world.vehicles.values():
        valid_frames = sorted(
            f for f, d in vehicle.dist_per_frame.items() if d > 0
        )
        if not valid_frames:
            continue

        for run in _contiguous_runs(valid_frames):
            if len(run) < polyorder + 2:
                continue
            sw = min(smooth_window, _odd(len(run) - 1))
            if sw < polyorder + 2:
                continue
            distances = np.array([vehicle.dist_per_frame[f] for f in run])
            smoothed = savgol_filter(distances, sw, polyorder)
            for i, f in enumerate(run):
                vehicle.dist_per_frame[f] = float(smoothed[i])


# ============================================================================
# 4. World-frame speed (PRODUCTION path -- the only speed path)
#
# Reconstruct the target's ABSOLUTE world position per frame (de-rotate the
# camera-frame depth/lateral by ego heading, translate by ego position), then
# smooth + differentiate ONCE with the chosen smoother. Heading is an integral
# quantity -- smooth even through sharp turns -- so all ego motion cancels when
# we differentiate. Kalman is the default and the only smoother emitting a std.
# ============================================================================

# Each smoother maps a list of world-position Runs -> {frame: (speed, std)}.
# theil_sen returns speed only, so it is wrapped to a 0.0 std.
_SMOOTHERS = {
    Constants.SMOOTHER_KALMAN:   lambda runs, fps, fts: kalman_speed_runs(runs, fps, frame_ts=fts),
    Constants.SMOOTHER_SAVGOL:   lambda runs, fps, fts: savgol_speed_runs(runs, fps),
    Constants.SMOOTHER_THEILSEN: lambda runs, fps, fts: {
        f: (v, 0.0) for f, v in theil_sen_speed_runs(runs, fps).items()
    },
}


def _wide_angle_noise_scale(depth: np.ndarray, lateral: np.ndarray) -> np.ndarray:
    """Per-frame measurement-noise multiplier (>= 1) keyed to the target's bearing.

    bearing = atan(|lateral|/depth); weight w = 1/(1+(bearing_deg/HALF)**POWER),
    floored at MIN_WEIGHT; the noise std is scaled by 1/sqrt(w). On-axis (bearing 0)
    -> w=1 -> scale=1 (no change); wide angle -> w<1 -> scale>1 (trusted less).
    """
    bearing_deg = np.degrees(np.arctan2(np.abs(lateral), np.maximum(depth, 1e-6)))
    w = 1.0 / (1.0 + (bearing_deg / Constants.WIDE_ANGLE_HALF_WEIGHT_DEG)
                     ** Constants.WIDE_ANGLE_POWER)
    w = np.clip(w, Constants.WIDE_ANGLE_MIN_WEIGHT, 1.0)
    return 1.0 / np.sqrt(w)


def _edge_touch_noise_scale(bboxes: list[tuple[int, int, int, int]],
                            image_wh: tuple[int, int]) -> np.ndarray:
    """Per-frame noise multiplier: large on frames whose bbox TOUCHES the image
    border (within Constants.EDGE_MARGIN_PX of any side), else 1.

    A clipped bbox is truncated by the frame, so the visible silhouette is only
    part of the vehicle: its width/height (-> depth) and its horizontal center
    (-> lateral) are both corrupted and jump as the vehicle exits. Returning
    Constants.EDGE_NOISE_SCALE there (default 8x std) makes the smoother nearly
    ignore the frame and coast on its model through the clipped stretch.
    """
    w, h = image_wh
    m = Constants.EDGE_MARGIN_PX
    out = np.ones(len(bboxes))
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        if x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m:
            out[i] = Constants.EDGE_NOISE_SCALE
    return out


def build_world_runs(bboxes_by_frame: dict[int, tuple[int, int, int, int]],
                     estimator: DistanceEstimator,
                     ego_pos: dict[int, tuple[float, float]],
                     ego_heading: dict[int, float],
                     lat_sign: int,
                     min_run_len: int,
                     reweight: bool = False,
                     edge_gate: bool = False,
                     image_wh: tuple[int, int] | None = None) -> list[Run]:
    """Reconstruct target absolute world position (Tx, Ty) per contiguous run.

    Optional per-frame Kalman noise multipliers (Run.meta['noise_scale'], trusted
    LESS where >1) combine multiplicatively:
      `reweight`  -> grows with the target's bearing (wide-angle frames).
      `edge_gate` -> large on frames whose bbox touches the image border
                     (needs `image_wh`=(width,height); ignored if None).
    With both off the runs carry no meta (unchanged behaviour).
    """
    apply_edge = edge_gate and image_wh is not None
    runs: list[Run] = []
    valid = sorted(f for f, b in bboxes_by_frame.items() if b != (0, 0, 0, 0))
    for run in _contiguous_runs(valid):
        flushed: list[list[tuple[int, float, float]]] = [[]]
        for f in run:
            r = estimator(bboxes_by_frame[f])
            if r is None:
                if flushed[-1]:
                    flushed.append([])
            else:
                flushed[-1].append((f, r[0], r[1]))
        for sub in flushed:
            if len(sub) < min_run_len:
                continue
            frames = [s[0] for s in sub]
            depth  = np.array([s[1] for s in sub])
            lateral = np.array([s[2] for s in sub])
            theta = np.array([ego_heading.get(f, 0.0) for f in frames])
            ex = np.array([ego_pos.get(f, (0.0, 0.0))[0] for f in frames])
            ey = np.array([ego_pos.get(f, (0.0, 0.0))[1] for f in frames])
            c, s = np.cos(theta), np.sin(theta)
            Tx = ex + depth * c + lat_sign * lateral * s
            Ty = ey + depth * s - lat_sign * lateral * c
            meta = {}
            if reweight or apply_edge:
                scale = np.ones(len(frames))
                if reweight:
                    scale = scale * _wide_angle_noise_scale(depth, lateral)
                if apply_edge:
                    scale = scale * _edge_touch_noise_scale(
                        [bboxes_by_frame[f] for f in frames], image_wh)
                meta = {"noise_scale": scale}
            runs.append(Run(frames=frames, x=Tx, y=Ty, meta=meta))
    return runs


def estimate_world_speeds(world: World,
                          ego_pos: dict[int, tuple[float, float]],
                          ego_heading: dict[int, float],
                          fps: float,
                          *,
                          fx: float, fy: float, cx: float, cy: float,
                          frame_ts: dict[int, int] | None = None,
                          method: str = "height",
                          smoother: str = Constants.DEFAULT_SMOOTHER,
                          lat_sign: int = Constants.LAT_SIGN,
                          lateral_ref: str = Constants.LATERAL_REF,
                          wide_angle_reweight: bool = Constants.WIDE_ANGLE_REWEIGHT,
                          edge_gate: bool = Constants.EDGE_GATE,
                          image_width: int | None = None,
                          image_height: int | None = None,
                          camera_height_m: float = Constants.DEFAULT_CAMERA_HEIGHT_M,
                          min_track_seconds: float = Constants.MIN_TRACK_SECONDS,
                          min_run_len: int | None = None,
                          store: bool = True
                          ) -> dict[int, dict[int, tuple[float, float]]]:
    """
    Estimate absolute speed (m/s) + uncertainty for every tracked vehicle via
    world-frame reconstruction, and (optionally) persist it on each Vehicle.

    Args:
        world:        the populated World (its .vehicles are iterated).
        ego_pos:      {frame -> (ego_x, ego_y)} world metres.
        ego_heading:  {frame -> ego heading (rad)}.
        fps:          video frame rate (differentiation timebase).
        fx, fy, cx, cy: camera intrinsics.
        method:       "combined" | "height" | "ground" distance estimator.
        smoother:     "kalman" (default) | "savgol" | "theilsen".
        lat_sign:     handedness sign for the world de-rotation (Constants.LAT_SIGN).
        lateral_ref:  cross-track reference point "center" | "near_edge"
                      (Constants.LATERAL_REF). "near_edge" suppresses the
                      wide-angle lateral blow-up by measuring lateral from the bbox
                      edge nearest the optical axis instead of the sliding center.
        wide_angle_reweight: when True (Constants.WIDE_ANGLE_REWEIGHT), inflate the
                      Kalman measurement noise on wide-bearing frames so the
                      smoother trusts them less (leans on the speed established at
                      small angles). False -> uniform trust (unchanged).
        edge_gate:    when True (Constants.EDGE_GATE), heavily down-weight frames
                      whose bbox touches the image border (clipped -> corrupted
                      depth & lateral). Needs image_width/image_height.
        image_width, image_height: video frame size in px, required by edge_gate.
        camera_height_m: ground-plane camera height.
        min_track_seconds: vehicles with fewer real detections than this many
                      seconds' worth of frames are skipped entirely (issue #4).
        min_run_len:  minimum length of a contiguous sub-run the smoother will
                      process. Defaults to max(SMOOTH_WINDOW, DERIV_WINDOW) -- the
                      validated value. Separate from min_track_seconds (the coarse
                      whole-track gate).
        store:        if True, write speed/std back onto each Vehicle (issue #2).

    Returns:
        {vehicle_id: {frame: (speed_mps, speed_std_mps)}}.

    Per-vehicle height (issue #1): from the vehicle's RESOLVED class via
    VEHICLE_HEIGHTS; for a car this is 1.5 m (unchanged single-height behaviour).
    """
    if smoother not in _SMOOTHERS:
        raise ValueError(f"unknown smoother {smoother!r}; "
                         f"choose from {sorted(_SMOOTHERS)}")
    if min_run_len is None:
        min_run_len = max(Constants.SMOOTH_WINDOW, Constants.DERIV_WINDOW)

    # Real (non-placeholder) detections needed before a track is worth estimating.
    min_track_frames = max(1, int(round(min_track_seconds * fps)))
    smooth = _SMOOTHERS[smoother]

    # Alignment guard: every detected bbox frame should have an ego pose. If frames.csv
    # is shorter than the video (e.g. a capture gap), ego_pos/ego_heading .get() silently
    # falls back to 0 and ego motion is NOT cancelled on those frames -> bogus speeds.
    detected = set()
    for v in world.vehicles.values():
        detected.update(f for f, b in v.bounding_box.items() if b != (0, 0, 0, 0))
    missing = sum(1 for f in detected if f not in ego_pos or f not in ego_heading)
    if missing:
        print(f"[world-speed][WARN] {missing}/{len(detected)} detected frames have NO ego "
              f"pose (ego motion not cancelled there). Likely frames.csv shorter than the "
              f"video / a capture gap -> frame-index misalignment.")

    results: dict[int, dict[int, tuple[float, float]]] = {}
    skipped_short = 0

    for vid, vehicle in world.vehicles.items():
        real_frames = sum(1 for b in vehicle.bounding_box.values()
                          if b != (0, 0, 0, 0))
        if real_frames < min_track_frames:
            skipped_short += 1
            continue

        height_m = VEHICLE_HEIGHTS.get(vehicle.vehicle_type, DEFAULT_VEHICLE_HEIGHT_M)
        estimator = _build_estimator(method, fx, fy, cx, cy,
                                     height_m, camera_height_m, lateral_ref)

        runs = build_world_runs(vehicle.bounding_box, estimator,
                                ego_pos, ego_heading, lat_sign, min_run_len,
                                reweight=wide_angle_reweight,
                                edge_gate=edge_gate,
                                image_wh=(image_width, image_height)
                                if image_width and image_height else None)
        if not runs:
            continue

        speed = smooth(runs, fps, frame_ts)
        if not speed:
            continue

        results[vid] = speed
        if store:
            for f, (sp, st) in speed.items():
                vehicle.speed_per_frame[f] = sp
                vehicle.speed_std_per_frame[f] = st

    edge_state = (edge_gate if (image_width and image_height)
                  else f"{edge_gate} (DISABLED: no image size)")
    print(f"[world-speed] estimated {len(results)} vehicle(s) "
          f"(method={method}, smoother={smoother}, lat_sign={lat_sign}, "
          f"lateral_ref={lateral_ref}, wide_angle_reweight={wide_angle_reweight}, "
          f"edge_gate={edge_state}); "
          f"skipped {skipped_short} track(s) shorter than {min_track_seconds}s")
    return results


# ============================================================================
# 5. bboxes CSV I/O (for the offline A/B CLI in robust_speed_pipeline.py)
# ============================================================================

def load_bboxes(path: str) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    """frame,vehicle_id,x1,y1,x2,y2  ->  {vehicle_id: {frame: bbox}}."""
    out: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            try:
                vid = int(row["vehicle_id"])
                fr  = int(row["frame"])
                bbox = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            except (KeyError, ValueError):
                continue
            out.setdefault(vid, {})[fr] = bbox
    return out
