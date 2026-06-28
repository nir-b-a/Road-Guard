from enum import IntEnum

class LPR:
    # Minimum vehicle bounding-box area (pixels²) before attempting plate recognition.
    # ~150×100px — below this the plate is too small to read reliably.
    MIN_VEHICLE_AREA = 15000


# yolo vehicle classes
class DetectClass(IntEnum):
    CAR = 2
    MOTORCYCLE = 3
    BUS = 5
    TRUCK = 7
    TRAFFIC_LIGHT = 9


# Membership sets. IntEnum members compare equal to their int value, so a raw
# `int(box.cls)` from YOLO matches these without conversion.
VEHICLE_CLASSES = frozenset({
    DetectClass.CAR, DetectClass.MOTORCYCLE, DetectClass.BUS, DetectClass.TRUCK,
})
TRAFFIC_LIGHT_CLASSES = frozenset({DetectClass.TRAFFIC_LIGHT})

# List of plain ints passed to YOLO's `classes=` filter. Kept as plain ints (not
# IntEnum members) and in this exact order to be identical to the historical
# `Detection_Classes = [Car, Motorcycle, Bus, Truck, Traffic_Light]`.
DETECTION_CLASSES = [int(c) for c in (
    DetectClass.CAR, DetectClass.MOTORCYCLE, DetectClass.BUS,
    DetectClass.TRUCK, DetectClass.TRAFFIC_LIGHT,
)]


def is_vehicle(class_id: int) -> bool:
    """True if a COCO class id is one of the vehicle classes we track.

    Replaces the scattered `object_type in DetectClass.Vehicle_Classes` checks
    with a single predicate, so callers no longer reach into a class-attribute
    list to ask "is this a vehicle?".
    """
    return class_id in VEHICLE_CLASSES


def is_traffic_light(class_id: int) -> bool:
    """True if a COCO class id is the traffic-light class."""
    return class_id in TRAFFIC_LIGHT_CLASSES


def class_name(class_id: int) -> str:
    """Human-readable name for a class id (e.g. 2 -> 'Car'); 'class<id>' if unknown."""
    try:
        return DetectClass(class_id).name.capitalize()
    except ValueError:
        return f"class{class_id}"


# ── Real-world vehicle heights (metres), keyed by class ───────────────────────
# Used as the height prior in HeightBasedDistance. Resolved PER VEHICLE after
# class re-evaluation (issue #1) so a bus/truck no longer reads as a 1.5 m car.
# CAR stays 1.5 m, so existing car-target simulations are unchanged.
#
# NOTE: the draft New_Constants.py listed TRUCK = 9 m, which is unphysical as a
# height (~6x a real truck) and would make trucks read ~6x too far -> grossly
# wrong speed. Set to 3.5 m here (typical box-truck/semi cab height). Adjust if
# you have a better figure for your footage.
VEHICLE_HEIGHTS = {
    DetectClass.CAR: 1.5,
    DetectClass.MOTORCYCLE: 1.1,
    DetectClass.BUS: 3.2,
    DetectClass.TRUCK: 3.5,
}
DEFAULT_VEHICLE_HEIGHT_M = 1.5   # fallback when a resolved class has no entry
DEFAULT_VEHICLE_WIDTH_M = 1.8
DEFAULT_CAMERA_HEIGHT_M = 1.4    # dashcam mount height (also the CARLA default)

# ── Detection / tracking ──────────────────────────────────────────────────────
# Vehicle detector/tracker is pinned to YOLOv11 (yolo11x). This is the live detection
# model for vehicles + traffic lights; --model can override it for an A/B run. NOTE: the
# yellow/solid lane-line model (weights/phase3_v3_yellowprotect.pt, loaded in main.py) is a
# CUSTOM-TRAINED YOLOv8-seg model and is intentionally NOT bumped to v11 -- swapping it would
# require retraining the lane-type dataset, which is out of scope for the baseline.
YOLO_VERSION = "yolo11x.pt"
CONFIDENCE_LVL = 0.5             # YOLO minimum detection confidence
YOLO_IMGSZ = 1280                # YOLO inference image size (px); 1984 w/o half
YOLO_TRACKER = "botsort.yaml"    # ultralytics tracker config

# ── Bounding-box gap interpolation ────────────────────────────────────────────
MAX_GAP_SECONDS = 0.5            # linearly interpolate bbox gaps up to this long

# ── Camera intrinsics (CARLA dashcam defaults) ────────────────────────────────
FOV_HORIZONTAL_DEG = 90.0        # horizontal field of view (square pixels)

# ── Speed-estimation smoothing (Savitzky-Golay) ───────────────────────────────
SMOOTH_WINDOW = 101               # pre-smoothing savgol window (odd; ~1.7s @30fps)
DERIV_WINDOW = 61                # differentiation savgol window (odd; ~0.7s @30fps)
POLYORDER = 2                    # savgol polynomial order

# ── Distance calculation method ──────────────────────────────────────────────────────
DISTANCE_CALCULATION_METHOD = "height"

# ── World-frame reconstruction ────────────────────────────────────────────────
# Resolves coordinate-system handedness when de-rotating camera-frame (depth,
# lateral) into world coordinates. -1 is the validated winner for CARLA; it MUST
# be re-validated on real Android footage (device axes / mount orientation differ
# -- see ANDROID_DATA_SPEC.md sign note).
LAT_SIGN = -1

# ── Lateral landmark (cross-track reference point) ────────────────────────────
# Which horizontal point of a vehicle's bbox the lateral offset is measured from
# when reconstructing its cross-track world position (the channel that feeds
# speed). The bbox CENTER is simple but is NOT a fixed 3D point: as a vehicle's
# aspect opens (its flank comes into view) the silhouette grows sideways and its
# center slides along the body by up to ~half a vehicle length, which
# differentiates into FALSE lateral speed at wide angles -- even for a parked car
# (the wide-angle blow-up). "near_edge" instead measures from the bbox edge
# nearest the optical axis -- the part of the vehicle least disturbed by
# flank-reveal foreshortening (which grows the FAR edge) -- so it stays far closer
# to a fixed 3D landmark. "center" is the legacy/validated default (sim unchanged);
# switch to near_edge on real footage (main.py: --lateral-ref near_edge) to
# suppress the wide-angle blow-up. Depth is unaffected either way (it rides on
# bbox HEIGHT), so the distance export does not change.
LATERAL_REF_CENTER = "center"
LATERAL_REF_NEAR_EDGE = "near_edge"
LATERAL_REF = LATERAL_REF_CENTER

# ── Wide-angle measurement down-weighting ─────────────────────────────────────
# A target far off the optical axis (wide bearing) has a corrupted reconstruction
# (bbox-center slide, lens distortion, and the depth-vs-ego-motion cancellation is
# worst there), so its world-position MEASUREMENT is less trustworthy. When
# enabled, each frame's Kalman measurement noise is inflated by a factor that
# grows with the target's bearing |atan(lateral/depth)|, so the smoother leans on
# its motion model (the speed established at small angles) instead of chasing the
# wide-angle measurement. OFF by default: the sim/validated path is bit-identical
# until you opt in (main.py: --wide-angle-reweight 1).
#   weight w(beta) = 1 / (1 + (beta_deg / HALF_WEIGHT_DEG) ** POWER), floored at
#   MIN_WEIGHT; the per-frame noise std is multiplied by 1/sqrt(w) (>= 1).
WIDE_ANGLE_REWEIGHT = False        # master toggle (False -> no change to anything)
WIDE_ANGLE_HALF_WEIGHT_DEG = 22.0  # bearing at which a frame's weight halves (noise x sqrt(2))
WIDE_ANGLE_POWER = 2.0             # how sharply weight falls past the half-weight angle
WIDE_ANGLE_MIN_WEIGHT = 0.04       # weight floor -> caps noise inflation at 1/sqrt(0.04) = 5x

# ── Frame-edge gate (clipped-bbox down-weighting) ─────────────────────────────
# When a bbox TOUCHES the image border it is truncated by the frame: the visible
# silhouette is only PART of the vehicle, so its width/height (-> depth) and its
# horizontal center (-> lateral) are both corrupted, and that artifact would enter
# the differentiated speed as a spike. When enabled, any frame whose bbox sits
# within EDGE_MARGIN_PX of any border gets its Kalman measurement noise multiplied
# by EDGE_NOISE_SCALE (a large factor -> the frame is nearly ignored, the smoother
# coasts on its model through it). Combines multiplicatively with WIDE_ANGLE_*.
# OFF by default (sim/validated path unchanged); main.py: --edge-gate 1.
EDGE_GATE = False                  # master toggle (False -> no change to anything)
EDGE_MARGIN_PX = 3                 # a bbox within this many px of a border counts as touching
EDGE_NOISE_SCALE = 8.0             # noise-std multiplier on touching frames (8x std = 64x variance)

# ── Frame-edge DROP gate (clipped-bbox REMOVAL from the SPEED path) ────────────
# Unlike EDGE_GATE above (which only DOWN-WEIGHTS a touching frame), this IGNORES
# the frame for speed entirely: a gated frame is treated as a gap, so the
# world-position run is cut there and the collapsing tail never reaches the
# differentiator. The bbox is STILL stored on the vehicle and STILL used by the
# distance path (estimateDistance iterates the raw bbox dict independently) -- the
# drop is speed-only. A bbox within EDGE_DROP_MARGIN_PX of any border counts as
# touching. OFF by default; main.py: --edge-drop 1.
#   NOTE: on the undistorted Pixel-8 clip the usable image ends ~117 px short of
#   the right border (the undistort warp leaves an invalid band), so boxes clip at
#   x~1803, NOT 1919. With --undistort, the FrameUndistorter now reports the valid
#   pixel rectangle and the drop test uses THAT inner border (not the raw frame
#   edge), so boxes touching the warped invalid band are dropped automatically --
#   no manual margin bump needed. Without --undistort the test uses the frame edge.
EDGE_DROP = True                   # ON by default: drop frames whose bbox touches the (valid) edge
EDGE_DROP_MARGIN_PX = 8            # bbox within this many px of a border -> dropped for speed

# ── Far-distance DROP gate (too-distant bbox REMOVAL from the SPEED path) ──────
# Height-based depth error grows with range: a far car is only a few px tall, so a
# 1 px height error becomes a large depth error and differentiates into a large
# speed error. Beyond MAX_SPEED_DISTANCE_M the per-frame depth is too noisy to
# trust, so those frames are DROPPED from the speed runs (treated as a gap; the
# bbox and the distance export are untouched -- speed-only). A vehicle that is
# always farther than the cutoff therefore gets no speed at all. The threshold is
# on the estimator's DEPTH (forward distance) -- the same quantity stored in
# dist_per_frame. ON by default; main.py: --distance-drop 0 to disable, or
# --max-speed-distance <m> to change the cutoff.
DISTANCE_DROP = True               # master toggle (False -> no distance-based drop)
MAX_SPEED_DISTANCE_M = 90.0        # drop frames whose estimated depth >= this (m)

# ── Shrink-rate DROP gate (collapsing-bbox REMOVAL from the SPEED path) ────────
# A bbox that COLLAPSES fast is a degrading measurement: as a vehicle exits (or is
# occluded) its silhouette shrinks asymmetrically, so its height (-> depth) jumps
# and the differentiated speed spikes -- even while the SMOOTHED distance still
# looks fine (smoothing hides the transient that differentiation amplifies). This
# gate IGNORES (for speed only, as a gap) any frame whose width OR height has
# dropped by more than SHRINK_DROP_RATIO relative to the last NON-dropped frame --
# so it catches the collapse wherever it happens, with no dependence on the frame
# border. Genuine recession (a car getting smoothly smaller) shrinks only
# ~1-2%/frame and is NOT gated. OFF by default; main.py: --shrink-drop 1.
SHRINK_DROP = False                # master toggle (False -> no change to anything)
SHRINK_DROP_RATIO = 0.12           # fractional width/height collapse vs last good frame that triggers

# ── Android heading integration ───────────────────────────────────────────────
# Sign applied when integrating the phone's vertical-axis yaw RATE into a world
# heading (ego_yaw.ego_heading_from_android). Like LAT_SIGN this is handedness /
# mount dependent and MUST be re-validated on real footage: if a turn makes the
# reconstructed target speed WORSE, flip this. +1 is the neutral default (there is
# no CARLA equivalent to inherit). Only affects the Android path; sim is unchanged.
HEADING_SIGN = 1

# ── Smoother selection ────────────────────────────────────────────────────────
SMOOTHER_KALMAN = "kalman"       # constant-acceleration Kalman + RTS (emits std)
SMOOTHER_SAVGOL = "savgol"       # Savitzky-Golay reference (std = 0)
SMOOTHER_THEILSEN = "theilsen"   # robust Theil-Sen slope (std = 0)
DEFAULT_SMOOTHER = SMOOTHER_KALMAN

# ── Track filtering (issue #4) ────────────────────────────────────────────────
# Vehicles detected in fewer than this many seconds' worth of frames are skipped
# for speed estimation -- too little data for a useful estimate. This is the
# coarse whole-track gate; it is intentionally SEPARATE from the per-contiguous-
# run length the smoother's window requires (max(SMOOTH_WINDOW, DERIV_WINDOW)).
MIN_TRACK_SECONDS = 1.0

# ── Relevance rejection (note #1) ─────────────────────────────────────────────
# A vehicle is REJECTED from speed estimation entirely (no speed stored, so no
# speed plot and no overspeed flag) when it is probably NOT on our road. Two
# criteria, each judged over the vehicle's tracked life and OR'd:
#   * |cross-track world offset| > REJECT_LATERAL_M (m) -- a large, persistent
#     lateral offset means another road/lane far from ego.
#   * its own motion points TOWARD the ego (oncoming) -- the other carriageway.
# A criterion rejects only if it holds for at least REJECT_LIFE_FRACTION of the
# vehicle's life (so a momentary wide bearing or a noisy frame can't reject a
# real target). Reconstruction is the SAME world geometry the speed path uses.
#   WARNING: this is a HARD filter -- it can backfire on far on-road traffic and
#   on curves. Raise the thresholds (or revert to advisory tagging) if it drops
#   vehicles you care about.
# Master toggles for the two criteria (independent). Set either to False (or pass
# --reject-lateral 0 / --reject-direction 0) to disable that criterion alone; set
# BOTH to False to switch relevance rejection off entirely (no vehicle is dropped).
REJECT_BIG_LATERAL = False        # enable the persistent-lateral-offset criterion
REJECT_DIRECTION = True          # enable the oncoming (motion-toward-ego) criterion
REJECT_LATERAL_M = 7.0           # lateral threshold (m)
REJECT_LIFE_FRACTION = 0.5       # fraction of tracked life a criterion must hold to reject

# ── Occlusion down-weighting (note #2) ────────────────────────────────────────
# A vehicle partially hidden behind a NEARER vehicle (one whose bbox bottom edge
# sits lower in the image) has a truncated silhouette, so its height/width (->
# depth) and center (-> lateral) are corrupted on those frames. When OCCLUSION_GATE
# is on, each frame's Kalman measurement noise is inflated by 1/(1-occ_fraction),
# capped at OCC_MAX_NOISE_SCALE, so the smoother LEANS ON ITS MODEL through the
# occluded stretch instead of chasing the bad box (it is down-weighted, NOT
# dropped -- so a mid-track occlusion never cuts the run / loses the track). An
# occlusion that hides less than OCC_MIN_FRACTION of the box is ignored. SPEED
# only -- the distance path is untouched. See speed_estimation/occlusion.py.
OCCLUSION_GATE = False            # master toggle (False -> no change to anything)
OCC_MIN_FRACTION = 0.15          # ignore overlaps hiding less than this fraction of a box
OCC_MAX_NOISE_SCALE = 10.0       # cap on the noise-std multiplier (occ ~0.9 hits the cap)

# ── Aspect-ratio gate (abrupt width/height-ratio distortion) ──────────────────
# A bbox whose width/height RATIO changes abruptly is usually corrupted: clipped
# at the frame border (entering/exiting), collapsed where a nearer object cuts its
# width/height, or detector flicker. When ASPECT_GATE is on, each frame's Kalman
# measurement noise is inflated when its aspect ratio is a LOCAL outlier -- it
# departs from a sliding-window MEDIAN of the ratio by more than ASPECT_N_SIGMAS
# robust-sigmas. Using the local MEDIAN as the reference is the whole point: a
# SMOOTH ratio change (a vehicle TURNING, its silhouette opening over many frames)
# moves WITH the median and is NOT penalised -- only an abrupt jump is. Down-weight,
# not drop. SPEED only (the distance path is untouched). Complements OCCLUSION_GATE
# (tracked occluders) and the edge gates (border clipping) by also catching
# non-tracked occluders and detector errors, and SUSTAINED distortion the
# position-level Hampel would miss.
#   NOTE: this is the most false-positive-prone gate (a sharp turn can momentarily
#   trip it). Validate on footage with turns; disable with --aspect-gate 0.
ASPECT_GATE = True               # master toggle (False -> no change to anything)
ASPECT_HALF_WINDOW = 9           # frames each side for the local ratio baseline (~0.3s @30fps)
ASPECT_N_SIGMAS = 3.0            # robust-sigma departure from the local median before down-weighting
ASPECT_MAX_NOISE_SCALE = 8.0     # cap on the noise-std multiplier
# Floor on the robust sigma, as a fraction of the local median ratio. Without it a
# STEADY box (ratio ~constant -> MAD=0) could never flag a clip; with it, a steady
# box uses a simple relative threshold (flag a jump > ASPECT_N_SIGMAS * this * ratio,
# i.e. ~15% by default), while a TURNING box keeps the larger MAD-based sigma (so its
# smooth ramp still isn't flagged). Raise it to be LESS sensitive on steady boxes.
ASPECT_SIGMA_FLOOR_FRAC = 0.05
