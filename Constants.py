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
DEFAULT_CAMERA_HEIGHT_M = 1.2    # dashcam mount height (also the CARLA default)

# ── Detection / tracking ──────────────────────────────────────────────────────
YOLO_VERSION = "yolo11x.pt"
CONFIDENCE_LVL = 0.5             # YOLO minimum detection confidence
YOLO_IMGSZ = 1984                # YOLO inference image size (px); 1984 w/o half
YOLO_TRACKER = "botsort.yaml"    # ultralytics tracker config

# ── Bounding-box gap interpolation ────────────────────────────────────────────
MAX_GAP_SECONDS = 0.5            # linearly interpolate bbox gaps up to this long

# ── Camera intrinsics (CARLA dashcam defaults) ────────────────────────────────
FOV_HORIZONTAL_DEG = 90.0        # horizontal field of view (square pixels)

# ── Speed-estimation smoothing (Savitzky-Golay) ───────────────────────────────
SMOOTH_WINDOW = 101               # pre-smoothing savgol window (odd; ~1.7s @30fps)
DERIV_WINDOW = 61                # differentiation savgol window (odd; ~0.7s @30fps)
POLYORDER = 2                    # savgol polynomial order

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
