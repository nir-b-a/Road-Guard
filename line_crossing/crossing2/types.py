"""Data contracts for the Phase 4 crossing detector.

The module is deliberately OFFLINE and batch: the caller runs its own detection /
tracking / lane loop over the whole clip, collects one `FrameObservation` per
frame, and hands the list over once. That buys centered (non-causal) smoothing,
two-pass lane association and bidirectional event confirmation - none of which a
per-frame monitor can do.

Nothing here changes `lane_types.Lane`; a `FrameObservation` just carries the list
of `Lane` objects the detector already returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BBox = tuple[int, int, int, int]


@dataclass
class VehicleBox:
    """One tracked vehicle in one frame, in full-resolution image coords."""

    track_id: int
    bbox: BBox
    cls_name: str | None = None   # "car"/"truck"/... only used for the width fallback


@dataclass
class FrameObservation:
    """Everything the crossing detector needs from a single frame."""

    frame_id: int
    lanes: list                    # list[lane_types.Lane]
    vehicles: list[VehicleBox]


@dataclass
class LineTrack:
    """One physical road marking, tracked across the clip.

    `X` is the whole-clip geometry matrix: `X[i, r]` is the marking's x coordinate
    in frame index `i` at row `r * row_grid_step`, NaN where the marking does not
    exist or was not observed. This is the object that replaces "a fresh polyline
    every frame with no identity and a fresh type vote".
    """

    track_id: int
    X: np.ndarray                  # (N, R) float32
    lane_type: str = "unknown"
    votes: dict[str, float] = field(default_factory=dict)
    suppressed: bool = False       # the weaker half of a double line
    is_double: bool = False
    is_gore: bool = False
    n_obs: int = 0

    @property
    def is_solid(self) -> bool:
        """Same convention as `lane_types.Lane.is_solid`, but decided ONCE for the
        track's whole life instead of re-voted every frame."""
        return self.lane_type.startswith("solid")

    def active(self, i: int) -> bool:
        return bool(np.isfinite(self.X[i]).any())


@dataclass
class VehicleAnchors:
    """Per-vehicle ground-contact series (Stage 1), on the vehicle's own frames."""

    vehicle_id: int
    idx: np.ndarray                # (T,) int   - frame indices into the observation list
    frame_ids: np.ndarray          # (T,) int   - the caller's frame ids
    y_c: np.ndarray                # (T,) float - contact row (smoothed)
    x_c: np.ndarray                # (T,) float - bbox centre x  (smoothed)
    x_left: np.ndarray             # (T,) float - corrected footprint, left
    x_right: np.ndarray            # (T,) float - corrected footprint, right
    w_box: np.ndarray              # (T,) float - raw bbox width (width fallback)
    aspect: np.ndarray             # (T,) float - raw bbox w/h; high = we see the flank
    near_side: np.ndarray          # (T,) int8  - +1 near side is the RIGHT edge, -1 LEFT
    valid: np.ndarray              # (T,) bool  - False where y2 is not a ground contact
    cls_name: str | None = None
    boxes: dict[int, BBox] = field(default_factory=dict)   # frame_id -> raw bbox (overlay)


@dataclass
class OffsetSeries:
    """Stage 2 output: the signed lateral offset of one vehicle from one line.

    `u_c` is the vehicle centre's offset in LANE WIDTHS, signed by the line itself
    (positive = vehicle is to the RIGHT of the marking). `u_l`/`u_r` are the same
    for the corrected footprint edges, so the line is under the vehicle body
    exactly when `u_l < 0 < u_r`.
    """

    vehicle_id: int
    track_id: int
    idx: np.ndarray                # (T,) int
    frame_ids: np.ndarray          # (T,) int
    u_c: np.ndarray                # (T,) float, NaN where unusable
    u_l: np.ndarray
    u_r: np.ndarray
    extrapolated: np.ndarray       # (T,) bool - line geometry was extended below its extent
    aspect: np.ndarray | None = None   # (T,) float - raw bbox w/h, for the flank gate

    @property
    def n_valid(self) -> int:
        return int(np.isfinite(self.u_c).sum())


@dataclass
class CrossingEvent:
    """A confirmed event. `kind` is "crossing" (the vehicle changed sides and
    stayed) or "strafe" (it put part of its body over the line and came back)."""

    vehicle_id: int
    track_id: int
    kind: str                      # "crossing" | "strafe"
    lane_type: str
    start_frame: int
    end_frame: int
    key_frame: int
    direction: str                 # "to_left" | "to_right" | ""
    peak_overlap: float            # max fraction of the footprint across the line
    confidence: float
    is_gore: bool = False
    is_double: bool = False
    extrapolated_frac: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class CrossingResult:
    """Everything `detect_crossings` produces: the events plus the intermediate
    objects, which the overlay needs in order to draw WHY something fired."""

    events: list[CrossingEvent]
    tracks: list[LineTrack]
    anchors: dict[int, VehicleAnchors]
    offsets: list[OffsetSeries]
    lane_width: np.ndarray         # (N, R) local lane width in px, NaN where unknown
    vanishing: np.ndarray          # (N, 2) per-frame (x_vp, y_vp), NaN where unknown
    frame_ids: np.ndarray          # (N,) the caller's frame ids
    row_grid: np.ndarray           # (R,) the y coordinate of each row index
    probes: dict | None = None     # v3 only: (vehicle, frame) -> what was tested, and
                                   # whether it hit. Lets the overlay draw the actual
                                   # probe geometry instead of just the verdict.

    def violator_ids(self, kinds: tuple[str, ...] = ("crossing",)) -> set[int]:
        """Vehicle ids with at least one event of the given kinds - directly
        comparable with the legacy `CrossingMonitor._violators`."""
        return {e.vehicle_id for e in self.events if e.kind in kinds}

    def violator_frames(self, kinds: tuple[str, ...] = ("crossing", "strafe")) -> dict[int, set[int]]:
        """vehicle id -> the frame ids its events span. The legacy detector cannot
        produce this: it latches a vehicle forever from a single frame."""
        out: dict[int, set[int]] = {}
        for e in self.events:
            if e.kind in kinds:
                out.setdefault(e.vehicle_id, set()).update(range(e.start_frame, e.end_frame + 1))
        return out
