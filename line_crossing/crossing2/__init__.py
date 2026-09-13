"""crossing2 - Phase 4 solid-line crossing detection (offline, temporal).

Replacement for `line_crossing.crossing_detector`, in a separate package so both
can run side by side on the same clip until the new one is demonstrably better.
The legacy module is untouched.

What changed, and why
---------------------
The legacy detector asks "is this vehicle's bottom-centre pixel ON a solid-lane
mask RIGHT NOW", and latches the vehicle as a violator forever the first time the
answer is yes. That test fails in both directions:

  * it MISSES, because the paint under a vehicle is occluded by that vehicle and
    therefore is not segmented - the mask is blank exactly where the event happens;
  * it FALSE-POSITIVES, because one bad frame (a truncated box, a flickering type
    vote, a far-field dashed line that foreshortens into apparent continuity) marks
    a vehicle permanently.

crossing2 measures instead the signed lateral offset of the vehicle's road contact
from each tracked solid marking, normalized by the local lane width, and detects a
sustained CHANGE OF SIDES over time. Geometry comes from the lane polylines (which
survive occlusion), the lane type is decided once per tracked line rather than per
frame, and every event has a start, an end, a direction and a confidence.

Stages
------
  0. lane_tracks.py - per-frame lanes -> tracked, typed, smoothed line objects
  1. anchors.py     - bbox -> ground contact + corrected footprint, with rejection
  2. offsets.py     - the signed offset u, in lane widths
  3. events.py      - traverse ("crossing") and straddle ("strafe") channels

edge_contact.py is a THIRD decision rule (v3) sharing stages 0-1: a per-frame bbox
half-edge contact test. yellow_crossing.py wires that rule into main.py for YELLOW
solid lines only - the white lines and the painted gore areas stay with the original
detector - and emits the same ViolationEvent, so nothing downstream changes.
"""
from .config import CrossingConfig
from .detector import detect_crossings, format_events, to_violation_events
from .edge_contact import EdgeContactConfig, detect_edge_contacts
from .lane_tracks import contour_to_lane
from .types import (
    BBox,
    CrossingEvent,
    CrossingResult,
    FrameObservation,
    LineTrack,
    OffsetSeries,
    VehicleAnchors,
    VehicleBox,
)
from .yellow_crossing import drop_duplicate_crossings, evaluate_yellow_crossing

__all__ = [
    "CrossingConfig",
    "EdgeContactConfig",
    "detect_crossings",
    "detect_edge_contacts",
    "drop_duplicate_crossings",
    "evaluate_yellow_crossing",
    "format_events",
    "to_violation_events",
    "contour_to_lane",
    "BBox",
    "CrossingEvent",
    "CrossingResult",
    "FrameObservation",
    "LineTrack",
    "OffsetSeries",
    "VehicleAnchors",
    "VehicleBox",
]
