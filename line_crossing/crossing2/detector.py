"""Top-level entry point for the Phase 4 solid-line crossing detector.

    from line_crossing.crossing2 import detect_crossings, FrameObservation, VehicleBox

    obs = []
    for frame_id, frame in enumerate(video):
        lanes = lane_detector.detect(frame)                    # list[lane_types.Lane]
        boxes = [VehicleBox(tid, bbox) for tid, bbox in tracker(frame)]
        obs.append(FrameObservation(frame_id, lanes, boxes))

    result = detect_crossings(obs, W, H, fps)
    for e in result.events:
        ...

OFFLINE and batch by design: the whole clip is ingested before anything is decided,
which is what makes centered smoothing, two-pass lane association and bidirectional
event confirmation possible. Nothing here mutates the caller's `Lane` objects, and
the module never imports the lane detector - it only consumes the `lane_types`
contract, so the classical adapter, CLRerNet and a plain segmentation model are all
valid inputs (use `contour_to_lane` for the last of those).

A/B against the legacy `CrossingMonitor`: `result.violator_ids()` is directly
comparable with `CrossingMonitor._violators`, and `result.violator_frames()` gives
the per-frame spans the legacy detector cannot produce because it latches a vehicle
for its whole lifetime from a single frame. See tools/test_crossing_ab.py.
"""
from __future__ import annotations

import numpy as np

from .anchors import build_anchors
from .config import CrossingConfig
from .events import detect_events
from .lane_tracks import build_line_tracks
from .offsets import build_offsets
from .types import CrossingResult, FrameObservation


def detect_crossings(observations: list[FrameObservation], frame_width: int,
                     frame_height: int, fps: float,
                     config: CrossingConfig | None = None,
                     stats: dict | None = None) -> CrossingResult:
    """Stages 0-3 over a whole clip. See the module docstrings for each stage.

    Pass a dict as `stats` to get per-stage rejection counts back - essential when the
    answer is "no events", since that can mean either "nothing happened" or "a gate
    discarded everything"."""
    cfg = config or CrossingConfig()
    n = len(observations)
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return CrossingResult([], [], {}, [], empty, np.zeros((0, 2), np.float32),
                              np.zeros(0, np.int64), np.zeros(0))

    tracks, lane_width, van, ys = build_line_tracks(
        observations, frame_width, frame_height, fps, cfg)
    anchors = build_anchors(observations, lane_width, ys, van,
                            frame_width, frame_height, cfg, stats)
    offsets = build_offsets(anchors, tracks, lane_width, ys, fps, cfg, stats)
    events = detect_events(offsets, tracks, fps, cfg)
    if stats is not None:
        stats["stage0"] = {
            "tracks": len(tracks),
            "solid": sum(1 for t in tracks if t.is_solid and not t.suppressed),
            "suppressed": sum(1 for t in tracks if t.suppressed),
            "lane_width_known_frac": round(float(np.isfinite(lane_width).mean()), 3),
        }

    return CrossingResult(
        events=events, tracks=tracks, anchors=anchors, offsets=offsets,
        lane_width=lane_width, vanishing=van,
        frame_ids=np.array([ob.frame_id for ob in observations], dtype=np.int64),
        row_grid=ys,
    )


def to_violation_events(events, only_kinds: tuple[str, ...] = ("crossing",)):
    """Convert `CrossingEvent`s into the repo-wide `ViolationEvent` records the
    evidence/export stage consumes. Imported lazily so this package stays usable
    (and testable) without the violations chain installed."""
    from violations.event import ViolationEvent, ViolationType

    out = []
    for e in events:
        if e.kind not in only_kinds:
            continue
        out.append(ViolationEvent(
            vehicle_id=e.vehicle_id,
            violation_type=ViolationType.SOLID_LINE_CROSSING,
            key_frame=e.key_frame,
            confidence=round(e.confidence, 4),
            details={"start_frame": e.start_frame, "end_frame": e.end_frame,
                     "n_frames_over": e.end_frame - e.start_frame + 1,
                     "kind": e.kind, "lane_type": e.lane_type,
                     "direction": e.direction, "line_track": e.track_id,
                     "peak_overlap": round(e.peak_overlap, 3),
                     "is_gore": e.is_gore, "is_double": e.is_double,
                     **e.details},
        ))
    return out


def format_events(events) -> str:
    """One line per event, for the console A/B table."""
    if not events:
        return "  (none)"
    rows = ["  {:>4}  {:<9} {:<13} {:>6}-{:<6} key={:<6} {:<9} ov={:<5} conf={:<5} {}".format(
        e.vehicle_id, e.kind, e.lane_type, e.start_frame, e.end_frame, e.key_frame,
        e.direction or "-", f"{e.peak_overlap:.2f}", f"{e.confidence:.2f}",
        ("GORE " if e.is_gore else "") + ("DOUBLE" if e.is_double else "")) for e in events]
    return "\n".join(rows)
