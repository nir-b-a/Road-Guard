"""
Classical-detector adapter (Phase 1).

Wraps the legacy classical `LaneDetector` so it satisfies the `LaneSource`
protocol, converting its native dict output UP into the new `list[Lane]` format.
This makes the classical and DL detectors drop-in swappable for A/B ablation.

IMPORTANT — the real legacy schema:
    LaneDetector.detect_lanes() returns geometry and type in SEPARATE keys:
        {
          "left_line":  (x_bottom, y_bottom, x_top, y_top),
          "left_type":  "solid" | "dashed" | "unknown",
          "right_line": (...),
          "right_type": ...,
        }
    (This is NOT the "solid_left" shape — that is what compat.to_legacy_dict
    produces DOWNSTREAM for crossing_detector. Don't confuse the two.)

A missing side simply omits both of its keys.
"""

from __future__ import annotations

import numpy as np

from .lane_types import Lane, LaneType
from .line_detector import LaneDetector

# side label -> ego-relative position (the classical detector only ever
# resolves the two ego-boundary lanes).
_SIDE_POSITION = {"left": -1, "right": 1}
_VALID_TYPES: set[str] = {"solid", "dashed", "unknown"}


def lanes_from_classical_output(
    legacy: dict[str, object],
) -> list[Lane]:
    """
    Convert one `LaneDetector.detect_lanes()` result dict into `list[Lane]`.

    Pure (no model state) so it can be unit-tested directly.
    """
    lanes: list[Lane] = []
    for side, position in _SIDE_POSITION.items():
        line = legacy.get(f"{side}_line")
        if line is None:
            continue
        x_b, y_b, x_t, y_t = (int(v) for v in line)  # type: ignore[misc]

        raw_type = legacy.get(f"{side}_type", "unknown")
        lane_type: LaneType = raw_type if raw_type in _VALID_TYPES else "unknown"  # type: ignore[assignment]

        # Honor the Lane convention: bottom (largest y) first.
        points = [(x_b, y_b), (x_t, y_t)]
        points.sort(key=lambda p: p[1], reverse=True)

        lanes.append(Lane(points=points, lane_type=lane_type, position=position))
    return lanes


class ClassicalLaneSource:
    """
    `LaneSource`-compatible wrapper around the classical `LaneDetector`.

    Holds one stateful `LaneDetector` for the lifetime of a video stream (its
    temporal smoothing / VP tracking accumulate across frames), and exposes the
    unified `detect(frame) -> list[Lane]` interface.
    """

    def __init__(self, detector: LaneDetector | None = None, **detector_kwargs) -> None:
        self._detector = detector if detector is not None else LaneDetector(**detector_kwargs)

    def detect(self, frame: np.ndarray) -> list[Lane]:
        return lanes_from_classical_output(self._detector.detect_lanes(frame))
