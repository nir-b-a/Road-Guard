"""
Shared lane data contracts for the deep-learning lane pipeline.

These types are the seam between *any* lane detector (classical or DL) and the
rest of Road Guard. Both the legacy classical `LaneDetector` and the new
`DLLaneDetector` are expected to be wrapped behind the `LaneSource` protocol so
they are drop-in swappable for A/B ablation.

Conventions
-----------
* A `Lane` is a polyline in FULL-RESOLUTION image coordinates (e.g. 1920x1080),
  *not* the model's input resolution. The wrapper is responsible for inverting
  any crop/resize before constructing a `Lane`.
* Points are ordered BOTTOM -> TOP, i.e. from the largest y (nearest the car)
  to the smallest y (toward the vanishing point). `bottom`/`top` rely on this.
* `lane_type` is "unknown" until a type source assigns it. The classical
  detector emits the color-agnostic "solid"/"dashed"; the Phase 3 DL head emits
  the color-aware "solid_white"/"solid_yellow"/"dashed". The geometric CLRerNet
  baseline leaves it "unknown" until `_classify_types` runs.
* `position` is the ego-relative lane index: -1 = first lane boundary to the
  left of the car, +1 = first to the right, -2/+2 further out, None if unknown.
  It is assigned GEOMETRICALLY in post-processing, not predicted by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

# Semantic lane-line type.
#   * Color-aware values come from the Phase 3 YOLOv8-seg head:
#       solid_white  <- solid_white_lane
#       solid_yellow <- yellow_solid_lane
#       dashed       <- dashed_lane (white & yellow unified)
#   * "solid" is the color-AGNOSTIC value the classical detector emits (it can't
#     tell white from yellow); kept so the classical ablation baseline stays
#     valid under this shared contract.
#   * Crossing logic treats EVERY "solid*" value as a non-crossable boundary
#     (see crossing_detector's startswith("solid") gate and Lane.is_solid).
LaneType = Literal["solid", "solid_white", "solid_yellow", "dashed", "unknown"]

Point = tuple[int, int]


@dataclass
class Lane:
    """A single detected lane line as a full-resolution polyline."""

    points: list[Point]
    lane_type: LaneType = "unknown"
    position: int | None = None
    score: float = 1.0

    @property
    def bottom(self) -> Point:
        """Nearest-to-car endpoint (largest y). Points are ordered bottom->top."""
        return self.points[0]

    @property
    def top(self) -> Point:
        """Farthest endpoint (smallest y)."""
        return self.points[-1]

    @property
    def is_solid(self) -> bool:
        """Any solid variant (color-agnostic "solid" or color-aware
        "solid_white"/"solid_yellow") is a non-crossable boundary."""
        return self.lane_type.startswith("solid")

    def x_at_y(self, y: int) -> float | None:
        """
        Interpolate the lane's x coordinate at image row `y`, or None if `y`
        falls outside the polyline's actual vertical extent.

        This is the primitive the Phase 4 polyline-aware crossing test will use
        instead of the legacy infinite-line assumption: a lane only "exists"
        between its bottom and top points.
        """
        if len(self.points) < 2:
            return None
        # Walk consecutive segments looking for the one that brackets y.
        for (x1, y1), (x2, y2) in zip(self.points, self.points[1:]):
            lo, hi = (y1, y2) if y1 <= y2 else (y2, y1)
            if lo <= y <= hi:
                if y2 == y1:
                    return float(x1)
                t = (y - y1) / (y2 - y1)
                return float(x1 + t * (x2 - x1))
        return None


@runtime_checkable
class LaneSource(Protocol):
    """
    The single method every lane detector exposes. Wrap the classical
    `LaneDetector` and the DL `DLLaneDetector` in this so the pipeline can
    switch backends with one line for A/B ablation.
    """

    def detect(self, frame: np.ndarray) -> list[Lane]:
        ...
