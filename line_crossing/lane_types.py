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
* `lane_type` is "unknown" until the Phase 3 type head exists; the geometric
  baseline (Phase 2) emits geometry only.
* `position` is the ego-relative lane index: -1 = first lane boundary to the
  left of the car, +1 = first to the right, -2/+2 further out, None if unknown.
  It is assigned GEOMETRICALLY in post-processing, not predicted by the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

# Semantic lane-line type. Kept deliberately small; map VIL-100's 10 categories
# down to this taxonomy in the Phase 3 dataloader.
LaneType = Literal["solid", "dashed", "unknown"]

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
        return self.lane_type == "solid"

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
