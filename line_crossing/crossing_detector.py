"""
Solid-line crossing detection.

Two APIs:

* check_crossing(...): post-hoc analysis over a fully-tracked Vehicle. Used by
  the legacy World pipeline once the whole video has been ingested.

* CrossingMonitor: per-frame stateful tracker. Call .update(vehicle_id, bbox,
  lanes) every frame for every active vehicle; query .is_violator(vehicle_id)
  for live visualization. Designed for the real-time main.py loop.

Both honor the convention that lane labels prefixed with "solid" are the only
ones that can trigger a violation; "dashed_" lines are ignored.
"""

import numpy as np

from Objects.Vehicle import Vehicle


def bbox_bottom_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """The point where the vehicle's tyres meet the road: horizontal center of the
    box at its bottom edge -> (x_center, y_max)."""
    x1, _y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))


def point_in_masks(point: tuple[int, int], masks) -> bool:
    """True if `point` lies inside (or on the border of) any polygon mask. `masks`
    is an iterable of cv2 contour arrays (int32 Nx2) in the SAME coordinate space
    as `point` (full-resolution frame pixels)."""
    import cv2  # local import: crossing_detector stays import-cheap

    px, py = float(point[0]), float(point[1])
    for poly in masks:
        if poly is not None and len(poly) >= 3 and cv2.pointPolygonTest(poly, (px, py), False) >= 0:
            return True
    return False


def check_crossing(vehicle: Vehicle, detected_lines: dict[int, dict[str, tuple]]) -> int | None:
    """
    Checks if a vehicle ever crossed a solid line during its tracked lifetime.
    Returns the frame_id of the first detected crossing, or None if no crossing occurred.
    """
    frame_ids = sorted(vehicle.bounding_box.keys())
    prev_point: tuple[int, int] | None = None

    for frame_id in frame_ids:
        bbox = vehicle.bounding_box.get(frame_id)
        if bbox is None or bbox == (0, 0, 0, 0):
            prev_point = None
            continue

        curr_point = vehicle.bottomCenterInFrame(frame_id)

        if prev_point is not None:
            for label, coords in detected_lines.get(frame_id, {}).items():
                if not label.startswith("solid"):
                    continue
                lx1, ly1, lx2, ly2 = coords
                prev_side = _side_of_line(lx1, ly1, lx2, ly2, *prev_point)
                curr_side = _side_of_line(lx1, ly1, lx2, ly2, *curr_point)
                if prev_side * curr_side < 0:
                    return frame_id

        prev_point = curr_point

    return None


class CrossingMonitor:
    """
    Per-frame stateful crossing detector.

    Instantaneous test: a solid lane line "intersects" a vehicle bbox if the
    four corners of the bbox don't all lie on the same side of the line.
    Equivalent to: the (infinite extension of the) line passes through the box.
    The lane lines span the full ROI height (600..1080), so within the vehicle
    activity area the infinite-line approximation matches the segment.

    A vehicle that is ever caught intersecting a solid line is marked as a
    violator and stays marked for the rest of its tracked lifetime, so the
    visual warning persists across subsequent frames.
    """

    def __init__(self) -> None:
        self._violators: set[int] = set()

    def update(
        self,
        vehicle_id: int,
        bbox: tuple[int, int, int, int],
        lanes: dict[str, tuple[int, int, int, int]],
    ) -> bool:
        """
        Returns True iff this update is the *first* time we caught vehicle_id
        intersecting a solid line — useful for one-shot logging.
        """
        newly_violating = False
        for label, coords in lanes.items():
            if not label.startswith("solid"):
                continue
            if _bbox_intersects_line(bbox, coords):
                if vehicle_id not in self._violators:
                    newly_violating = True
                self._violators.add(vehicle_id)
        return newly_violating

    def update_from_masks(
        self,
        vehicle_id: int,
        bbox: tuple[int, int, int, int],
        solid_masks,
    ) -> bool:
        """
        Mask-based crossing test (Phase 3): flag `vehicle_id` if its bottom-center
        anchor (where the tyres meet the road) lies inside any SOLID lane-type
        segmentation mask. `solid_masks` is an iterable of polygon contours
        (int32 Nx2, full-res coords), e.g. from `DLLaneDetector.solid_lane_masks()`.

        Returns True iff this is the FIRST frame `vehicle_id` is flagged. The
        violator state persists for the vehicle's lifetime, exactly like update().
        """
        anchor = bbox_bottom_center(bbox)
        newly_violating = False
        if point_in_masks(anchor, solid_masks):
            if vehicle_id not in self._violators:
                newly_violating = True
            self._violators.add(vehicle_id)
        return newly_violating

    def update_from_mask_image(
        self,
        vehicle_id: int,
        bbox: tuple[int, int, int, int],
        solid_mask: np.ndarray | None,
    ) -> bool:
        """
        Raster variant of the mask-based crossing test (Phase 3). `solid_mask` is a
        STABILIZED binary image (HxW, nonzero = solid lane pixel), e.g. the output
        of SolidMaskStabilizer.update(). Flags `vehicle_id` if its bottom-center
        anchor (where the tyres meet the road) falls on a nonzero pixel - an O(1)
        lookup, no polygon test.

        Latches exactly like update()/update_from_masks(): once flagged, the vehicle
        stays a violator for its tracked lifetime. Returns True iff this is the FIRST
        frame it is flagged.
        """
        newly_violating = False
        if solid_mask is not None and solid_mask.size:
            ax, ay = bbox_bottom_center(bbox)
            h, w = solid_mask.shape[:2]
            if 0 <= ay < h and 0 <= ax < w and solid_mask[ay, ax] > 0:
                if vehicle_id not in self._violators:
                    newly_violating = True
                self._violators.add(vehicle_id)
        return newly_violating

    def is_violator(self, vehicle_id: int) -> bool:
        return vehicle_id in self._violators


def _bbox_intersects_line(
    bbox: tuple[int, int, int, int],
    line: tuple[int, int, int, int],
) -> bool:
    """
    True iff the line passes through the axis-aligned bbox — i.e. the four
    corners of the bbox straddle the line (some on each side).
    """
    bx1, by1, bx2, by2 = bbox
    lx1, ly1, lx2, ly2 = line
    sides = [
        _side_of_line(lx1, ly1, lx2, ly2, bx1, by1),
        _side_of_line(lx1, ly1, lx2, ly2, bx2, by1),
        _side_of_line(lx1, ly1, lx2, ly2, bx2, by2),
        _side_of_line(lx1, ly1, lx2, ly2, bx1, by2),
    ]
    return any(s > 0 for s in sides) and any(s < 0 for s in sides)


def _side_of_line(x1: int, y1: int, x2: int, y2: int, px: int, py: int) -> float:
    """
    Signed cross product of vector (x1,y1)→(x2,y2) and vector (x1,y1)→(px,py).
    Positive and negative values indicate opposite sides of the line.
    """
    return float((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))
