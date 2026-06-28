"""
Pure geometry primitives for the Stage-2 cascade.

Everything here is dependency-light (stdlib + optional numpy) and side-effect free, so it is
fully unit-testable without a GPU, a video, or a model. Coordinates are image pixels with the
usual convention: x grows right, y grows DOWN. A "bbox" is (x1, y1, x2, y2).

Lane geometry comes from the seg-model contours cached per frame: a contour is a list of
[x, y] vertices. We treat a contour both as a closed polygon (point-in-polygon tests) and as
a sequence of edges (segment-intersection tests).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# segment / point primitives
# --------------------------------------------------------------------------- #
def _orient(a: Point, b: Point, c: Point) -> float:
    """>0 if c is left of a->b, <0 if right, 0 if collinear (2x signed triangle area)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-9) -> bool:
    """True if collinear point p lies within the bounding box of segment a-b."""
    return (min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps)


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1-p2 intersects segment p3-p4 (proper or touching)."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    # collinear / touching cases
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


def point_to_segment_dist(p: Point, a: Point, b: Point) -> float:
    """Euclidean distance from point p to segment a-b."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon. `poly` is an ordered list of vertices (auto-closed)."""
    n = len(poly)
    if n < 3:
        return False
    x, y = p
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def point_side_of_line(p: Point, a: Point, b: Point) -> float:
    """Signed side of directed line a->b. >0 = left, <0 = right, 0 = on the line."""
    return _orient(a, b, p)


# --------------------------------------------------------------------------- #
# contour helpers
# --------------------------------------------------------------------------- #
def contour_edges(contour: Sequence[Point]) -> Iterable[Tuple[Point, Point]]:
    """Yield consecutive edges of a contour, closing it back to the first vertex."""
    n = len(contour)
    for i in range(n):
        yield contour[i], contour[(i + 1) % n]


def segment_hits_contour(p1: Point, p2: Point, contour: Sequence[Point],
                         margin: float = 0.0) -> bool:
    """
    True if segment p1-p2 intersects (or comes within `margin` px of) a lane contour.

    `margin` (wheel_vector_margin) is the pixel tolerance: an axle vector that passes just
    shy of the painted line still counts as a straddle, absorbing seg-mask thinning and
    sub-pixel projection error.
    """
    if len(contour) < 2:
        return False
    for a, b in contour_edges(contour):
        if segments_intersect(p1, p2, a, b):
            return True
        if margin > 0.0:
            # cheap proximity test: either endpoint of the edge near the vector, or vice-versa
            if (point_to_segment_dist(a, p1, p2) <= margin or
                    point_to_segment_dist(b, p1, p2) <= margin or
                    point_to_segment_dist(p1, a, b) <= margin or
                    point_to_segment_dist(p2, a, b) <= margin):
                return True
    return False


def point_near_contour(p: Point, contour: Sequence[Point], margin: float) -> bool:
    """True if point p is within `margin` px of any edge of the contour."""
    for a, b in contour_edges(contour):
        if point_to_segment_dist(p, a, b) <= margin:
            return True
    return False


# --------------------------------------------------------------------------- #
# Spec B: smart crop
# --------------------------------------------------------------------------- #
def smart_crop_box(bbox: BBox, expand_bottom_frac: float = 0.15,
                   frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> BBox:
    """
    Expand a vehicle bbox's BOTTOM edge downward by `expand_bottom_frac` of its height so the
    tire crop can never clip the wheels, then clamp to the frame.
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    y2e = y2 + expand_bottom_frac * h
    if frame_w is not None:
        x1 = max(0.0, min(x1, frame_w - 1))
        x2 = max(0.0, min(x2, frame_w - 1))
    if frame_h is not None:
        y2e = min(y2e, frame_h - 1)
    y1 = max(0.0, y1)
    return (x1, y1, x2, y2e)


# --------------------------------------------------------------------------- #
# Spec E: near-field horizon gate
# --------------------------------------------------------------------------- #
def passes_horizon_gate(bbox: BBox, y_horizon_cutoff: float) -> bool:
    """
    Keep only vehicles whose tire-contact point (bbox bottom) is BELOW (numerically greater
    than) the horizon cutoff line. Far vehicles sit high in the frame where 2D bboxes are
    geometrically unstable, so we ignore their line-crossing geometry entirely.
    """
    _, _, _, y2 = bbox
    return y2 >= y_horizon_cutoff


# --------------------------------------------------------------------------- #
# Spec C: tire selection + axle-vector straddle
# --------------------------------------------------------------------------- #
def tire_center(tire_box: BBox) -> Point:
    """Contact point of a tire = bottom-center of its box (where rubber meets road)."""
    x1, y1, x2, y2 = tire_box
    return ((x1 + x2) / 2.0, y2)


def rear_axle_pair(tire_boxes: Sequence[BBox]) -> Optional[Tuple[Point, Point]]:
    """
    Pick the rear axle: the two tires with the LOWEST contact points (largest y, nearest the
    camera on a vehicle ahead), returned as (left, right) by x. Needs >= 2 tires.
    """
    if len(tire_boxes) < 2:
        return None
    by_y = sorted(tire_boxes, key=lambda b: tire_center(b)[1], reverse=True)
    a, b = by_y[0], by_y[1]
    ca, cb = tire_center(a), tire_center(b)
    return (ca, cb) if ca[0] <= cb[0] else (cb, ca)


def axle_vector_straddle(tire_boxes: Sequence[BBox], contours: Sequence[Sequence[Point]],
                         margin: float = 0.0) -> bool:
    """
    Spec C.1 -- Axle-Vector Straddle. With >= 2 tires, build the T_L -> T_R segment between the
    rear axle's tire contact points and test whether it cuts (within `margin`) any lane contour
    (real OR ghost). The car straddling the paint means the invisible axle line crosses it even
    while the paint under the chassis is occluded.
    """
    pair = rear_axle_pair(tire_boxes)
    if pair is None:
        return False
    tl, tr = pair
    for contour in contours:
        if segment_hits_contour(tl, tr, contour, margin=margin):
            return True
    return False


# --------------------------------------------------------------------------- #
# Spec C: single-tire relative-position check
# --------------------------------------------------------------------------- #
def relative_position_violation(tire_boxes: Sequence[BBox],
                                lane_polyline: Sequence[Point],
                                solid_on_right: bool,
                                margin: float = 0.0) -> bool:
    """
    Spec C.2 -- Relative Position Check. When only ONE tire is visible, decide whether it has
    crossed a known solid line using which side of the line the tire sits on.

    `lane_polyline` is the (ordered, roughly vertical) set of points of the solid line.
    `solid_on_right` True  -> the solid line is the vehicle's right-hand boundary; a tire to the
                              RIGHT of it has crossed.
                     False -> mirror case (left-hand boundary).
    We test the tire against the nearest line edge and require it to be past the line by `margin`.
    """
    if not tire_boxes or len(lane_polyline) < 2:
        return False
    # one visible tire -> use the lowest (closest) one
    tb = max(tire_boxes, key=lambda b: tire_center(b)[1])
    p = tire_center(tb)
    # nearest edge of the polyline to the tire
    best = None
    for a, b in zip(lane_polyline[:-1], lane_polyline[1:]):
        d = point_to_segment_dist(p, a, b)
        if best is None or d < best[0]:
            best = (d, a, b)
    if best is None:
        return False
    _, a, b = best
    # orient the edge top->bottom so "left/right" is stable in image space
    if a[1] > b[1]:
        a, b = b, a
    side = point_side_of_line(p, a, b)  # >0 left, <0 right
    crossed = (side < -margin) if solid_on_right else (side > margin)
    return crossed


# --------------------------------------------------------------------------- #
# Spec D: license-plate ground-plane parallax projection
# --------------------------------------------------------------------------- #
def project_plate_to_ground(plate_center: Point, plate_parallax_offset_y: float) -> Point:
    """
    Spec D -- Parallax correction. A plate floats ~30-90 cm off the road, so its image point
    sits ABOVE the true ground contact. We have no real camera calibration for these YouTube
    dashcams, so this is an HONEST tunable heuristic (NOT a projection matrix): push the plate
    point straight down by `plate_parallax_offset_y` px to approximate where it meets the road,
    then test THAT against the lane boundary. The offset is one of the swept parameters.
    """
    return (plate_center[0], plate_center[1] + plate_parallax_offset_y)


def plate_crosses_boundary(plate_center: Point, lane_polyline: Sequence[Point],
                           solid_on_right: bool, plate_parallax_offset_y: float,
                           margin: float = 0.0) -> bool:
    """Project the plate to the ground plane, then run the relative-position side test."""
    ground = project_plate_to_ground(plate_center, plate_parallax_offset_y)
    if len(lane_polyline) < 2:
        return False
    best = None
    for a, b in zip(lane_polyline[:-1], lane_polyline[1:]):
        d = point_to_segment_dist(ground, a, b)
        if best is None or d < best[0]:
            best = (d, a, b)
    _, a, b = best
    if a[1] > b[1]:
        a, b = b, a
    side = point_side_of_line(ground, a, b)
    return (side < -margin) if solid_on_right else (side > margin)


# --------------------------------------------------------------------------- #
# orientation helper for the LPR fallback gate (Spec D)
# --------------------------------------------------------------------------- #
def is_parallel_ahead(bbox: BBox, aspect_tol: float = 0.25) -> bool:
    """
    Spec D gate -- a vehicle driving parallel directly ahead presents a roughly square rear
    (w ~= h). True when |w/h - 1| <= aspect_tol. (Weak proxy; the optimizer / caller should
    pair it with trajectory when available -- see handoff notes.)
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return False
    return abs((w / h) - 1.0) <= aspect_tol


# --------------------------------------------------------------------------- #
# Orientation steering (aspect-ratio pseudo-angle)
# --------------------------------------------------------------------------- #
def aspect_ratio(bbox: BBox) -> float:
    """Width / height of a bbox (0.0 if degenerate height)."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    return (x2 - x1) / h if h > 0 else 0.0


def orientation_pseudo_angle(bbox: BBox, rear_aspect: float = 1.2,
                             side_aspect: float = 2.6) -> float:
    """
    Estimate how side-on a vehicle is, as a PSEUDO-angle in [0, 90) degrees, from its 2D bbox
    aspect ratio. These are uncalibrated dashcams (no intrinsics / pose), so a true yaw is
    unrecoverable -- this is an HONEST heuristic, not photogrammetry (cf. project_plate_to_ground).

    Intuition: a vehicle seen rear-on presents ~ its rear width over its height
    (`rear_aspect`); as it yaws toward a side view its apparent width grows because we start to
    see its length, so w/h climbs toward `side_aspect`. We map that band linearly to 0..90 deg:

        w/h <= rear_aspect            -> 0 deg   (rear-on / parallel-ahead -> trust the PLATE)
        w/h >= side_aspect            -> 90 deg  (full side view          -> trust the TIRES)

    The caller compares this against `orientation_angle_threshold` to steer tire vs plate.
    `rear_aspect`/`side_aspect` are tunable so the pseudo-degrees can be re-calibrated later.
    """
    r = aspect_ratio(bbox)
    if side_aspect <= rear_aspect:
        return 0.0
    frac = (r - rear_aspect) / (side_aspect - rear_aspect)
    return max(0.0, min(1.0, frac)) * 90.0


# --------------------------------------------------------------------------- #
# Spec C (relaxed): same-vehicle axle gate + axle-midpoint line-band test
# --------------------------------------------------------------------------- #
def same_vehicle_axle_pair(tire_boxes: Sequence[BBox], vehicle_bbox: BBox,
                           max_sep_frac: float = 1.05) -> Optional[Tuple[Point, Point]]:
    """
    Like `rear_axle_pair`, but reject a pair whose horizontal separation exceeds
    `max_sep_frac` x vehicle width -- a guard against stitching two tires from DIFFERENT
    vehicles / axles into one bogus axle vector. Returns (left, right) by x, or None.
    """
    pair = rear_axle_pair(tire_boxes)
    if pair is None:
        return None
    (lx, _), (rx, _) = pair
    veh_w = vehicle_bbox[2] - vehicle_bbox[0]
    if veh_w > 0 and abs(rx - lx) > max_sep_frac * veh_w:
        return None
    return pair


def axle_midpoint_in_band(axle_pair: Tuple[Point, Point],
                          contours: Sequence[Sequence[Point]], band_px: float) -> bool:
    """
    Require the MIDPOINT of the axle vector to lie within `band_px` of a solid contour edge.
    A thin painted line is a narrow band, whereas a filled `traffic_island` polygon is a blob;
    insisting the axle midpoint sit ON the paint (not merely that the vector clips a far polygon
    edge) suppresses island over-firing. Returns True if the midpoint is inside the band.
    """
    tl, tr = axle_pair
    mid = ((tl[0] + tr[0]) / 2.0, (tl[1] + tr[1]) / 2.0)
    return any(point_near_contour(mid, c, band_px) for c in contours)


# --------------------------------------------------------------------------- #
# Spec D (extended): license-plate axle-proxy
# --------------------------------------------------------------------------- #
def plate_axle_proxy(plate_center: Point, plate_w: float, x_extension_pct: float,
                     plate_parallax_offset_y: float) -> Tuple[Point, Point]:
    """
    Build an axle-proxy SEGMENT from a license plate. A plate is a tight, narrow target compared
    to the vehicle's track width, so we widen its horizontal extent by `x_extension_pct` of the
    plate width on EACH side, then drop the whole segment to the road plane with the parallax
    offset. The result stands in for the rear axle when no tires are detected (rear-on / truck).

        half = plate_w/2 + x_extension_pct * plate_w
        y    = plate_center_y + plate_parallax_offset_y      (honest parallax heuristic)
        -> (left=(cx-half, y), right=(cx+half, y))
    """
    cx, cy = plate_center
    half = plate_w / 2.0 + x_extension_pct * plate_w
    y = cy + plate_parallax_offset_y
    return ((cx - half, y), (cx + half, y))


def plate_proxy_straddle(plate_center: Point, plate_w: float,
                         contours: Sequence[Sequence[Point]], x_extension_pct: float,
                         plate_parallax_offset_y: float, margin: float = 0.0) -> bool:
    """
    Spec D (axle-proxy form) -- widen the plate into a horizontal axle-proxy segment, project it
    to the ground, and test whether THAT segment straddles any solid contour (real or ghost),
    exactly like `axle_vector_straddle` does for real tires.
    """
    if plate_w <= 0:
        return False
    pl, pr = plate_axle_proxy(plate_center, plate_w, x_extension_pct, plate_parallax_offset_y)
    for contour in contours:
        if segment_hits_contour(pl, pr, contour, margin=margin):
            return True
    return False


# --------------------------------------------------------------------------- #
# NEW ARCH -- strict oncoming + bbox-bottom + corner-arm + curve gate
#
# These replace `relative_position_violation` (single_tire_SIDE), which owned 72% of FP because
# its single-point side test is perspective-fragile and its `solid_on_right` convention is
# meaningless for oncoming traffic (we view ACROSS the line). The primitives below are all
# SIDE-AGNOSTIC: they test penetration / intersection, never "which side", so they work for
# oncoming without a (broken) lane-side prior.
# --------------------------------------------------------------------------- #
def _segment_intersection_point(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """Intersection point of segments p1-p2 and p3-p4, or None if they don't cross."""
    r = (p2[0] - p1[0], p2[1] - p1[1])
    s = (p4[0] - p3[0], p4[1] - p3[1])
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None                       # parallel / degenerate
    qp = (p3[0] - p1[0], p3[1] - p1[1])
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return (p1[0] + t * r[0], p1[1] + t * r[1])
    return None


def tire_crosses_line(tire_box: BBox, contours: Sequence[Sequence[Point]],
                      penetration: float = 0.0) -> bool:
    """
    SUPER-TIGHT single-tire test (replaces single_tire_SIDE for oncoming): the lane line must
    pass THROUGH the tire's contact footprint -- not merely graze it. We use the tire's bottom
    edge p_L=(x1,y2) -> p_R=(x2,y2) (where rubber meets road) and require:

      * the contour to actually intersect that bottom edge (the line enters between the corners),
      * the crossing point to sit at least `penetration` px inside BOTH corners,

    so a bumper / shadow box that only kisses the paint at a corner is rejected ("crosses, not
    touches"). Side-agnostic: no solid_on_right prior, which is wrong for oncoming. `penetration`
    is the swept "super-tight" knob -- 0 = any clean cross, larger = deeper overshoot required.
    """
    x1, _y1, x2, y2 = tire_box
    pL, pR = (x1, y2), (x2, y2)
    for contour in contours:
        for a, b in contour_edges(contour):
            ip = _segment_intersection_point(pL, pR, a, b)
            if ip is None:
                continue
            dL = math.hypot(ip[0] - pL[0], ip[1] - pL[1])
            dR = math.hypot(ip[0] - pR[0], ip[1] - pR[1])
            if min(dL, dR) >= penetration:
                return True
    return False


def bbox_bottom_intersects_line(bbox: BBox, contours: Sequence[Sequence[Point]],
                                margin: float = 0.0) -> bool:
    """
    True if the BOTTOM EDGE of the Stage-1 vehicle bbox ( (x1,y2)->(x2,y2) ) intersects (within
    `margin`) a solid contour. One half of the strict oncoming AND: the box base must be ON the
    line. Stronger than the Stage-1 bottom-CENTER proximity trigger because it asks the whole
    base edge, but it overhangs the wheels (bumper), so it is only ever ANDed with a tire test.
    """
    x1, _y1, x2, y2 = bbox
    return any(segment_hits_contour((x1, y2), (x2, y2), c, margin=margin) for c in contours)


def bbox_bottom_corner_on_line(bbox: BBox, contours: Sequence[Sequence[Point]],
                               margin: float = 0.0) -> bool:
    """
    Ghost-mask ARM trigger (Spec C.3, tightened): True when the bottom-LEFT or bottom-RIGHT
    corner of the bbox is actively on a solid contour (within `margin`). We only start / refresh
    a track's ghost-line memory at the instant a base corner contacts the paint, so the ghost
    can never be armed by a vehicle merely driving NEAR a line. Trusting the bumper corner just
    for the mask trigger is acceptable because the actual hit is gated downstream by the
    tire-crosses / side-switch tests.
    """
    x1, _y1, x2, y2 = bbox
    bl, br = (x1, y2), (x2, y2)
    return any(point_near_contour(bl, c, margin) or point_near_contour(br, c, margin)
               for c in contours)


def signed_side_of_contour(p: Point, contour: Sequence[Point]) -> float:
    """
    Signed side (>0 left, <0 right, 0 on) of point `p` relative to the NEAREST edge of `contour`,
    oriented top->bottom so left/right is stable in image space. Basis for the temporal
    side-switch test: a vehicle whose bbox bottom-center flips sign across frames has physically
    crossed from one side of the line to the other.
    """
    if len(contour) < 2:
        return 0.0
    best = None
    for a, b in contour_edges(contour):
        d = point_to_segment_dist(p, a, b)
        if best is None or d < best[0]:
            best = (d, a, b)
    _, a, b = best
    if a[1] > b[1]:
        a, b = b, a
    return point_side_of_line(p, a, b)


def contour_curvature(contour: Sequence[Point]) -> float:
    """
    Curvature proxy in ~[0, 1] for a line-like contour: the max perpendicular deviation of its
    vertices from the contour's DIAMETER chord (its farthest-apart pair of points), divided by
    that chord length (sagitta / chord). ~0 = straight lane line (even if steeply angled or near
    horizontal); larger = bent (curve). Uncalibrated and RELATIVE -- it feeds the hysteresis
    curve gate, not a metric radius.

    The chord is the diameter, NOT the top-to-bottom span: a near-horizontal seg fragment has a
    tiny vertical span, which would explode a top/bottom-chord ratio (observed max ~5). The
    diameter is robust to line orientation. Meant for solid LINE contours (a filled
    traffic_island blob reads high; callers gate those out). Points are capped for an O(n^2)-safe
    diameter on dense contours.
    """
    pts = list(contour)
    if len(pts) < 3:
        return 0.0
    if len(pts) > 64:                              # subsample to keep the diameter search cheap
        step = len(pts) // 64 + 1
        pts = pts[::step]
    a, b, best = pts[0], pts[1], -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2
            if d > best:
                best, a, b = d, pts[i], pts[j]
    chord = math.sqrt(best)
    if chord <= 1e-6:
        return 0.0
    maxd = max(point_to_segment_dist(p, a, b) for p in pts)
    return maxd / chord
