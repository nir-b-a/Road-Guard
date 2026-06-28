"""
Unit tests for the Stage-2 cascade math (geometry + temporal). GPU-free; run with:
    python -m pytest tests/test_cascade.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from violations.cascade import geometry as G  # noqa: E402
from violations.cascade.temporal import (CurveGate, GhostLineBuffer,  # noqa: E402
                                         HeadingEstimator, KOfMFilter,
                                         SideSwitchTracker, heading_is_oncoming,
                                         k_of_m_events)
from violations.cascade.stage2 import CascadeParams, Stage2Cascade  # noqa: E402


# --------------------------------------------------------------------------- #
# segment / point primitives
# --------------------------------------------------------------------------- #
def test_segments_intersect_cross():
    assert G.segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))


def test_segments_intersect_disjoint():
    assert not G.segments_intersect((0, 0), (1, 1), (5, 5), (6, 6))


def test_segments_intersect_touching():
    assert G.segments_intersect((0, 0), (10, 0), (5, 0), (5, 10))


def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert G.point_in_polygon((5, 5), sq)
    assert not G.point_in_polygon((15, 5), sq)


def test_point_to_segment_dist():
    assert G.point_to_segment_dist((5, 5), (0, 0), (10, 0)) == pytest.approx(5.0)
    assert G.point_to_segment_dist((-5, 0), (0, 0), (10, 0)) == pytest.approx(5.0)  # past endpoint


def test_point_side_of_line():
    # directed line going DOWN the image (top->bottom): point to the right is negative
    assert G.point_side_of_line((10, 5), (0, 0), (0, 10)) < 0   # right of vertical line
    assert G.point_side_of_line((-10, 5), (0, 0), (0, 10)) > 0  # left


# --------------------------------------------------------------------------- #
# Spec B: smart crop
# --------------------------------------------------------------------------- #
def test_smart_crop_expands_bottom_and_clamps():
    x1, y1, x2, y2 = G.smart_crop_box((100, 100, 200, 200), 0.15, frame_w=640, frame_h=480)
    assert (x1, y1, x2) == (100, 100, 200)
    assert y2 == pytest.approx(115.0 + 100)  # 200 + 0.15*100


def test_smart_crop_clamps_to_frame_bottom():
    _, _, _, y2 = G.smart_crop_box((0, 0, 50, 470), 0.15, frame_w=640, frame_h=480)
    assert y2 == 479


# --------------------------------------------------------------------------- #
# Spec E: horizon gate
# --------------------------------------------------------------------------- #
def test_horizon_gate():
    assert G.passes_horizon_gate((0, 0, 10, 500), 300)       # contact below cutoff -> keep
    assert not G.passes_horizon_gate((0, 0, 10, 200), 300)   # far/high -> drop


# --------------------------------------------------------------------------- #
# Spec C.1: axle-vector straddle
# --------------------------------------------------------------------------- #
def _vertical_line_contour(x, y0, y1):
    # thin near-vertical solid line as a degenerate 2-point contour
    return [(x, y0), (x, y1)]


def test_axle_straddle_crosses():
    # two rear tires bracketing a vertical line at x=100 -> axle vector crosses it
    tires = [(60, 380, 90, 420), (110, 380, 140, 420)]   # L center ~75, R center ~125 at y=420
    contour = _vertical_line_contour(100, 300, 460)
    assert G.axle_vector_straddle(tires, [contour], margin=2.0)


def test_axle_straddle_no_cross():
    # both tires to the LEFT of the line -> no straddle
    tires = [(20, 380, 50, 420), (60, 380, 90, 420)]
    contour = _vertical_line_contour(200, 300, 460)
    assert not G.axle_vector_straddle(tires, [contour], margin=2.0)


def test_axle_straddle_margin_catches_near_miss():
    # axle vector ends just shy of the line; margin rescues it
    tires = [(20, 380, 50, 420), (80, 380, 95, 420)]   # right tire center ~87.5
    contour = _vertical_line_contour(92, 300, 460)
    assert not G.axle_vector_straddle(tires, [contour], margin=1.0)
    assert G.axle_vector_straddle(tires, [contour], margin=8.0)


# --------------------------------------------------------------------------- #
# Spec C.2: relative-position single tire
# --------------------------------------------------------------------------- #
def test_relative_position_right_boundary_crossed():
    line = [(100, 300), (100, 460)]
    tire_right = [(110, 400, 130, 430)]   # center x=120 -> right of line
    assert G.relative_position_violation(tire_right, line, solid_on_right=True, margin=2.0)


def test_relative_position_inside():
    line = [(100, 300), (100, 460)]
    tire_left = [(60, 400, 80, 430)]      # center x=70 -> left of line, not crossed
    assert not G.relative_position_violation(tire_left, line, solid_on_right=True, margin=2.0)


# --------------------------------------------------------------------------- #
# Spec D: plate parallax projection
# --------------------------------------------------------------------------- #
def test_plate_ground_projection_drops_y():
    gx, gy = G.project_plate_to_ground((300, 200), 40)
    assert (gx, gy) == (300, 240)


def test_plate_crosses_after_projection():
    # plate center sits left of the line, but projecting down + a right-leaning line crosses it
    line = [(100, 300), (140, 460)]
    # choose a plate whose grounded point lands right of the line near y=440
    assert G.plate_crosses_boundary((140, 400), line, solid_on_right=True,
                                    plate_parallax_offset_y=40, margin=1.0)


def test_is_parallel_ahead():
    assert G.is_parallel_ahead((0, 0, 100, 100), 0.25)        # square
    assert not G.is_parallel_ahead((0, 0, 200, 100), 0.25)    # wide


# --------------------------------------------------------------------------- #
# Spec A: K-of-M
# --------------------------------------------------------------------------- #
def test_k_of_m_basic_fire():
    f = KOfMFilter(3, 5)
    seq = [True, False, True, False, True]   # 3 hits in window of 5
    res = [f.update(7, h) for h in seq]
    assert res[-1] is True
    assert res[0] is False


def test_k_of_m_window_expires():
    f = KOfMFilter(2, 3)
    # two early hits then misses -> drops below K once they slide out
    out = [f.update(1, h) for h in [True, True, False, False, False]]
    assert out[1] is True
    assert out[-1] is False


def test_k_of_m_per_track_isolated():
    f = KOfMFilter(2, 3)
    f.update(1, True); f.update(1, True)
    assert f.is_active(1)
    assert not f.is_active(2)


def test_k_of_m_events_merges_contiguous():
    hits = [False, True, True, True, False, False, True, True, True, False]
    ev = k_of_m_events(hits, k=2, m=3)
    assert len(ev) == 2


def test_k_of_m_rejects_bad_params():
    with pytest.raises(ValueError):
        KOfMFilter(5, 3)


# --------------------------------------------------------------------------- #
# Spec C.3: ghost-line buffer
# --------------------------------------------------------------------------- #
def test_ghost_buffer_remembers_then_serves():
    g = GhostLineBuffer(maxlen=3)
    line = [(100, 300), (100, 460)]
    g.push(5, [line])
    assert g.has_memory(5)
    assert g.ghost_contours(5) == [line]
    # empty push does not erase memory
    g.push(5, [])
    assert g.ghost_contours(5) == [line]


def test_ghost_buffer_maxlen_and_union():
    g = GhostLineBuffer(maxlen=2)
    g.push(1, [[(0, 0), (0, 10)]])
    g.push(1, [[(1, 0), (1, 10)]])
    g.push(1, [[(2, 0), (2, 10)]])   # evicts the first
    allc = g.all_ghost_contours(1)
    assert len(allc) == 2


def test_ghost_buffer_per_track():
    g = GhostLineBuffer()
    g.push(1, [[(0, 0), (0, 10)]])
    assert not g.has_memory(2)


# --------------------------------------------------------------------------- #
# NEW primitives: tire-crosses / bbox-bottom / corner-arm / side / curvature
# --------------------------------------------------------------------------- #
_VLINE = [(100, 0), (102, 0), (102, 300), (100, 300)]   # thin vertical solid line at x~100


def test_tire_crosses_line_deep_vs_graze():
    # tire footprint (corners 80..120) straddles x=100 deeply -> crosses even at penetration 5
    assert G.tire_crosses_line((80, 270, 120, 300), [_VLINE], penetration=5)
    # tire whose left corner sits ~on the line -> a graze, rejected at penetration 5
    assert not G.tire_crosses_line((100, 270, 140, 300), [_VLINE], penetration=5)
    # tire fully left of the line -> never crosses
    assert not G.tire_crosses_line((40, 270, 90, 300), [_VLINE], penetration=0)


def test_bbox_bottom_intersects_line():
    assert G.bbox_bottom_intersects_line((60, 100, 160, 200), [_VLINE])      # base at y200 crosses
    assert not G.bbox_bottom_intersects_line((60, 100, 160, 400), [_VLINE])  # base below line extent


def test_bbox_bottom_corner_on_line():
    # bottom-left corner (98,150) within margin of x~100 -> armed
    assert G.bbox_bottom_corner_on_line((98, 100, 200, 150), [_VLINE], margin=5)
    # both bottom corners far from the line -> not armed
    assert not G.bbox_bottom_corner_on_line((140, 100, 200, 150), [_VLINE], margin=5)


def test_signed_side_of_contour_flips():
    left = G.signed_side_of_contour((50, 150), _VLINE)
    right = G.signed_side_of_contour((150, 150), _VLINE)
    assert (left > 0) and (right < 0)        # opposite signs => opposite sides


def test_contour_curvature_straight_vs_bent():
    straight = [(100, 0), (100, 150), (100, 300)]         # vertical line -> ~0
    angled = [(0, 0), (150, 150), (300, 300)]             # straight but 45deg -> still ~0
    bent = [(100, 0), (160, 150), (100, 300)]             # bowed -> > 0
    assert G.contour_curvature(straight) < 0.02
    assert G.contour_curvature(angled) < 0.02
    assert G.contour_curvature(bent) > 0.1


# --------------------------------------------------------------------------- #
# NEW temporal: heading estimator / curve gate / side-switch
# --------------------------------------------------------------------------- #
def test_heading_estimator_growth_splits_oncoming():
    he = HeadingEstimator(fps=30.0, window=10, min_frames=3)
    rate = None
    for i in range(6):                       # area doubles every frame -> steep growth
        s = 10 * (2 ** i)
        rate = he.update(1, (0, 0, s, s), i)
    assert rate is not None and rate > 4.0
    assert heading_is_oncoming(rate, grow_thresh=2.5)
    # a stable-size (followed) track -> low rate -> not oncoming
    he2 = HeadingEstimator(fps=30.0, window=10, min_frames=3)
    r2 = None
    for i in range(6):
        r2 = he2.update(2, (0, 0, 100, 100), i)
    assert not heading_is_oncoming(r2, grow_thresh=2.5)


def test_heading_unknown_is_not_oncoming():
    assert not heading_is_oncoming(None, grow_thresh=2.5)


def test_curve_gate_hysteresis():
    g = CurveGate(hi=0.2, lo=0.1, on_frames=3, off_frames=3)
    assert [g.update(c) for c in (0.3, 0.3)] == [False, False]   # not yet 3 highs
    assert g.update(0.3) is True                                 # 3rd high -> ON
    assert g.update(0.15) is True                                # mid-band -> held
    assert [g.update(0.0) for _ in range(2)] == [True, True]     # 2 lows -> still held
    assert g.update(0.0) is False                                # 3rd low -> OFF


def test_side_switch_tracker():
    s = SideSwitchTracker(deadband=2.0)
    assert not s.update(1, 10.0)         # firmly left only
    assert s.update(1, -10.0)            # now seen right too -> switched
    assert s.has_switched(1)
    assert not s.has_switched(2)


# --------------------------------------------------------------------------- #
# Stage2 integration (no model: inject fake detections; heading-steered)
# --------------------------------------------------------------------------- #
def _same(**kw):
    """CascadeParams tuned so a wide box gates IN and reads same-direction by default."""
    base = dict(straight_horizon_frac=0.1, heading_grow_thresh=2.5)
    base.update(kw)
    return CascadeParams(**base)


def test_stage2_axle_straddle_fires_after_kofm():
    p = _same(k=3, m=5, wheel_vector_margin=4.0)
    contour = [(100, 300), (100, 460)]
    cas = Stage2Cascade(p, tire_detect=lambda f, b: [(60, 400, 90, 440, 0.9),
                                                     (110, 400, 140, 440, 0.9)])
    fired = []
    for _ in range(5):
        f, dec = cas.step(None, 1, (40, 200, 160, 450), [contour], frame_h=720,
                          heading_rate=0.0)     # stable -> same-direction
        fired.append(f)
    assert dec.reason == "same_axle_straddle"
    assert fired[-1] is True and fired[0] is False    # K-of-M satisfied, not on frame 1


def test_stage2_far_field_gated():
    p = _same(k=1, m=1, straight_horizon_frac=0.9)    # horizon at 0.9*720=648 > y2=200
    cas = Stage2Cascade(p, tire_detect=lambda f, b: [])
    f, dec = cas.step(None, 1, (0, 0, 10, 200), [[(0, 0), (0, 10)]], frame_h=720)
    assert not f and dec.reason == "far_field_gated"


def test_stage2_uses_ghost_when_live_gone():
    p = _same(k=1, m=1, wheel_vector_margin=4.0, corner_arm_margin=60.0)
    contour = [(100, 300), (100, 460)]
    cas = Stage2Cascade(p, tire_detect=lambda f, b: [(60, 400, 90, 440, .9),
                                                     (110, 400, 140, 440, .9)])
    # frame 1: live line present + a bottom corner near it -> ghost ARMED & remembered
    cas.step(None, 1, (40, 200, 160, 450), [contour], frame_h=720, heading_rate=0.0)
    # frame 2: live line GONE (occluded) -> ghost still confirms
    f, dec = cas.step(None, 1, (40, 200, 160, 450), [], frame_h=720, heading_rate=0.0)
    assert f and dec.reason == "same_axle_straddle"


def test_stage2_oncoming_bottom_plus_tirecross():
    # oncoming: needs bbox-bottom on line AND a tire that CROSSES (penetrates) the line
    p = _same(k=1, m=1, wheel_vector_margin=2.0, tire_cross_penetration=3.0)
    contour = [(300, 200), (303, 200), (303, 520), (300, 520)]
    cas = Stage2Cascade(p, tire_detect=lambda f, b: [(280, 490, 320, 520, .9)])  # straddles x~300
    f, dec = cas.step(None, 1, (250, 300, 360, 520), [contour], frame_h=720,
                      vehicle_class=2, heading_rate=10.0)   # high growth -> oncoming
    assert f and dec.steer == "oncoming" and dec.reason == "oncoming_bottom+tirecross"


def test_stage2_oncoming_rejects_graze_without_cross():
    # oncoming with a tire that only touches (no penetration) -> no confirmation
    p = _same(k=1, m=1, wheel_vector_margin=2.0, tire_cross_penetration=8.0)
    contour = [(300, 200), (303, 200), (303, 520), (300, 520)]
    cas = Stage2Cascade(p, tire_detect=lambda f, b: [(302, 490, 340, 520, .9)])  # grazes right edge
    f, dec = cas.step(None, 1, (250, 300, 360, 520), [contour], frame_h=720,
                      vehicle_class=2, heading_rate=10.0)
    assert not f and dec.steer == "oncoming"


def test_stage2_same_direction_rear_on_uses_plate():
    p = _same(k=1, m=1, orientation_angle_threshold=30.0, plate_x_extension_pct=0.30,
              plate_parallax_offset_y=40.0, wheel_vector_margin=2.0)
    contour = [(300, 100), (300, 460)]
    cas = Stage2Cascade(p, tire_detect=lambda f, b: [],
                        plate_locate=lambda f, b: ((300.0, 200.0), 40.0, 20.0, 0.9))
    # square rear-on box (angle ~0), stable -> same-direction -> plate proxy
    f, dec = cas.step(None, 1, (250, 400, 350, 500), [contour], frame_h=720,
                      vehicle_class=2, heading_rate=0.0)
    assert f and dec.steer == "same" and "same_plate_proxy" in dec.reason


def test_stage2_same_direction_wide_no_tire_dropped():
    # wide side-on box, no tires, no plate path (angle above threshold) -> dropped to Stage-1
    p = _same(k=1, m=1, orientation_angle_threshold=30.0)
    contour = [(100, 300), (100, 460)]
    cas = Stage2Cascade(p, tire_detect=lambda f, b: [], plate_locate=lambda f, b: None)
    f, dec = cas.step(None, 1, (40, 400, 230, 500), [contour], frame_h=720,
                      vehicle_class=2, heading_rate=0.0)
    assert not f and dec.reason == "same_dropped_wideangle"


# --------------------------------------------------------------------------- #
# Orientation steering (aspect-ratio pseudo-angle)
# --------------------------------------------------------------------------- #
def test_orientation_pseudo_angle_bounds_and_monotonic():
    rear = (0, 0, 100, 100)     # r=1.0 <= rear_aspect -> 0 deg (rear-on)
    mid = (0, 0, 190, 100)      # r=1.9 -> halfway -> ~45 deg
    side = (0, 0, 300, 100)     # r=3.0 >= side_aspect -> clamp 90 deg
    a_rear = G.orientation_pseudo_angle(rear)
    a_mid = G.orientation_pseudo_angle(mid)
    a_side = G.orientation_pseudo_angle(side)
    assert a_rear == 0.0
    assert a_side == pytest.approx(90.0)
    assert a_rear < a_mid < a_side
    assert a_mid == pytest.approx(45.0, abs=1.0)


def test_aspect_ratio_degenerate():
    assert G.aspect_ratio((0, 0, 100, 0)) == 0.0


# --------------------------------------------------------------------------- #
# Same-vehicle axle gate + axle-midpoint band
# --------------------------------------------------------------------------- #
def test_same_vehicle_axle_pair_accepts_close():
    tires = [(60, 400, 90, 440), (110, 400, 140, 440)]      # centers 75 / 125, sep 50
    pair = G.same_vehicle_axle_pair(tires, (40, 200, 160, 450), max_sep_frac=1.05)
    assert pair is not None


def test_same_vehicle_axle_pair_rejects_wide():
    # two tires from different vehicles: separation >> vehicle width
    tires = [(20, 400, 40, 440), (300, 400, 330, 440)]      # sep ~ 285, veh width 120
    assert G.same_vehicle_axle_pair(tires, (40, 200, 160, 450), max_sep_frac=1.05) is None


def test_axle_midpoint_in_band():
    contour = [(300, 100), (300, 460)]
    on = ((280, 420), (320, 420))     # midpoint x=300 -> on the line
    off = ((60, 420), (140, 420))     # midpoint x=100 -> far
    assert G.axle_midpoint_in_band(on, [contour], band_px=25.0)
    assert not G.axle_midpoint_in_band(off, [contour], band_px=25.0)


# --------------------------------------------------------------------------- #
# Plate axle-proxy (30% extension)
# --------------------------------------------------------------------------- #
def test_plate_axle_proxy_span_and_drop():
    pl, pr = G.plate_axle_proxy((300, 200), plate_w=40, x_extension_pct=0.30,
                                plate_parallax_offset_y=40)
    # half = 40/2 + 0.30*40 = 20 + 12 = 32
    assert pl == pytest.approx((268.0, 240.0))
    assert pr == pytest.approx((332.0, 240.0))


def test_plate_proxy_straddle_hit_and_miss():
    contour = [(300, 100), (300, 460)]
    assert G.plate_proxy_straddle((300, 200), 40, [contour], 0.30, 40, margin=1.0)   # spans the line
    assert not G.plate_proxy_straddle((100, 200), 40, [contour], 0.30, 40, margin=1.0)  # left of line


def test_plate_proxy_zero_width_is_safe():
    assert not G.plate_proxy_straddle((300, 200), 0, [[(300, 100), (300, 460)]], 0.30, 40)


# --------------------------------------------------------------------------- #
# Same-direction truck heuristic (bus/truck bypass tires -> plate)
# --------------------------------------------------------------------------- #
def test_stage2_truck_bypasses_tires_to_plate():
    # wide box (angle > threshold) would normally need a 2-tire straddle; class=7 (truck) with a
    # single tire falls through to the plate axle-proxy instead.
    p = _same(k=1, m=1, orientation_angle_threshold=30.0, plate_x_extension_pct=0.30,
              plate_parallax_offset_y=40.0, wheel_vector_margin=2.0)
    contour = [(300, 100), (300, 460)]
    cas = Stage2Cascade(p, tire_detect=lambda f, b: [(60, 400, 90, 440, .9)],   # only 1 tire
                        plate_locate=lambda f, b: ((300.0, 200.0), 40.0, 20.0, 0.9))
    f, dec = cas.step(None, 1, (110, 400, 300, 500), [contour], frame_h=720,
                      vehicle_class=7, heading_rate=0.0)
    assert f and dec.steer == "same" and "same_plate_proxy" in dec.reason
    assert "[truck]" in dec.reason
