"""
Stage-2 cascade orchestrator (v3 -- pure-geometric, no plate proxy).

Given a Stage-1 candidate (a tracked vehicle whose contact point is near a solid line), decide
whether it is a CONFIRMED crossing.  The flow:

  1. Dynamic horizon gate (E)         -- curve-aware: choke nearer when road is curved.
  2. Ghost ARM (C.3)                  -- arm the ghost line ONLY when the bbox BOTTOM EDGE
                                         (not just a corner) intersects the lane contour.
  3. Heading split (kinematic)        -- oncoming (growing fast) vs same-direction.
  4. Geometric confirmation (BOTH headings):
       Condition A  -- a tire physically crosses the line (tire_crosses_line).
       Condition B  -- >= 2 tires straddling the line (axle_vector_straddle).
       Condition C  -- 0 tires detected: bbox bottom intersects line with ENLARGED margin.
     NOTE: license-plate proxy logic is FULLY REMOVED.  All confirmation is tire + bbox geometry.
  5. Oncoming FP killer kept         -- heading_rate gate still rejects negative-direction FP.
  6. K-of-M temporal persistence (A) -- only surviving frames count; fires on K-of-M.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from . import geometry as G
from .temporal import GhostLineBuffer, KOfMFilter, SideSwitchTracker, heading_is_oncoming

Point = Tuple[float, float]
BBox  = Tuple[float, float, float, float]

TRUCK_BUS_CLASSES = frozenset({5, 7})


@dataclass
class CascadeParams:
    """Every Stage-2 tunable. SWEPT ones are flagged."""
    # temporal (SWEPT)
    k: int = 2
    m: int = 8
    ghost_maxlen: int = 9
    # geometry margins (SWEPT)
    wheel_vector_margin: float = 2.0
    no_tire_margin_factor: float = 6.0   # Condition C: margin = wheel_vector_margin * factor
    # heading + strict oncoming (SWEPT)
    heading_grow_thresh: float = 2.5
    tire_cross_penetration: float = 3.0
    side_switch_deadband: float = 4.0
    corner_arm_margin: float = 4.0
    # dynamic curve distance gate (SWEPT)
    straight_horizon_frac: float = 0.0   # disabled — real-world distance gate used instead
    curve_horizon_frac: float = 0.0      # disabled — real-world distance gate used instead
    # curve hysteresis (fixed)
    curve_hi: float = 0.18
    curve_lo: float = 0.12
    curve_on_frames: int = 3
    curve_off_frames: int = 5
    # geometry toggles (SWEPT)
    exclude_traffic_island: bool = False
    require_axle_band: bool = False
    same_vehicle_axle: bool = False
    # real-world distance gate (replaces pixel-horizon heuristic)
    max_dist_m: float = 25.0        # 0 = disabled; gate vehicles farther than this
    distance_fy_px: float = 640.0   # focal length in pixels (W/2 / tan(FOV/2), FOV=90°, W=1280)
    distance_vehicle_h_m: float = 1.5  # assumed vehicle height for HeightBasedDistance formula
    # fixed knobs
    rear_aspect: float = 1.2
    side_aspect: float = 2.6
    axle_band_px: float = 25.0
    axle_max_sep_frac: float = 1.05
    expand_bottom_frac: float = 0.15
    use_ghost_when_no_live: bool = True
    solid_on_right: bool = True   # kept for legacy callers; not used in geometric logic

    def horizon_for(self, frame_h: int, is_curved: bool = False) -> float:
        frac = self.curve_horizon_frac if is_curved else self.straight_horizon_frac
        return frac * frame_h

    def depth_m(self, bbox: tuple) -> float:
        """Estimate forward distance from camera using bbox pixel height."""
        ph = max(1.0, bbox[3] - bbox[1])
        return self.distance_vehicle_h_m * self.distance_fy_px / ph

    def crossing_confidence(self, bbox: tuple) -> float:
        """
        Non-linear confidence score in (0, 1] for a solid-line crossing based on
        forward distance.  Close vehicles → high confidence; vehicles near the
        max_dist_m gate → low confidence.

        Formula: inverse-sigmoid centered at 60% of max_dist_m.
          center = 0.6 * max_dist_m  (9 m when gate = 15 m)
          scale  = max_dist_m / 5    (3 m)

        At max_dist_m = 15 m:
          2 m  → 0.93   (very close, almost certainly a crossing)
          6 m  → 0.73
          9 m  → 0.50   (midpoint)
          12 m → 0.27
          15 m → 0.10   (right at gate — borderline, needs human review)

        If distance gate is disabled (max_dist_m == 0) returns 1.0 (binary rule).
        """
        if self.max_dist_m <= 0:
            return 1.0
        d = self.depth_m(bbox)
        center = 0.6 * self.max_dist_m
        scale  = self.max_dist_m / 5.0
        return 1.0 / (1.0 + math.exp((d - center) / scale))


# Pluggable deps ---------------------------------------------------------------
TireDetectFn  = Callable[[object, BBox], List[Tuple[float, float, float, float, float]]]
# PlateLocateFn kept in signature for backward compat but is NEVER called internally
PlateLocateFn = Callable[[object, BBox], Optional[Tuple[Point, float, float, float]]]


@dataclass
class FrameDecision:
    hit: bool
    reason: str
    tires: List[Tuple[float, float, float, float, float]] = field(default_factory=list)
    angle: float = 0.0
    steer: str = ""
    heading_rate: Optional[float] = None
    is_curved: bool = False
    crossing_confidence: float = 1.0   # distance-based; only meaningful when hit=True


class Stage2Cascade:
    """
    Stateful per-clip cascade.  Call `step` once per frame per candidate vehicle.
    Returns (fired_after_KofM, FrameDecision).
    """

    def __init__(self, params: CascadeParams,
                 tire_detect: Optional[TireDetectFn] = None,
                 plate_locate: Optional[PlateLocateFn] = None):
        self.p = params
        self.tire_detect = tire_detect
        # plate_locate accepted for backward compat but intentionally unused
        self.kofm  = KOfMFilter(params.k, params.m)
        self.ghost = GhostLineBuffer(maxlen=params.ghost_maxlen)
        self.side  = SideSwitchTracker(deadband=params.side_switch_deadband)

    # ── shared confirmation logic (same for both headings) ─────────────────── #
    def _geometric_confirm(self, track_id: int, vehicle_bbox: BBox,
                           tire_boxes: List[BBox], contours: Sequence,
                           live_solid: Sequence, allow_sideswitch: bool = False
                           ) -> Tuple[bool, str]:
        """
        Conditions A / B / C checked in priority order.

        B (axle straddle, >=2 tires) checked first -- strongest geometric signal.
        A (any tire crosses the line) -- single-tire confirmation.
        C (0 tires) -- bbox bottom with enlarged margin as last resort.
        """
        # Condition B: clean 2-tire axle straddle
        if len(tire_boxes) >= 2:
            pair = (G.same_vehicle_axle_pair(tire_boxes, vehicle_bbox, self.p.axle_max_sep_frac)
                    if self.p.same_vehicle_axle else G.rear_axle_pair(tire_boxes))
            if pair is not None:
                if G.axle_vector_straddle(tire_boxes, contours, self.p.wheel_vector_margin):
                    if (not self.p.require_axle_band or
                            G.axle_midpoint_in_band(pair, contours, self.p.axle_band_px)):
                        return True, "axle_straddle"

        # Condition A: at least one tire crosses the line
        if tire_boxes:
            if any(G.tire_crosses_line(t, contours, self.p.tire_cross_penetration)
                   for t in tire_boxes):
                return True, "tire_crosses_line"
            return False, "tire_present_no_cross"

        # Condition C: no tires -- bbox bottom with enlarged margin
        big_margin = self.p.wheel_vector_margin * self.p.no_tire_margin_factor
        if G.bbox_bottom_intersects_line(vehicle_bbox, contours, big_margin):
            return True, "no_tire_bbox_bottom_fallback"

        # extra temporal side-switch gate (oncoming only)
        if allow_sideswitch and live_solid:
            bc = ((vehicle_bbox[0] + vehicle_bbox[2]) / 2.0, vehicle_bbox[3])
            switched = any(self.side.update(track_id, G.signed_side_of_contour(bc, c))
                           for c in live_solid)
            if switched:
                return True, "oncoming_sideswitch"

        return False, "no_evidence"

    # ── single-frame geometric verdict ─────────────────────────────────────── #
    def _frame_hit(self, frame, track_id: int, vehicle_bbox: BBox,
                   solid_contours: Sequence, frame_h: int,
                   vehicle_class: Optional[int], heading_rate: Optional[float],
                   is_curved: bool) -> FrameDecision:

        # (E) dynamic horizon gate
        if not G.passes_horizon_gate(vehicle_bbox, self.p.horizon_for(frame_h, is_curved)):
            return FrameDecision(False, "far_field_gated",
                                 heading_rate=heading_rate, is_curved=is_curved)

        # (E2) real-world distance gate (when enabled)
        if self.p.max_dist_m > 0 and self.p.depth_m(vehicle_bbox) > self.p.max_dist_m:
            return FrameDecision(False, "distance_gated",
                                 heading_rate=heading_rate, is_curved=is_curved)

        # Ghost ARM: arm ONLY when bbox BOTTOM EDGE intersects the lane (not just a corner)
        if G.bbox_bottom_intersects_line(vehicle_bbox, solid_contours, self.p.corner_arm_margin):
            self.ghost.push(track_id, solid_contours)

        live_solid = list(solid_contours)
        contours   = live_solid if live_solid else (
            self.ghost.all_ghost_contours(track_id) if self.p.use_ghost_when_no_live else [])
        if not contours:
            return FrameDecision(False, "no_line",
                                 heading_rate=heading_rate, is_curved=is_curved)

        tires      = self.tire_detect(frame, vehicle_bbox) if self.tire_detect else []
        tire_boxes = [t[:4] for t in tires]
        angle      = G.orientation_pseudo_angle(vehicle_bbox, self.p.rear_aspect, self.p.side_aspect)

        # Heading split: oncoming FP killer still active
        is_oncoming = heading_is_oncoming(heading_rate, self.p.heading_grow_thresh)
        steer       = "oncoming" if is_oncoming else "same"

        hit, reason = self._geometric_confirm(
            track_id, vehicle_bbox, tire_boxes, contours, live_solid,
            allow_sideswitch=is_oncoming,
        )
        return FrameDecision(hit, reason, tires, angle, steer, heading_rate, is_curved)

    def step(self, frame, track_id: int, vehicle_bbox: BBox,
             solid_contours: Sequence, frame_h: int,
             vehicle_class: Optional[int] = None, heading_rate: Optional[float] = None,
             is_curved: bool = False) -> Tuple[bool, FrameDecision]:
        decision = self._frame_hit(frame, track_id, vehicle_bbox, solid_contours,
                                   frame_h, vehicle_class, heading_rate, is_curved)
        fired = self.kofm.update(track_id, decision.hit)
        if fired:
            decision.crossing_confidence = self.p.crossing_confidence(vehicle_bbox)
        return fired, decision

    def reset(self) -> None:
        self.kofm.reset()
        self.ghost.reset()
        self.side.reset()
