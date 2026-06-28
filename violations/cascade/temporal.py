"""
Temporal components of the Stage-2 cascade.

Two cooperating pieces, both pure-python and unit-testable:

  KOfMFilter      -- Spec A. Per-track sliding window: a candidate must persist for >= K of the
                     last M frames before it survives Stage 1. Kills single-frame shadow/mirror/
                     bumper touches.

  GhostLineBuffer -- Spec C.3. Per-track rolling memory of the lane curve from the preceding
                     ~10-15 frames. When a vehicle drives ONTO a solid line it occludes the paint
                     and the seg model loses it; we then test tires / axle vectors against the
                     remembered ("ghost") curve instead of the (now missing) live one.

These are deliberately separate from tools/ghost_mask.py's GhostMaskTracker: that one carries a
forward-motion-compensated pixel *mask*; this one is a lightweight *polyline* memory keyed per
vehicle track, sized for the cascade sweep where we replay cached contours, not pixels.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Spec A: K-of-M sliding window
# --------------------------------------------------------------------------- #
class KOfMFilter:
    """
    Sliding-window persistence filter, one window per track id.

    Feed a boolean "candidate hit" per frame for each track; `update` returns True once that
    track has accumulated >= K hits within the last M observations (and stays True while the
    condition holds). M is the window length; only the last M observations are retained.

    Example: K=6, M=10 -> fire when 6 of the last 10 frames were hits.
    """

    def __init__(self, k: int, m: int):
        if k <= 0 or m <= 0 or k > m:
            raise ValueError(f"require 0 < k <= m (got k={k}, m={m})")
        self.k = k
        self.m = m
        self._win: Dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=m))

    def update(self, track_id: int, hit: bool) -> bool:
        win = self._win[track_id]
        win.append(1 if hit else 0)
        return sum(win) >= self.k

    def is_active(self, track_id: int) -> bool:
        win = self._win.get(track_id)
        return bool(win) and sum(win) >= self.k

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._win.clear()
        else:
            self._win.pop(track_id, None)


def k_of_m_events(hits: Sequence[bool], k: int, m: int) -> List[Tuple[int, int]]:
    """
    Offline helper for the sweep: given a per-frame hit sequence for ONE track, return the
    list of (start_frame, end_frame) windows where the K-of-M condition holds. Contiguous
    active frames are merged into a single event.
    """
    f = KOfMFilter(k, m)
    events: List[Tuple[int, int]] = []
    active = False
    start = 0
    for i, h in enumerate(hits):
        on = f.update(0, h)
        if on and not active:
            active, start = True, i
        elif not on and active:
            active = False
            events.append((start, i - 1))
    if active:
        events.append((start, len(hits) - 1))
    return events


# --------------------------------------------------------------------------- #
# Spec C.3: ghost-line buffer
# --------------------------------------------------------------------------- #
class HeadingEstimator:
    """
    Causal per-track heading proxy from bbox kinematics -- an HONEST stand-in for the production
    direction model when replaying the cache (no calibration, no model, history-only so it
    matches the live path frame-for-frame).

    Oncoming vehicles approach at closing speed (our speed + theirs): their bbox AREA grows
    steeply frame to frame. Same-direction (followed) vehicles hold a roughly stable apparent
    size. We estimate the AREA-GROWTH RATE per second over the last `window` frames and return
    it; the caller splits oncoming vs same with a single (swept) threshold, so the decision
    boundary costs nothing to re-tune.

        rate = (area_now / area_then) ** (fps / frame_span)     # multiplicative growth / sec

    Confound (documented, not hidden): a STOPPED or much-slower lead vehicle we close on also
    grows, so a high `rate` is "approaching", not strictly "oncoming". The swept threshold trades
    that off; in production the real direction model disambiguates. cf. orientation_pseudo_angle
    and project_plate_to_ground -- same candor.
    """

    def __init__(self, fps: float, window: int = 10, min_frames: int = 5):
        self.fps = fps if fps > 0 else 30.0
        self.window = window
        self.min_frames = min_frames
        self._hist: Dict[int, Deque[Tuple[int, float]]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, bbox: Sequence[float], frame_idx: int) -> Optional[float]:
        """Push this frame's bbox and return the causal area-growth rate / sec, or None until
        the track has `min_frames` of history. >1 = growing (approaching), ~1 = stable."""
        x1, y1, x2, y2 = bbox
        area = max(1.0, (x2 - x1) * (y2 - y1))
        h = self._hist[track_id]
        h.append((frame_idx, area))
        if len(h) < self.min_frames:
            return None
        f0, a0 = h[0]
        f1, a1 = h[-1]
        span = f1 - f0
        if span <= 0 or a0 <= 0:
            return None
        return (a1 / a0) ** (self.fps / span)

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._hist.clear()
        else:
            self._hist.pop(track_id, None)


def heading_is_oncoming(growth_rate: Optional[float], grow_thresh: float) -> bool:
    """Split HeadingEstimator's rate into oncoming (approaching fast) vs same/unknown. Unknown
    (None, too few frames) is treated as NOT oncoming so it takes the same-direction steering."""
    return growth_rate is not None and growth_rate >= grow_thresh


class CurveGate:
    """
    Stateful straight/curve classifier with HYSTERESIS, shared across the whole frame (not
    per-track). On a curve, perspective distortion makes distant vehicles look like they cross
    lane lines, so the dynamic distance gate chokes the tracking horizon closer when curved.

    Hysteresis is the whole point: a raw per-frame curvature threshold flickers as the noisy
    seg contours wobble, and under K=2 a single flicker frame could fire. So we use two
    thresholds and dwell counts -- flip to CURVED only after `on_frames` consecutive frames with
    curvature >= `hi`, flip back to STRAIGHT only after `off_frames` frames < `lo`. Between `lo`
    and `hi` the state is held.

    Feed the frame's MAX curvature across solid LINE contours (islands excluded by the caller).
    `is_curved` is then mapped to a swept horizon fraction (near for curves, far for straights).
    """

    def __init__(self, hi: float = 0.08, lo: float = 0.04, on_frames: int = 3, off_frames: int = 5):
        self.hi = hi
        self.lo = lo
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.curved = False
        self._above = 0
        self._below = 0

    def update(self, curvature: float) -> bool:
        if not self.curved:
            self._above = self._above + 1 if curvature >= self.hi else 0
            if self._above >= self.on_frames:
                self.curved = True
                self._below = 0
        else:
            self._below = self._below + 1 if curvature < self.lo else 0
            if self._below >= self.off_frames:
                self.curved = False
                self._above = 0
        return self.curved

    def reset(self) -> None:
        self.curved = False
        self._above = 0
        self._below = 0


class SideSwitchTracker:
    """
    Per-track memory of which side of the solid line a reference point (the bbox bottom-center)
    has occupied. Reports True once the track has been FIRMLY on BOTH sides at some point in its
    life -- i.e. it physically crossed from one side of the line to the other. This is the
    oncoming fallback signal: an oncoming vehicle legally in its lane stays on the far side
    forever and never switches (no FP); one drifting over the centerline toward us flips sides.

    `deadband` (px) ignores sub-pixel jitter right at the paint so noise near the line is not a
    'switch'. Side sign comes from geometry.signed_side_of_contour (>0 left / <0 right).
    """

    def __init__(self, deadband: float = 4.0):
        self.deadband = deadband
        self._pos: Dict[int, bool] = defaultdict(bool)
        self._neg: Dict[int, bool] = defaultdict(bool)

    def update(self, track_id: int, signed_side: float) -> bool:
        if signed_side > self.deadband:
            self._pos[track_id] = True
        elif signed_side < -self.deadband:
            self._neg[track_id] = True
        return self._pos[track_id] and self._neg[track_id]

    def has_switched(self, track_id: int) -> bool:
        return self._pos.get(track_id, False) and self._neg.get(track_id, False)

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._pos.clear()
            self._neg.clear()
        else:
            self._pos.pop(track_id, None)
            self._neg.pop(track_id, None)


class GhostLineBuffer:
    """
    Per-track rolling memory of solid-line contours.

    Each frame, before testing a vehicle, push the currently-visible solid-line contours that
    are spatially relevant to the vehicle. When the live paint vanishes (vehicle straddles it),
    `ghost_contours` returns the most recent remembered contours so the axle-vector / tire tests
    still have a line to hit.

    `maxlen` frames of memory (~10-15). Memory is keyed per track so two vehicles don't pollute
    each other's ghost line.
    """

    def __init__(self, maxlen: int = 12):
        self.maxlen = maxlen
        self._mem: Dict[int, Deque[List[List[Point]]]] = defaultdict(lambda: deque(maxlen=maxlen))

    def push(self, track_id: int, solid_contours: Sequence[Sequence[Point]]) -> None:
        """Record this frame's relevant solid-line contours for a track (skip empty pushes)."""
        if solid_contours:
            self._mem[track_id].append([list(c) for c in solid_contours])

    def ghost_contours(self, track_id: int) -> List[List[Point]]:
        """Most recent remembered contours for the track, or [] if no memory yet."""
        mem = self._mem.get(track_id)
        if not mem:
            return []
        return mem[-1]

    def all_ghost_contours(self, track_id: int) -> List[List[Point]]:
        """Union of every remembered frame's contours (wider net for fast forward motion)."""
        mem = self._mem.get(track_id)
        if not mem:
            return []
        out: List[List[Point]] = []
        for frame in mem:
            out.extend(frame)
        return out

    def has_memory(self, track_id: int) -> bool:
        return bool(self._mem.get(track_id))

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._mem.clear()
        else:
            self._mem.pop(track_id, None)
