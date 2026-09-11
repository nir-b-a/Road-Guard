"""Stage 3 - events from the signed offset.

Two channels on the same scalar, deliberately kept separate because they are not
the same violation:

  * TRAVERSE ("crossing") - the vehicle was committed to one side of the marking,
    moved across, and stayed committed to the other side. This is the event that
    matters. It is a SIGN CHANGE, which is why it survives a biased anchor: a
    constant bias shifts only WHEN the crossing is declared, it cannot invent or
    erase the event. An instantaneous contact test has no such protection.

  * STRADDLE ("strafe") - the line ran under the vehicle's body for long enough to
    be deliberate, without a completed change of sides. Reported at a lower
    severity: this is the "light strafing" case, and a traverse always contains one,
    so a straddle inside a traverse is folded into it rather than double-reported.

Both open as candidates and only confirm after a dwell, so no single frame can
produce an event - and nothing latches. Each event carries a start, an end and a
trace, which is what an evidence clip needs anyway.
"""
from __future__ import annotations

import numpy as np

from .config import CrossingConfig
from .smoothing import runs_of
from .types import CrossingEvent, LineTrack, OffsetSeries


def _flank(s: OffsetSeries, a: int, b: int, cfg: CrossingConfig) -> float | None:
    """Median bbox aspect over [a, b], or None when unknown. High means we are looking
    at the vehicle's SIDE - junction cross traffic rather than a lane manoeuvre."""
    if s.aspect is None:
        return None
    seg = s.aspect[a:b + 1]
    seg = seg[np.isfinite(seg)]
    return float(np.median(seg)) if len(seg) else None


def _overlap(u_l: np.ndarray, u_r: np.ndarray) -> np.ndarray:
    """Fraction of the corrected footprint on the far side of the line: 0 when the
    body is entirely on one side, 0.5 when the line runs down its middle."""
    span = u_r - u_l
    ok = np.isfinite(span) & (span > 1e-6) & (u_l < 0) & (u_r > 0)
    out = np.zeros(len(span), dtype=np.float64)
    out[ok] = np.minimum(-u_l[ok], u_r[ok]) / span[ok]
    return out


def _confidence(hold: float, net: float, mono: float, extrap: float) -> float:
    """Heuristic, and labelled as one. Rewards a decisive transit (large net offset
    change, monotone) that was measured on real geometry rather than on extrapolated
    line geometry, and that held its committed states."""
    c = (0.40
         + 0.20 * min(1.0, net / 0.60)
         + 0.20 * min(1.0, mono)
         + 0.10 * min(1.0, hold)
         + 0.10 * (1.0 - min(1.0, extrap)))
    return float(np.clip(c, 0.30, 1.0))


def _traverse(s: OffsetSeries, tr: LineTrack, fps: float,
              cfg: CrossingConfig) -> list[CrossingEvent]:
    u = s.u_c
    finite = np.isfinite(u)
    hold = cfg.frames(cfg.traverse_hold_sec, fps)
    max_transit = cfg.frames(cfg.traverse_max_transit_sec, fps)
    min_transit = cfg.frames(cfg.traverse_min_transit_sec, fps)
    ov = _overlap(s.u_l, s.u_r)

    committed = []
    for sign in (+1, -1):
        mask = finite & (u * sign > cfg.traverse_margin)
        committed += [(a, b, sign) for a, b in runs_of(mask) if b - a + 1 >= hold]
    committed.sort()

    out: list[CrossingEvent] = []
    for (a0, a1, sa), (b0, b1, sb) in zip(committed, committed[1:]):
        if sa == sb or b0 <= a1:
            continue
        transit = b0 - a1
        if transit < min_transit or transit > max_transit:
            continue
        seg = slice(a1, b0 + 1)
        if not finite[seg].all():
            continue                                   # a gap inside the transit: unprovable
        d = np.diff(u[seg])
        total = float(np.abs(d).sum())
        net = float(abs(u[b0] - u[a1]))
        mono = net / total if total > 1e-9 else 0.0
        if mono < cfg.traverse_monotonic:
            continue                                   # jitter that happened to end up across
        flank = _flank(s, a1, b0, cfg)
        if flank is not None and flank > cfg.max_event_aspect:
            continue                                   # side-on: cross traffic at a junction

        flips = np.flatnonzero(np.sign(u[a1:b0 + 1]) != np.sign(u[a1]))
        key = a1 + int(flips[0]) if len(flips) else b0
        extrap = float(s.extrapolated[seg].mean())
        out.append(CrossingEvent(
            vehicle_id=s.vehicle_id, track_id=s.track_id, kind="crossing",
            lane_type=tr.lane_type,
            start_frame=int(s.frame_ids[a1]), end_frame=int(s.frame_ids[b0]),
            key_frame=int(s.frame_ids[key]),
            direction="to_left" if sa > 0 else "to_right",
            peak_overlap=float(ov[seg].max()),
            confidence=_confidence((a1 - a0 + b1 - b0) / (2.0 * hold), net, mono, extrap),
            is_gore=tr.is_gore, is_double=tr.is_double, extrapolated_frac=extrap,
            details={"u_start": round(float(u[a1]), 3), "u_end": round(float(u[b0]), 3),
                     "transit_frames": int(transit), "monotonicity": round(mono, 3),
                     "hold_before": int(a1 - a0 + 1), "hold_after": int(b1 - b0 + 1)},
        ))
    return out


def _straddle(s: OffsetSeries, tr: LineTrack, fps: float, cfg: CrossingConfig,
              covered: list[tuple[int, int]]) -> list[CrossingEvent]:
    ov = _overlap(s.u_l, s.u_r)
    hold = cfg.frames(cfg.straddle_hold_sec, fps)
    mask = np.isfinite(s.u_c) & (ov >= cfg.straddle_min_frac)
    # "Approached from a side" means the vehicle held a committed position off the
    # line for a real dwell - not a stray sample at a filter edge.
    commit_hold = cfg.frames(cfg.traverse_hold_sec, fps)
    committed = np.isfinite(s.u_c) & (np.abs(s.u_c) > cfg.traverse_margin)
    commit_runs = [r for r in runs_of(committed) if r[1] - r[0] + 1 >= commit_hold]

    out: list[CrossingEvent] = []
    for a, b in runs_of(mask):
        if b - a + 1 < hold:
            continue
        if cfg.straddle_require_approach and not any(
                b0 < a or a0 > b for a0, b0 in commit_runs):
            continue                                   # never off the line: lane-hugging
        flank = _flank(s, a, b, cfg)
        if flank is not None and flank > cfg.max_event_aspect:
            continue                                   # side-on: cross traffic at a junction
        f0, f1 = int(s.frame_ids[a]), int(s.frame_ids[b])
        if any(cs <= f0 and f1 <= ce for cs, ce in covered):
            continue                                   # already reported as a full crossing
        peak = int(a + np.argmax(ov[a:b + 1]))
        extrap = float(s.extrapolated[a:b + 1].mean())
        out.append(CrossingEvent(
            vehicle_id=s.vehicle_id, track_id=s.track_id, kind="strafe",
            lane_type=tr.lane_type, start_frame=f0, end_frame=f1,
            key_frame=int(s.frame_ids[peak]), direction="",
            peak_overlap=float(ov[peak]),
            confidence=_confidence((b - a + 1) / (2.0 * hold), float(ov[peak]), 1.0, extrap),
            is_gore=tr.is_gore, is_double=tr.is_double, extrapolated_frac=extrap,
            details={"dwell_frames": int(b - a + 1),
                     "peak_overlap": round(float(ov[peak]), 3)},
        ))
    return out


def detect_events(offsets: list[OffsetSeries], tracks: list[LineTrack],
                  fps: float, cfg: CrossingConfig) -> list[CrossingEvent]:
    """Run both channels over every (vehicle, line) offset series."""
    by_id = {t.track_id: t for t in tracks}
    events: list[CrossingEvent] = []
    for s in offsets:
        tr = by_id.get(s.track_id)
        if tr is None:
            continue
        crossings = _traverse(s, tr, fps, cfg)
        events += crossings
        events += _straddle(s, tr, fps, cfg,
                            [(e.start_frame, e.end_frame) for e in crossings])
    events.sort(key=lambda e: (e.start_frame, e.vehicle_id))
    return events
