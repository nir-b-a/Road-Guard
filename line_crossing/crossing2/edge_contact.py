"""v3 - bbox half-edge contact test (the rule you proposed), on the v2 pipeline.

Everything upstream and downstream is shared with v2: the same vehicle tracker, the
same lane model, the same Stage 0 lane tracks (association, fragment merging, row
fill, one type per track, double/flat suppression), and the same event objects,
overlay, CSV and video. ONLY the decision rule is different.

The rule, verbatim
------------------
For every frame, for every SOLID lane line, for every vehicle bbox:

  * line on the RIGHT of the image centre -> flag if the line touches
        the LEFT half of the bbox's bottom edge,   or
        the BOTTOM half of the bbox's left edge;
  * line on the LEFT of the image centre  -> mirrored:
        the RIGHT half of the bottom edge,          or
        the BOTTOM half of the right edge.

Which side the line is on is decided per frame from the line's x at its lowest
visible row, against the image centre - the same convention `_assign_positions`
uses for `Lane.position`.

How the two touches are computed
--------------------------------
  bottom edge : the bottom edge is horizontal at y = y2, so the line touches it
                exactly where its own x at that row falls inside the half in
                question - one interpolation, no segment intersection needed.
  side edge   : the edge is vertical at x = x1 (or x2). The line touches it if
                (x_line(y) - x_edge) CHANGES SIGN anywhere in the bottom half of
                that edge's y range - i.e. the polyline passes through it.

Buffer
------
`buffer_frac` inflates the bbox on all four sides by that fraction of its WIDTH
before the halves are computed, so the buffer scales with apparent size: near
vehicles have big boxes and get a big buffer, far ones a small one, with no camera
calibration anywhere. 0 disables it.

Temporal handling
-----------------
The rule is per-frame, and that is how it runs. Consecutive contact frames are then
grouped into one reported event purely so the CSV, the overlay and the video have a
span to show; `min_frames = 1` keeps the raw rule. Raise it to require persistence,
or set `latch = True` to reproduce the legacy "violator forever" behaviour.

Stage 1 gates (truncated box, occluded wheels, too-far vehicle) are NOT applied by
default, since they are not part of the proposed rule - `use_gates` turns them on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .anchors import build_anchors
from .config import CrossingConfig
from .lane_tracks import build_line_tracks, sample_lane
from .types import CrossingEvent, CrossingResult, FrameObservation


@dataclass
class EdgeContactConfig:
    """Knobs specific to v3. Stage 0 still reads `CrossingConfig`."""

    buffer_frac: float = 0.0
    """Inflate the bbox by this fraction of its width on every side before testing.
    Self-scaling with distance: no calibration, no metric range."""

    min_frames: int = 1
    """Consecutive contact frames needed to report. 1 = the rule exactly as proposed."""

    use_gates: bool = False
    """Apply the Stage 1 validity gates (frame-edge truncation, occluded wheels,
    too-small box). Off by default because they are not part of the proposed rule."""

    geometry: str = "tracked"
    """"tracked" tests against the Stage 0 line tracks (denser, temporally stable,
    survives dropouts); "raw" tests against this frame's lane polylines exactly as the
    lane model emitted them."""

    latch: bool = False
    """Once flagged, keep the vehicle flagged for the rest of its life - what the
    legacy CrossingMonitor did. Off by default."""

    lane_types: tuple[str, ...] | None = None
    """Restrict the rule to tracks whose voted type is one of these, e.g.
    ("solid_yellow",). None = every solid track, which is the standalone runner's
    behaviour and the default.

    This filters which markings can FIRE, not which ones are built: Stage 0 still
    sees every lane class, so the association, the lane-width ruler and the double
    suppression are computed from the whole scene exactly as they are without the
    filter. Feeding Stage 0 one colour only would quietly change all three."""


def _x_at(xs: np.ndarray, ys: np.ndarray, y: float) -> float:
    """Line x at image row y, or NaN outside the line's own vertical extent."""
    ok = np.isfinite(xs)
    if int(ok.sum()) < 2:
        return np.nan
    return float(np.interp(y, ys[ok], xs[ok], left=np.nan, right=np.nan))


def _crosses_vertical(xs: np.ndarray, ys: np.ndarray, x_edge: float,
                      y_lo: float, y_hi: float) -> bool:
    """True if the polyline passes through the vertical segment x = x_edge between
    y_lo and y_hi - i.e. (x_line(y) - x_edge) changes sign inside that band."""
    band = np.isfinite(xs) & (ys >= y_lo) & (ys <= y_hi)
    if int(band.sum()) < 2:
        return False
    d = xs[band] - x_edge
    return bool(d.min() <= 0.0 <= d.max())


def _test_bbox(xs: np.ndarray, ys: np.ndarray, bbox, frame_width: int,
               buffer_frac: float):
    """The proposed rule for one (line, vehicle) pair in one frame.

    Returns (hit, which, frac_over, probe_box, side) where `frac_over` is the share of
    the bbox bottom edge lying on the far side of the line - a rough "how much of the
    vehicle is across" - and `probe_box` is the buffered box actually tested."""
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    pad = buffer_frac * w
    bx1, by1, bx2, by2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    bw = max(1.0, bx2 - bx1)
    xm, ym = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)

    ok = np.isfinite(xs)
    if int(ok.sum()) < 2:
        return False, "", 0.0, (bx1, by1, bx2, by2), ""
    # Side of the image centre, from the line's LOWEST visible point.
    x_bottom = float(xs[ok][-1]) if ys[ok][-1] >= ys[ok][0] else float(xs[ok][0])
    side = "right" if x_bottom >= frame_width / 2.0 else "left"

    x_at_bottom = _x_at(xs, ys, by2)
    if side == "right":
        lo, hi = bx1, xm                       # left half of the bottom edge
        edge_x = bx1                           # ... and the left edge
        frac = (bx2 - x_at_bottom) / bw if np.isfinite(x_at_bottom) else 0.0
    else:
        lo, hi = xm, bx2                       # right half of the bottom edge
        edge_x = bx2                           # ... and the right edge
        frac = (x_at_bottom - bx1) / bw if np.isfinite(x_at_bottom) else 0.0

    if np.isfinite(x_at_bottom) and lo <= x_at_bottom <= hi:
        return True, "bottom", float(np.clip(frac, 0.0, 1.0)), (bx1, by1, bx2, by2), side
    if _crosses_vertical(xs, ys, edge_x, ym, by2):
        return True, "side", 1.0, (bx1, by1, bx2, by2), side
    return False, "", float(np.clip(frac, 0.0, 1.0)) if np.isfinite(frac) else 0.0, \
        (bx1, by1, bx2, by2), side


def detect_edge_contacts(observations: list[FrameObservation], frame_width: int,
                         frame_height: int, fps: float,
                         config: CrossingConfig | None = None,
                         ec_config: EdgeContactConfig | None = None,
                         stats: dict | None = None) -> CrossingResult:
    """Run v3. Returns the same `CrossingResult` v2 does, so every downstream
    consumer (overlay, CSV, annotated video) works unchanged."""
    cfg = config or CrossingConfig()
    ec = ec_config or EdgeContactConfig()
    n = len(observations)
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float32)
        return CrossingResult([], [], {}, [], empty, np.zeros((0, 2), np.float32),
                              np.zeros(0, np.int64), np.zeros(0))

    tracks, lane_width, van, ys = build_line_tracks(
        observations, frame_width, frame_height, fps, cfg)
    anchors = build_anchors(observations, lane_width, ys, van,
                            frame_width, frame_height, cfg, stats)
    valid_at = {}
    if ec.use_gates:
        for va in anchors.values():
            for k in range(len(va.idx)):
                valid_at[(va.vehicle_id, int(va.idx[k]))] = bool(va.valid[k])

    solid = [t for t in tracks if t.is_solid and not t.suppressed]
    if ec.lane_types is not None:
        solid = [t for t in solid if t.lane_type in ec.lane_types]
    tally = {"frames": n, "solid_tracks": len(solid), "tests": 0, "contacts": 0,
             "gated_out": 0, "by_bottom": 0, "by_side": 0}

    # (vehicle, line) -> [(frame index, frame id, which, frac)]
    hits: dict[tuple[int, int], list] = {}
    probes: dict[tuple[int, int], tuple] = {}

    for i, ob in enumerate(observations):
        if ec.geometry == "raw":
            lines = [(-1, sample_lane(ln, ys)) for ln in ob.lanes
                     if str(getattr(ln, "lane_type", "")).startswith("solid")
                     and (ec.lane_types is None
                          or getattr(ln, "lane_type", "") in ec.lane_types)]
        else:
            lines = [(t.track_id, t.X[i]) for t in solid]
        if not lines:
            continue
        for v in ob.vehicles:
            if ec.use_gates and not valid_at.get((v.track_id, i), True):
                tally["gated_out"] += 1
                continue
            best = None
            for tid, xs in lines:
                tally["tests"] += 1
                hit, which, frac, box, side = _test_bbox(
                    xs, ys, v.bbox, frame_width, ec.buffer_frac)
                if best is None or hit:
                    best = (box, side, hit)
                if not hit:
                    continue
                tally["contacts"] += 1
                tally["by_bottom" if which == "bottom" else "by_side"] += 1
                hits.setdefault((v.track_id, tid), []).append(
                    (i, ob.frame_id, which, frac))
                if hit:
                    break                       # one solid line is enough for this frame
            if best is not None:
                probes[(v.track_id, ob.frame_id)] = best

    events = _events_from_hits(hits, ec, fps)
    if ec.latch:
        events = _latch(events, observations)
    if stats is not None:
        stats["stage0"] = {
            "tracks": len(tracks), "solid": len(solid),
            "suppressed": sum(1 for t in tracks if t.suppressed),
            "types": ec.lane_types or "any",
        }
        stats["v3"] = tally

    return CrossingResult(
        events=events, tracks=tracks, anchors=anchors, offsets=[],
        lane_width=lane_width, vanishing=van,
        frame_ids=np.array([ob.frame_id for ob in observations], dtype=np.int64),
        row_grid=ys, probes=probes)


def _events_from_hits(hits: dict, ec: EdgeContactConfig, fps: float) -> list[CrossingEvent]:
    """Group consecutive contact frames into reportable spans. This is presentation
    only - it adds no evidence the per-frame rule did not already have."""
    out: list[CrossingEvent] = []
    for (vid, tid), rows in hits.items():
        rows.sort()
        run: list = []
        for rec in rows + [None]:
            if run and (rec is None or rec[0] != run[-1][0] + 1):
                if len(run) >= ec.min_frames:
                    out.append(_make_event(vid, tid, run, fps))
                run = []
            if rec is not None:
                run.append(rec)
    out.sort(key=lambda e: (e.start_frame, e.vehicle_id))
    return out


def _make_event(vid: int, tid: int, run: list, fps: float) -> CrossingEvent:
    n = len(run)
    fracs = [r[3] for r in run]
    peak = int(np.argmax(fracs))
    which = {"bottom": 0, "side": 0}
    for r in run:
        which[r[2]] += 1
    return CrossingEvent(
        vehicle_id=vid, track_id=tid, kind="crossing", lane_type="solid",
        start_frame=int(run[0][1]), end_frame=int(run[-1][1]),
        key_frame=int(run[peak][1]), direction="",
        peak_overlap=float(fracs[peak]),
        # Same shape of heuristic the legacy evaluate_crossing used: longer contact
        # runs are trusted more. It is NOT calibrated.
        confidence=float(min(1.0, max(0.30, n / max(1.0, 0.3 * fps)))),
        details={"n_frames": n, "by_bottom": which["bottom"], "by_side": which["side"],
                 "rule": "v3_half_edge_contact"},
    )


def _latch(events: list[CrossingEvent], observations) -> list[CrossingEvent]:
    """Legacy behaviour: extend each vehicle's first event to the end of its track."""
    last_seen: dict[int, int] = {}
    for ob in observations:
        for v in ob.vehicles:
            last_seen[v.track_id] = ob.frame_id
    first: dict[int, CrossingEvent] = {}
    for e in events:
        if e.vehicle_id not in first:
            first[e.vehicle_id] = e
            e.end_frame = last_seen.get(e.vehicle_id, e.end_frame)
            e.details["latched"] = True
    return list(first.values())
