"""Stage 0 - per-frame lane blobs -> tracked, typed, smoothed line objects.

A `Lane` has no identity across frames and its `lane_type` is re-voted every frame,
so a physical marking reads solid / unknown / solid / dashed / solid down a clip and
any consumer that gates on "is it solid THIS frame" loses events for no reason.
Stage 0 fixes both, and is the prerequisite for anything temporal downstream.

What it does
------------
  a. normalize the input to a centerline polyline  (CLRerNet `Lane`, or a
     segmentation contour via `contour_to_lane`, so the module is source-agnostic)
  b. sample every lane onto a fixed ROW GRID -> x as a function of image row
  c. associate detections to tracks frame-to-frame, PREDICTING each track's lateral
     motion first (lines sweep fast on curves and lane changes - exactly when the
     events happen - so a static gate breaks tracks at the worst moment)
  d. OFFLINE second pass: stitch fragments across gaps, so a marking hidden behind a
     truck for half a second stays one line instead of becoming two
  e. centered median+mean smoothing of the geometry on the row grid
  f. ONE lane-type decision per track, weighted toward the near field where dashed
     and solid are actually separable
  g. pair adjacent tracks -> the local lane width (the ruler Stage 2 divides by),
     plus double-line suppression and gore tagging

Output: `list[LineTrack]` + an (N, R) local-lane-width map + a per-frame vanishing
point. Nothing downstream needs to know which lane detector produced the input.
"""
from __future__ import annotations

import warnings

import numpy as np

from .config import CrossingConfig
from .smoothing import interp_gaps, smooth
from .types import FrameObservation, LineTrack


def sample_lane(lane, ys: np.ndarray) -> np.ndarray:
    """Resample a `Lane` polyline onto the row grid: x at every y, NaN outside the
    polyline's actual vertical extent (the same contract as `Lane.x_at_y`, but
    vectorized - `x_at_y` is O(segments) per call and this runs R*lanes*frames
    times)."""
    pts = getattr(lane, "points", None)
    if not pts or len(pts) < 2:
        return np.full(len(ys), np.nan, dtype=np.float32)
    arr = np.asarray(pts, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 1])]                      # ascending y, as np.interp needs
    yy, xx = arr[:, 1], arr[:, 0]
    keep = np.concatenate([[True], np.diff(yy) > 0])      # drop duplicate rows
    yy, xx = yy[keep], xx[keep]
    if len(yy) < 2:
        return np.full(len(ys), np.nan, dtype=np.float32)
    return np.interp(ys, yy, xx, left=np.nan, right=np.nan).astype(np.float32)


def contour_to_lane(contour, frame_width: int, *, step: int = 8,
                    lane_type: str = "unknown", score: float = 1.0):
    """ADAPTER: a segmentation polygon -> a centerline `Lane`.

    Lets this module consume main.py's `seg_lanes` records (YOLOv8-seg contours)
    without a second lane model, by taking, for every row the polygon spans, the
    midpoint of its horizontal extent. Returns None for blobs that are not lines.
    """
    from ..lane_types import Lane                          # local: keep imports cheap

    c = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(c) < 3:
        return None
    y0, y1 = float(c[:, 1].min()), float(c[:, 1].max())
    if y1 - y0 < 2 * step:
        return None
    nxt = np.roll(c, -1, axis=0)
    pts: list[tuple[int, int]] = []
    widths: list[float] = []
    for y in np.arange(y0, y1 + 1e-6, step):
        ya, yb = c[:, 1], nxt[:, 1]
        hit = ((ya <= y) & (y < yb)) | ((yb <= y) & (y < ya))
        if not hit.any():
            continue
        t = (y - ya[hit]) / (yb[hit] - ya[hit])
        xs = c[hit, 0] + t * (nxt[hit, 0] - c[hit, 0])
        pts.append((int(round(0.5 * (xs.min() + xs.max()))), int(round(y))))
        widths.append(float(xs.max() - xs.min()))
    if len(pts) < 2:
        return None
    if float(np.median(widths)) > 0.15 * frame_width:      # a blob, not a marking
        return None
    pts.sort(key=lambda p: p[1], reverse=True)             # bottom -> top, as the contract says
    return Lane(points=pts, lane_type=lane_type, score=score)


class _Frag:
    """One contiguous run of detections believed to be the same physical marking.

    State is the whole ROW GRID, not a handful of reference rows: a segmentation head
    routinely returns a marking as a short fragment that never reaches the near field,
    and a fixed-row signature is all-NaN for those - so they could never match and
    every frame would mint a new track."""

    __slots__ = ("idx", "xs", "types", "scores", "state", "vel", "last")

    def __init__(self, i: int, xs: np.ndarray, lane_type: str, score: float):
        self.idx = [i]
        self.xs = [xs]
        self.types = [lane_type]
        self.scores = [score]
        self.state = xs.astype(np.float64)
        self.vel = np.zeros_like(self.state)
        self.last = i

    def add(self, i: int, xs: np.ndarray, lane_type: str, score: float) -> None:
        dt = max(1, i - self.last)
        delta = (xs - self.state) / dt
        # EMA the lateral velocity, on the rows both the track and the detection have.
        self.vel = np.where(np.isfinite(delta), 0.5 * self.vel + 0.5 * delta, self.vel)
        self.state = np.where(np.isfinite(xs), xs, self.state)
        self.idx.append(i)
        self.xs.append(xs)
        self.types.append(lane_type)
        self.scores.append(score)
        self.last = i

    def predict(self, i: int) -> np.ndarray:
        return self.state + self.vel * (i - self.last)


def _grid_cost(pred: np.ndarray, xs: np.ndarray, min_overlap: int) -> float:
    """Mean |dx| over the grid rows the prediction and the detection share."""
    d = np.abs(pred - xs)
    ok = np.isfinite(d)
    n = int(ok.sum())
    return float(d[ok].mean()) if n >= min_overlap else np.inf


def _extrapolate(xs: np.ndarray, r_from: int, r_to: int, span: int = 5) -> float:
    """Value of `xs` continued along its local tangent from row r_from to r_to."""
    step = -1 if r_to < r_from else 1
    r_a = r_from - step * span
    if not (0 <= r_a < len(xs)) or not np.isfinite(xs[r_a]) or not np.isfinite(xs[r_from]):
        return np.nan
    slope = (xs[r_from] - xs[r_a]) / float(r_from - r_a)
    return float(xs[r_from] + slope * (r_to - r_from))


def _try_merge(a: np.ndarray, b: np.ndarray, tol: float, max_gap: int) -> np.ndarray | None:
    """Fuse two same-frame fragments of ONE marking, or return None.

    Two ways they can belong together: they overlap in rows and agree there
    (the head split one marking lengthwise), or they are vertically disjoint and one
    continues the other along its tangent (the head broke it at an occlusion)."""
    ok_a, ok_b = np.isfinite(a), np.isfinite(b)
    if not ok_a.any() or not ok_b.any():
        return None
    both = ok_a & ok_b
    if int(both.sum()) >= 3:
        if float(np.median(np.abs(a[both] - b[both]))) > tol:
            return None
    else:
        lo_a, hi_a = int(np.argmax(ok_a)), len(a) - 1 - int(np.argmax(ok_a[::-1]))
        lo_b, hi_b = int(np.argmax(ok_b)), len(b) - 1 - int(np.argmax(ok_b[::-1]))
        if lo_b > hi_a:
            gap, src, dst, r_from, r_to = lo_b - hi_a, a, b, hi_a, lo_b
        elif lo_a > hi_b:
            gap, src, dst, r_from, r_to = lo_a - hi_b, b, a, hi_b, lo_a
        else:
            return None
        if gap > max_gap:
            return None
        pred = _extrapolate(src, r_from, r_to)
        if not np.isfinite(pred) or abs(pred - dst[r_to]) > tol:
            return None
    out = np.where(ok_a & ok_b, (np.nan_to_num(a) + np.nan_to_num(b)) / 2.0,
                   np.where(ok_a, a, b))
    return out.astype(np.float32)


def _merge_frame_fragments(dets: list, frame_width: int, cfg: CrossingConfig) -> list:
    """Reconcile THIS frame's detections into physical markings before association.

    The polyline equivalent of a dilate + connected-components pass over the masks:
    without it, a head that emits a marking as three contours produces three tracks,
    three duplicate offset series and three copies of every event."""
    tol = cfg.frag_merge_tol_frac * frame_width
    dets = [list(d) for d in dets]
    merged = True
    while merged and len(dets) > 1:
        merged = False
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                fused = _try_merge(dets[i][0], dets[j][0], tol, cfg.frag_merge_gap_rows)
                if fused is None:
                    continue
                keep = i if np.isfinite(dets[i][0]).sum() >= np.isfinite(dets[j][0]).sum() else j
                dets[i] = [fused, dets[keep][1], max(dets[i][2], dets[j][2])]
                dets.pop(j)
                merged = True
                break
            if merged:
                break
    return dets


def _frame_spacing(dets: list, ref: np.ndarray, frame_width: int) -> float:
    """Local lane spacing estimated from THIS frame's detections alone - the
    association gate needs a ruler before any track exists."""
    gaps: list[float] = []
    for r in ref:
        xs = sorted(float(d[0][r]) for d in dets if np.isfinite(d[0][r]))
        if len(xs) >= 2:
            g = np.diff(xs)
            gaps.extend(g[g > 0.02 * frame_width].tolist())
    return float(np.median(gaps)) if gaps else 0.25 * frame_width


def _associate(obs: list[FrameObservation], ys: np.ndarray, ref: np.ndarray,
               frame_width: int, cfg: CrossingConfig, fps: float) -> list[_Frag]:
    """Frame-to-frame association with lateral prediction, then offline stitching."""
    coast = cfg.frames(cfg.track_coast_sec, fps)
    mid = len(ref) // 2
    active: list[_Frag] = []
    closed: list[_Frag] = []

    for i, ob in enumerate(obs):
        dets = []
        for lane in ob.lanes:
            xs = sample_lane(lane, ys)
            if not np.isfinite(xs).any():
                continue
            dets.append([xs, getattr(lane, "lane_type", "unknown"),
                         float(getattr(lane, "score", 1.0))])
        dets = _merge_frame_fragments(dets, frame_width, cfg)
        spacing = _frame_spacing(dets, ref, frame_width)
        gate = cfg.assoc_gate_lane_frac * spacing

        pairs = []
        for ti, tr in enumerate(active):
            pred = tr.predict(i)
            for di, d in enumerate(dets):
                c = _grid_cost(pred, d[0], cfg.assoc_min_overlap_rows)
                if c <= gate:
                    pairs.append((c, ti, di))
        pairs.sort()
        used_t: set[int] = set()
        used_d: set[int] = set()
        for c, ti, di in pairs:                            # greedy is ample for <=8 lanes
            if ti in used_t or di in used_d:
                continue
            xs, lt, sc = dets[di]
            active[ti].add(i, xs, lt, sc)
            used_t.add(ti)
            used_d.add(di)

        still: list[_Frag] = []
        for ti, tr in enumerate(active):
            if ti in used_t or i - tr.last <= coast:
                still.append(tr)
            else:
                closed.append(tr)
        active = still
        for di, d in enumerate(dets):
            if di not in used_d:
                active.append(_Frag(i, d[0], d[1], d[2]))

    closed.extend(active)
    return closed


def _stitch(frags: list[_Frag], cfg: CrossingConfig, frame_width: int,
            fps: float) -> list[_Frag]:
    """OFFLINE second pass: a marking lost behind a vehicle and reacquired at the
    same place is ONE line. A causal tracker cannot know that; we can, and it
    directly recovers events that would otherwise be split in half.

    The gate here uses a nominal quarter-frame lane width rather than a per-frame
    estimate: across a gap there is nothing better to measure, and the test only
    has to reject a jump to a DIFFERENT marking."""
    max_gap = cfg.frames(cfg.stitch_gap_sec, fps)
    gate = cfg.assoc_gate_lane_frac * 0.25 * frame_width
    chains: list[_Frag] = []
    for f in sorted(frags, key=lambda fr: fr.idx[0]):
        best, best_c = None, np.inf
        for ch in chains:
            if not (0 < f.idx[0] - ch.last <= max_gap):
                continue
            c = _grid_cost(ch.predict(f.idx[0]), f.xs[0], cfg.assoc_min_overlap_rows)
            if c < best_c and c <= gate:
                best, best_c = ch, c
        if best is None:
            chains.append(f)
            continue
        for j, i in enumerate(f.idx):
            best.add(i, f.xs[j], f.types[j], f.scores[j])
    return chains


def _row_fill(X: np.ndarray, cfg: CrossingConfig) -> np.ndarray:
    """Densify a track's geometry ALONG ROWS, per frame, by fitting its shape.

    Without this the detector quietly starves: a segmentation head emits a marking as
    short far-field segments, so the track has no x at the row where the vehicle's
    wheels are, and every Stage 2 sample is discarded as "no line geometry". A lane
    line is a smooth curve, so a quadratic through the rows we did observe is a far
    better estimate than nothing - bounded by `row_fill_extrap_rows` beyond the
    observed extent and never outside the rows the track has ever covered.

    Vectorized over frames (normal equations, batched solve): a per-frame polyfit loop
    would be ~200k fits on a clip and dominate the runtime.
    """
    n_frames, n_rows = X.shape
    r = np.arange(n_rows, dtype=np.float64)
    ok = np.isfinite(X)
    n = ok.sum(axis=1)
    if not ok.any():
        return X

    xz = np.where(ok, np.nan_to_num(X.astype(np.float64)), 0.0)
    rz = np.where(ok, r[None, :], 0.0)                 # masked rows contribute 0 to S1..S4
    s0 = n.astype(np.float64)
    s1, s2, s3, s4 = rz.sum(1), (rz ** 2).sum(1), (rz ** 3).sum(1), (rz ** 4).sum(1)
    t0, t1, t2 = xz.sum(1), (xz * rz).sum(1), (xz * rz * rz).sum(1)

    coef = np.full((n_frames, 3), np.nan)
    quad = n >= cfg.row_fill_min_points
    if quad.any():
        A = np.stack([np.stack([s0, s1, s2], -1),
                      np.stack([s1, s2, s3], -1),
                      np.stack([s2, s3, s4], -1)], axis=1)[quad]
        b = np.stack([t0, t1, t2], -1)[quad]
        good = np.abs(np.linalg.det(A)) > 1e-6
        if good.any():
            sol = np.linalg.solve(A[good], b[good])
            idx = np.flatnonzero(quad)[good]
            coef[idx] = sol
    lin = (n >= 2) & ~np.isfinite(coef[:, 0])
    if lin.any():
        A = np.stack([np.stack([s0, s1], -1), np.stack([s1, s2], -1)], axis=1)[lin]
        b = np.stack([t0, t1], -1)[lin]
        good = np.abs(np.linalg.det(A)) > 1e-6
        if good.any():
            sol = np.linalg.solve(A[good], b[good])
            idx = np.flatnonzero(lin)[good]
            coef[idx, 0], coef[idx, 1], coef[idx, 2] = sol[:, 0], sol[:, 1], 0.0

    first = np.where(n > 0, np.argmax(ok, axis=1), 0)
    last = np.where(n > 0, n_rows - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
    g_ok = ok.any(axis=0)
    g_lo, g_hi = int(np.argmax(g_ok)), n_rows - 1 - int(np.argmax(g_ok[::-1]))
    lo = np.maximum(first - cfg.row_fill_extrap_rows, g_lo)
    hi = np.minimum(last + cfg.row_fill_extrap_rows, g_hi)

    # The quadratic is used only BETWEEN the observed rows. Outside them it is
    # replaced by the tangent at the boundary: an extrapolated parabola diverges,
    # and a line that suddenly bends 200 px sideways reads as three vehicles all
    # crossing it in the same two frames.
    def _eval(rr):
        return coef[:, 0:1] + coef[:, 1:2] * rr + coef[:, 2:3] * (rr ** 2)

    def _slope(rr):
        return coef[:, 1:2] + 2.0 * coef[:, 2:3] * rr

    grid = np.broadcast_to(r[None, :], X.shape)
    f_lo, f_hi = first[:, None].astype(np.float64), last[:, None].astype(np.float64)
    inside = _eval(grid)
    below = _eval(f_lo) + _slope(f_lo) * (grid - f_lo)
    above = _eval(f_hi) + _slope(f_hi) * (grid - f_hi)
    fit = np.where(grid < f_lo, below, np.where(grid > f_hi, above, inside))

    band = (grid >= lo[:, None]) & (grid <= hi[:, None])
    fill = band & ~ok & np.isfinite(fit)
    return np.where(fill, fit, X).astype(np.float32)


def _vote_type(frag: _Frag, ys: np.ndarray, cfg: CrossingConfig,
               frame_height: int) -> tuple[str, dict[str, float]]:
    """ONE type per track. Each frame's vote is weighted by how much of the sampled
    polyline sat in the near field, because that is the only place dashed and solid
    are distinguishable - foreshortening makes distant dashes look continuous."""
    near = ys >= cfg.near_field_frac * frame_height
    votes: dict[str, float] = {}
    for xs, lt, sc in zip(frag.xs, frag.types, frag.scores):
        if lt == "unknown":
            continue
        w = float(np.isfinite(xs[near]).sum()) / max(1, int(near.sum()))
        if w <= 0.0:
            continue
        votes[lt] = votes.get(lt, 0.0) + w * max(sc, 0.1)
    if not votes:
        return "unknown", votes
    return max(votes, key=votes.get), votes


def _nearest_neighbour_width(stack: np.ndarray) -> np.ndarray:
    """(K, N, R) track geometry -> (N, R) local lane width: the median distance from
    a marking to its nearest neighbouring marking at the same row. Scale-free ruler
    for Stage 2, computed from the tracks themselves - no calibration anywhere."""
    if stack.shape[0] < 2:
        return np.full(stack.shape[1:], np.nan, dtype=np.float32)
    s = np.sort(stack, axis=0)                             # NaN sorts last
    g = np.diff(s, axis=0)
    left = np.full_like(s, np.nan)
    right = np.full_like(s, np.nan)
    left[1:] = g
    right[:-1] = g
    nn = np.fmin(left, right)                              # fmin ignores NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(nn, axis=0).astype(np.float32)


def _fit_lines(X: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame least-squares fit x = a*y + b over each frame's valid rows."""
    ok = np.isfinite(X)
    n = ok.sum(axis=1).astype(np.float64)
    Y = np.where(ok, ys[None, :], 0.0)
    Xv = np.where(ok, np.nan_to_num(X), 0.0)
    Sy, Sx = Y.sum(1), Xv.sum(1)
    Syy, Sxy = (Y * Y).sum(1), (Xv * Y).sum(1)
    den = n * Syy - Sy * Sy
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where((n >= 3) & (np.abs(den) > 1e-9), (n * Sxy - Sy * Sx) / den, np.nan)
        b = np.where(np.isfinite(a), (Sx - a * Sy) / np.maximum(n, 1), np.nan)
    return a, b


def _vanishing(tracks: list[LineTrack], ys: np.ndarray, n: int,
               frame_width: int, frame_height: int, cfg: CrossingConfig) -> np.ndarray:
    """Per-frame vanishing point from the pairwise intersections of the tracked
    lines. Stage 1 uses its x to decide which bottom corner of a bbox is a real
    wheel contact; the gore test uses its y."""
    fits = [_fit_lines(t.X, ys) for t in tracks]
    out = np.full((n, 2), np.nan, dtype=np.float32)
    for i in range(n):
        xs, yy = [], []
        for p in range(len(fits)):
            for q in range(p + 1, len(fits)):
                ap, bp = fits[p][0][i], fits[p][1][i]
                aq, bq = fits[q][0][i], fits[q][1][i]
                if not (np.isfinite(ap) and np.isfinite(aq)) or abs(ap - aq) < 1e-6:
                    continue
                y = (bq - bp) / (ap - aq)
                x = ap * y + bp
                if -frame_height <= y <= frame_height and -frame_width <= x <= 2 * frame_width:
                    xs.append(x)
                    yy.append(y)
        if xs:
            out[i] = (np.median(xs), np.median(yy))
    # A short centered median + gap fill: the vanishing point is a slowly varying
    # quantity and a per-frame estimate off two noisy fits is not.
    out = smooth(out, cfg.smooth_median_half, cfg.smooth_mean_half)
    out = interp_gaps(out, n)
    if not np.isfinite(out).any():
        out[:] = (frame_width / 2.0, frame_height * 0.45)
    else:                                                   # hold the median where unknown
        med = np.nanmedian(out, axis=0)
        bad = ~np.isfinite(out).all(axis=1)
        out[bad] = med
    return out


def build_line_tracks(obs: list[FrameObservation], frame_width: int, frame_height: int,
                      fps: float, cfg: CrossingConfig):
    """Run Stage 0. Returns (tracks, lane_width (N,R), vanishing (N,2), row_grid (R,))."""
    n = len(obs)
    ys = np.arange(0, frame_height, cfg.row_grid_step, dtype=np.float64)
    ref = np.array([int(np.argmin(np.abs(ys - f * frame_height))) for f in cfg.ref_row_fracs])

    frags = _associate(obs, ys, ref, frame_width, cfg, fps)
    frags = _stitch(frags, cfg, frame_width, fps)
    gap_frames = cfg.frames(cfg.stitch_gap_sec, fps)

    tracks: list[LineTrack] = []
    for tid, f in enumerate(frags):
        if len(f.idx) < 2:
            continue
        X = np.full((n, len(ys)), np.nan, dtype=np.float32)
        for i, xs in zip(f.idx, f.xs):
            X[i] = xs
        X = _row_fill(X, cfg)                          # densify along rows, per frame
        X = interp_gaps(X, gap_frames)
        X = smooth(X, cfg.smooth_median_half, cfg.smooth_mean_half)
        lt, votes = _vote_type(f, ys, cfg, frame_height)
        tracks.append(LineTrack(track_id=tid, X=X, lane_type=lt, votes=votes, n_obs=len(f.idx)))

    if not tracks:
        return [], np.full((n, len(ys)), np.nan, np.float32), \
            _vanishing([], ys, n, frame_width, frame_height, cfg), ys

    _suppress_flat(tracks, cfg)
    _suppress_doubles(tracks, ref, frame_width, cfg)
    live = [t for t in tracks if not t.suppressed] or tracks
    lane_width = _nearest_neighbour_width(np.stack([t.X for t in live]))
    van = _vanishing(live, ys, n, frame_width, frame_height, cfg)
    _tag_gores(tracks, ys, van, frame_height, cfg)
    return tracks, lane_width, van, ys


def _suppress_flat(tracks: list[LineTrack], cfg: CrossingConfig) -> None:
    """Drop tracks that are too close to horizontal to be lane boundaries.

    Catches two different failures with one test: a geometry fit that has swept a
    "line" across the whole frame (it then flips the sign of every vehicle's offset
    as it moves), and a genuine STOP LINE at a junction, which vehicles cross legally
    all day. Suppressed rather than deleted, so they still appear in the overlay."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in tracks:
            d = np.abs(np.diff(t.X, axis=1)) / float(cfg.row_grid_step)
            if not np.isfinite(d).any():
                t.suppressed = True
                continue
            if float(np.nanmedian(d)) > cfg.max_line_slope:
                t.suppressed = True


def _suppress_doubles(tracks: list[LineTrack], ref: np.ndarray,
                      frame_width: int, cfg: CrossingConfig) -> None:
    """Two solid markings a few centimetres apart are ONE boundary (a double line,
    or the two edges of one thick marking split by the detector). Reporting both
    would double-count every crossing, and the gap between them would be mistaken
    for a lane when Stage 2 looks for its ruler."""
    row = ref[0]
    seps: dict[tuple[int, int], float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for p in range(len(tracks)):
            for q in range(p + 1, len(tracks)):
                d = np.abs(tracks[p].X[:, row] - tracks[q].X[:, row])
                if np.isfinite(d).sum() >= 3:
                    seps[(p, q)] = float(np.nanmedian(d))
    if not seps:
        return
    wide = [v for v in seps.values() if v > 0.06 * frame_width]
    lane = float(np.median(wide)) if wide else 0.25 * frame_width
    for (p, q), v in sorted(seps.items(), key=lambda kv: kv[1]):
        a, b = tracks[p], tracks[q]
        if a.suppressed or b.suppressed:
            continue
        if v < cfg.double_line_frac * lane and a.is_solid and b.is_solid:
            weak, strong = (b, a) if a.n_obs >= b.n_obs else (a, b)
            weak.suppressed = True
            strong.is_double = True


def _tag_gores(tracks: list[LineTrack], ys: np.ndarray, van: np.ndarray,
               frame_height: int, cfg: CrossingConfig) -> None:
    """A parallel pair's separation scales with (y - y_vanishing). A pair that
    diverges faster than perspective predicts is a gore / painted island, where a
    vehicle sitting on the paint is normal. Tag it - do not drop it - so the
    overlay shows you whether the gore is what fired."""
    live = [t for t in tracks if not t.suppressed]
    if len(live) < 2:
        return
    r_near = int(np.argmin(np.abs(ys - 0.92 * frame_height)))
    r_far = int(np.argmin(np.abs(ys - 0.70 * frame_height)))
    y_vp = float(np.nanmedian(van[:, 1])) if np.isfinite(van[:, 1]).any() else 0.45 * frame_height
    denom = (ys[r_far] - y_vp)
    if abs(denom) < 1e-6:
        return
    expect = (ys[r_near] - y_vp) / denom
    if expect <= 0:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for p in range(len(live)):
            for q in range(p + 1, len(live)):
                sn = np.abs(live[p].X[:, r_near] - live[q].X[:, r_near])
                sf = np.abs(live[p].X[:, r_far] - live[q].X[:, r_far])
                ok = np.isfinite(sn) & np.isfinite(sf) & (sf > 1.0)
                if ok.sum() < 5:
                    continue
                if float(np.median(sn[ok] / sf[ok])) / expect > cfg.gore_divergence:
                    live[p].is_gore = True
                    live[q].is_gore = True
