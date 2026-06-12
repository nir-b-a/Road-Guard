"""
MotionDirectionFilter -- classical-CV oncoming-vehicle damper for Road Guard.

Static pixel geometry cannot tell a same-direction line-driver from an oncoming car that
merely bleeds over the solid line in 2D projection (proven on the S9k portrait clip: a real
violator's centre sits past the line exactly like an oncoming car does). So we decide
DIRECTION from MOTION only -- cached bbox + ego optical-flow shift, no DL, no extra tracking
-- turn it into an oncoming probability P in [0,1], and instead of rejecting we RAISE the
consecutive-frame requirement for suspect tracks:

    K_dynamic = K_base * (1 + alpha * P)

Recall-first: a same-direction violator has P~0 and keeps the normal K. Only a strongly
oncoming track must persist much longer to register -- which clean-highway oncoming traffic,
passing through in a fraction of a second, never does.

Signals (both ego-motion aware, both from cached data):
  A) Y-ORIGIN ANCHOR  -- oncoming tracks are born near the vanishing point (small y) and
     slide DOWN; if a track's lifetime min_y is above the horizon AND it is now below
     mid-frame, that's the oncoming signature.
  B) EGO-COMPENSATED VERTICAL VELOCITY (V_y) -- oncoming cars slide down the image faster
     than the background does: V_y = ((cy_now - cy_old) - sum_dy_ego) / bbox_height >> 0.
"""
from __future__ import annotations

from collections import defaultdict, deque


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class MotionDirectionFilter:
    """Streaming per-track oncoming-motion estimator -> dynamic consecutive-frame threshold.

    All thresholds that depend on image position are given as FRACTIONS of frame height so
    the filter is resolution/orientation agnostic (works on 1920x1080 landscape and the
    608x1080 portrait clip alike).
    """

    def __init__(self, H: int, *, window: int = 10,
                 horizon_frac: float = 0.45, mid_frac: float = 0.55,
                 vy_scale: float = 0.6, w_vy: float = 0.7, w_anchor: float = 0.3,
                 fusion: str = "sum", k_base: float = 15.0, alpha: float = 3.0,
                 stale_frames: int = 30):
        self.H = float(H)
        self.window = int(window)
        self.horizon_y = horizon_frac * H      # above this y (smaller) = near the vanishing point
        self.mid_y = mid_frac * H              # below this y (larger) = approaching / near
        self.vy_scale = float(vy_scale)        # V_y that maps to P_vy = 1.0
        self.w_vy, self.w_anchor = float(w_vy), float(w_anchor)
        self.fusion = fusion                   # "sum" (weighted) or "max"
        self.k_base = float(k_base)
        self.alpha = float(alpha)
        self.stale_frames = int(stale_frames)
        self._t: dict[int, dict] = {}          # track_id -> lightweight state

    # ------------------------------------------------------------------ #
    def update(self, frame_idx: int, tracks: list, dy_ego: float) -> None:
        """Fold one frame of tracks into the state. `tracks` items need `track_id` and
        `bbox` [x1,y1,x2,y2]; `contact_y` is used if present, else bbox bottom (y2)."""
        dy_ego = float(dy_ego)
        for v in tracks:
            tid = v["track_id"]
            x1, y1, x2, y2 = v["bbox"]
            cy = float(v.get("contact_y", y2))
            h = max(1.0, float(y2 - y1))
            s = self._t.get(tid)
            if s is None:
                s = self._t[tid] = {"buf": deque(maxlen=self.window),
                                    "dy": deque(maxlen=self.window),
                                    "min_y": cy, "h": h, "cy": cy, "seen": frame_idx}
            s["buf"].append((frame_idx, cy))
            s["dy"].append(dy_ego)
            if cy < s["min_y"]:
                s["min_y"] = cy
            s["h"], s["cy"], s["seen"] = h, cy, frame_idx
        # memory management: drop tracks unseen for a while (prevents unbounded growth)
        if self._t:
            for tid in [t for t, s in self._t.items() if frame_idx - s["seen"] > self.stale_frames]:
                del self._t[tid]

    # ------------------------------------------------------------------ #
    def _vy(self, s: dict) -> float:
        buf = s["buf"]
        if len(buf) < 2:
            return 0.0
        (f0, y0), (f1, y1) = buf[0], buf[-1]
        if f1 <= f0:
            return 0.0
        accum_dy = 0.0
        for d in s["dy"]:          # ego vertical motion accumulated over the window
            accum_dy += d
        return ((y1 - y0) - accum_dy) / s["h"]

    def _anchor(self, s: dict) -> float:
        return 1.0 if (s["min_y"] < self.horizon_y and s["cy"] > self.mid_y) else 0.0

    def oncoming_score(self, track_id: int) -> float:
        s = self._t.get(track_id)
        if s is None:
            return 0.0
        p_vy = _clip01(self._vy(s) / self.vy_scale)   # only downward slide (oncoming) counts
        anc = self._anchor(s)
        p = max(self.w_vy * p_vy, self.w_anchor * anc) if self.fusion == "max" \
            else self.w_vy * p_vy + self.w_anchor * anc
        return _clip01(p)

    def get_dynamic_K(self, track_id: int) -> float:
        """Consecutive-frame requirement for this track: K_base for same-direction (P~0),
        stretched up to K_base*(1+alpha) for strongly oncoming tracks."""
        return self.k_base * (1.0 + self.alpha * self.oncoming_score(track_id))

    def features(self, track_id: int) -> dict:
        """Raw signals for CSV/threshold tuning."""
        s = self._t.get(track_id)
        if s is None:
            return {"v_y": 0.0, "anchor": 0.0, "p": 0.0, "k_dyn": self.k_base,
                    "min_y": 0.0, "cy": 0.0, "h": 0.0}
        return {"v_y": self._vy(s), "anchor": self._anchor(s),
                "p": self.oncoming_score(track_id), "k_dyn": self.get_dynamic_K(track_id),
                "min_y": s["min_y"], "cy": s["cy"], "h": s["h"]}


# --------------------------------------------------------------------------- #
# Cheap per-clip helpers (no cv2): oncoming P per (track,frame) + dynamic-K events
# --------------------------------------------------------------------------- #
def compute_motion_scores(frames, H, **mf_kwargs) -> dict:
    """One light pass over cached frames -> {track_id: {frame: oncoming_P}}. P does NOT
    depend on K, so compute once and sweep K_base cheaply afterwards."""
    mf = MotionDirectionFilter(H, **mf_kwargs)
    out = defaultdict(dict)
    for fr in frames:
        dy = fr.get("shift", [0.0, 0.0])[1]
        mf.update(fr["frame"], fr["vehicles"], dy)
        for v in fr["vehicles"]:
            out[v["track_id"]][fr["frame"]] = mf.oncoming_score(v["track_id"])
    return out


def merge_events(events, fps, cooldown_sec=3.0):
    """Per-track cooldown: collapse events from the SAME vehicle whose gap is <= cooldown into
    one incident (never merges different tracks). One alert/clip per vehicle per incident."""
    gap = cooldown_sec * fps
    by = defaultdict(list)
    for tid, s, e in events:
        by[tid].append((s, e))
    out = []
    for tid, evs in by.items():
        evs.sort()
        cs, ce = evs[0]
        for s, e in evs[1:]:
            if s - ce <= gap:
                ce = max(ce, e)
            else:
                out.append((tid, cs, ce)); cs, ce = s, e
        out.append((tid, cs, ce))
    return out


def _eff_hit(rec, P, far_bias, W):
    """Idea A: a recorded hit only counts if the surface intersects the FAR-from-ego side of
    the bbox bottom. Shift is scaled by oncoming P, so same-direction cars (P~0) are untouched.
    far_bias=0 reduces to the plain baseline hit (rec['on'])."""
    if not rec["on"]:
        return False
    e = far_bias * P
    if e <= 0.0:
        return True
    e = min(1.0, e)
    fracs = rec["fracs"]
    if not fracs:
        return False
    if rec["cx"] < 0.5 * W:                  # ego is to the right -> far side is the LEFT
        hi = 0.90 - e * 0.80
        return any(f <= hi for f in fracs)
    hi_lo = 0.10 + e * 0.80                   # ego to the left -> far side is the RIGHT
    return any(f >= hi_lo for f in fracs)


def _horizon_weight(y, horizon_y):
    """Idea B: ~1 near the horizon (small y), ~0 low in the frame (large y, close = trustworthy)."""
    if y >= horizon_y:
        return 0.0
    return min(1.0, (horizon_y - y) / horizon_y)


def events_configured(timeline, scores, k_base_frames, H, W, *, far_bias=0.0, alpha=0.0,
                      horizon_gain=0.0, horizon_frac=0.45, agg="median"):
    """Unified, fully-isolatable event extractor over an ENRICHED timeline.
      Idea A  : far_bias > 0  -> require deep far-side overlap (scaled by P).
      Idea B  : horizon_gain  -> stretch K by proximity-to-horizon weight.
      dyn-K   : alpha         -> stretch K by oncoming motion P.
    With all three at 0 this is exactly the static baseline. K_dynamic per run:
      K = k_base * (1 + alpha*P_agg + horizon_gain*Hweight_agg)."""
    import numpy as np
    horizon_y = horizon_frac * H
    events = []
    for tid, fh in timeline.items():
        frames = sorted(fh)
        i = 0
        while i < len(frames):
            f = frames[i]
            if not _eff_hit(fh[f], scores.get(tid, {}).get(f, 0.0), far_bias, W):
                i += 1
                continue
            run = [f]
            j = i + 1
            while j < len(frames) and frames[j] == frames[j - 1] + 1 \
                    and _eff_hit(fh[frames[j]], scores.get(tid, {}).get(frames[j], 0.0), far_bias, W):
                run.append(frames[j])
                j += 1
            ps = [scores.get(tid, {}).get(ff, 0.0) for ff in run]
            hs = [_horizon_weight(fh[ff]["y"], horizon_y) for ff in run]
            if agg == "max":
                p_agg, h_agg = max(ps), max(hs)
            elif agg == "mean":
                p_agg, h_agg = sum(ps) / len(ps), sum(hs) / len(hs)
            else:
                p_agg, h_agg = float(np.median(ps)), float(np.median(hs))
            need = k_base_frames * (1.0 + alpha * p_agg + horizon_gain * h_agg)
            if len(run) >= need:
                events.append((tid, run[0], run[-1]))
            i = j
    return events


def events_from_timeline_dynamic(timeline, scores, k_base_frames, alpha, agg="median"):
    """Like events_from_timeline, but each consecutive verdict-hit RUN must be at least
    K_dynamic = k_base * (1 + alpha * P_agg) long, where P_agg aggregates the track's
    oncoming score over the run. Recall-first: same-direction runs (P~0) need only k_base."""
    import numpy as np
    events = []
    for tid, fh in timeline.items():
        frames = sorted(fh)
        i = 0
        while i < len(frames):
            if not fh[frames[i]]:
                i += 1
                continue
            start = end = frames[i]
            j = i + 1
            while j < len(frames) and fh[frames[j]] and frames[j] == frames[j - 1] + 1:
                end = frames[j]
                j += 1
            run = frames[i:j]
            ps = [scores.get(tid, {}).get(f, 0.0) for f in run]
            if not ps:
                p_agg = 0.0
            elif agg == "max":
                p_agg = max(ps)
            elif agg == "mean":
                p_agg = sum(ps) / len(ps)
            else:
                p_agg = float(np.median(ps))
            need = k_base_frames * (1.0 + alpha * p_agg)
            if (end - start + 1) >= need:
                events.append((tid, start, end))
            i = j
    return events
