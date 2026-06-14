"""
Road Guard occlusion + line-stability logic.

Two cooperating ideas:

A) STATE-TRIGGERED GHOST MASK (occlusion) -- a violating vehicle covers the line it sits on, so
   the segmentation model can't label it exactly when the violation happens. When a tire trigger
   dot first touches a violation surface we snapshot a "ghost" of that surface, bind it to the
   vehicle's ByteTrack id, and carry it forward by compensating for the dashcam's forward motion
   (Lucas-Kanade optical flow + an affine shift -- strictly OpenCV pixel math, no DL). A wide
   central "verdict line" then judges against the ghost while the real line is hidden.

B) LINE-TRACKING STATE MACHINE (false-positive control) -- per-frame class noise (a white line
   flipping to yellow for a few frames, or a phantom line popping in for a second) creates fake
   violations. We give each physical lane line a persistent ID across frames and apply:
     * color hysteresis: a historically-white line ignores low-confidence yellow flips;
     * Israeli road prior: yellow is expected only on the FAR-RIGHT edge -- center/left yellow is
       heavily penalised unless near-certain;
     * phantom filter: the violation logic may only read a line that has existed >= ~1 s, OR whose
       mask confidence is exceptionally high (> 0.85).

GEOMETRY (vehicle bbox, width W, x1..x2, bottom y2):
  Trigger dots:  x1+0.10W and x1+0.90W  at y2        (lock-on, near the tires)
  Verdict line:  x1+0.15W .. x1+0.85W   at y2        (the judge -- kept WIDE for recall)
"""
from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

VIOLATION_CLASSES = {"solid_white_lane", "traffic_island"}
ALL_CLASSES = ("dashed_lane", "solid_white_lane", "traffic_island", "yellow_solid_lane")
WHITE_CLASSES = {"dashed_lane", "solid_white_lane"}

# Line-tracker priors / hysteresis
VOTE_DECAY = 0.90              # recency decay of per-line class votes
RIGHT_X_FRAC = 0.78           # x beyond this fraction of width = "far right edge" (yellow ok)
YELLOW_CENTER_PRIOR = 0.25    # multiplier on yellow votes NOT on the far right (Israeli prior)
HIGH_CONF = 0.85              # mask conf that bypasses the phantom age gate
ASSOC_GATE_FRAC = 0.07        # centroid association gate (fraction of frame diagonal)
TRACK_MAX_MISS = 8            # drop a line track after this many unmatched frames

# Side-of-line gate (oncoming-traffic false-positive suppression)
SIDE_TOL_FRAC = 0.20          # car must be >this fraction of its width off the line to count as "on a side"
SIDE_BAND_PX = 6              # half-height of the contact-row band sampled for the line's x
SIDE_MARGIN_FRAC = 0.15       # widen the vehicle x-window by this fraction when sampling the line


# --------------------------------------------------------------------------- #
# Ego-motion (Lucas-Kanade optical flow on background asphalt)
# --------------------------------------------------------------------------- #
def estimate_ego_shift(prev_gray: np.ndarray, cur_gray: np.ndarray) -> tuple[float, float]:
    H, W = prev_gray.shape[:2]
    roi = np.zeros((H, W), np.uint8)
    roi[int(H * 0.55):H, int(W * 0.20):int(W * 0.80)] = 255
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=80, qualityLevel=0.01,
                                 minDistance=8, mask=roi)
    if p0 is None or len(p0) < 6:
        return (0.0, 0.0)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, p0, None,
                                         winSize=(21, 21), maxLevel=2)
    if p1 is None or st is None:
        return (0.0, 0.0)
    good = st.flatten() == 1
    if good.sum() < 6:
        return (0.0, 0.0)
    d = (p1[good] - p0[good]).reshape(-1, 2)
    return (float(np.median(d[:, 0])), float(np.median(d[:, 1])))


# --------------------------------------------------------------------------- #
# Spatial reconciliation: lane fragments -> physical-line COMPONENTS (#4, kept as-is)
# --------------------------------------------------------------------------- #
def reconcile_components(lanes, H: int, W: int):
    """Merge geometrically-continuous fragments (dilate + connected components = one physical
    line). Returns (components, label_img, union_mask) where each component is
    {id, cls(majority by pixel area), conf(max mask conf), cx, cy}."""
    if not lanes:
        return [], np.zeros((H, W), np.int32), np.zeros((H, W), np.uint8)
    masks = {c: np.zeros((H, W), np.uint8) for c in ALL_CLASSES}
    conf_map = np.zeros((H, W), np.float32)
    for l in lanes:
        if l["cls"] not in masks:
            continue
        cnt = np.asarray(l["contour"], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(masks[l["cls"]], [cnt], 255)
        tmp = np.zeros((H, W), np.uint8)
        cv2.fillPoly(tmp, [cnt], 1)
        conf_map = np.maximum(conf_map, tmp.astype(np.float32) * float(l["conf"]))

    union = np.zeros((H, W), np.uint8)
    for m in masks.values():
        union = cv2.bitwise_or(union, m)
    if not union.any():
        return [], np.zeros((H, W), np.int32), union

    k = max(3, (int(0.012 * W) | 1))
    dil = cv2.dilate(union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, _stats, cents = cv2.connectedComponentsWithStats(dil)

    comps = []
    for cid in range(1, n):
        cmask = lab == cid
        best_cls, best_area = None, 0
        for cls, m in masks.items():
            a = int(np.count_nonzero(m[cmask]))
            if a > best_area:
                best_area, best_cls = a, cls
        if best_cls is None or best_area == 0:
            continue
        conf = float(conf_map[cmask].max()) if cmask.any() else 0.0
        cx, cy = float(cents[cid][0]), float(cents[cid][1])
        comps.append({"id": cid, "cls": best_cls, "conf": conf, "cx": cx, "cy": cy})
    return comps, lab, union


# --------------------------------------------------------------------------- #
# B) Line-tracking state machine (persistent IDs + color hysteresis + priors + phantom filter)
# --------------------------------------------------------------------------- #
class LaneLineTracker:
    def __init__(self, H, W, phantom_min_frames: int):
        self.H, self.W = H, W
        self.gate = ASSOC_GATE_FRAC * np.hypot(H, W)
        self.phantom_min = phantom_min_frames
        self.tracks = {}          # line_id -> dict(cx,cy,votes,age,miss,conf)
        self._next = 0

    def _prior(self, cls, cx):
        if cls == "yellow_solid_lane" and cx < RIGHT_X_FRAC * self.W:
            return YELLOW_CENTER_PRIOR        # center/left yellow penalised (Israeli prior)
        return 1.0

    def update(self, comps):
        """Associate this frame's components to line tracks; update votes/age. Returns
        {component_id: {"line_id", "stable_cls", "readable"}}."""
        # age existing tracks
        for t in self.tracks.values():
            t["miss"] += 1
        used, out = set(), {}
        # greedy nearest-centroid association
        for c in comps:
            best_id, best_d = None, self.gate
            for lid, t in self.tracks.items():
                if lid in used:
                    continue
                d = np.hypot(c["cx"] - t["cx"], c["cy"] - t["cy"])
                if d < best_d:
                    best_d, best_id = d, lid
            if best_id is None:
                lid = self._next
                self._next += 1
                self.tracks[lid] = {"cx": c["cx"], "cy": c["cy"],
                                    "votes": defaultdict(float), "age": 0, "miss": 0, "conf": 0.0}
            else:
                lid = best_id
            used.add(lid)
            t = self.tracks[lid]
            t["cx"], t["cy"], t["miss"], t["conf"] = c["cx"], c["cy"], 0, c["conf"]
            t["age"] += 1
            for cls in t["votes"]:
                t["votes"][cls] *= VOTE_DECAY
            t["votes"][c["cls"]] += c["conf"] * self._prior(c["cls"], c["cx"])
            stable = max(t["votes"], key=t["votes"].get)
            readable = (t["age"] >= self.phantom_min) or (c["conf"] > HIGH_CONF)
            out[c["id"]] = {"line_id": lid, "stable_cls": stable, "readable": readable,
                            "age": t["age"]}
        # reap dead tracks
        self.tracks = {lid: t for lid, t in self.tracks.items() if t["miss"] <= TRACK_MAX_MISS}
        return out


def _fit_line(pix, min_rows=12):
    """Robust 2nd-degree centreline fit x = a y^2 + b y + c for a near-vertical line component.
    Residual-trimmed refit rejects stray points where the mask merges with a neighbour.
    Returns (coef, ymin, ymax) or None if too small / too horizontal."""
    ys, xs = np.where(pix > 0)
    if ys.size < min_rows * 3:
        return None
    ymin, ymax, xmin, xmax = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if (xmax - xmin) > (ymax - ymin):                 # more horizontal than vertical -> bail
        return None
    rows = defaultdict(list)
    for y, x in zip(ys, xs):
        rows[y].append(x)
    ry = np.array(sorted(rows), dtype=np.float64)
    if ry.size < min_rows:
        return None
    rx = np.array([np.mean(rows[int(y)]) for y in ry])
    coef = np.polyfit(ry, rx, 2)
    for _ in range(2):                                # residual-trimmed robust refit
        res = np.abs(rx - np.polyval(coef, ry))
        keep = res < 2.5 * (res.std() + 1e-6)
        if keep.sum() < min_rows:
            break
        coef = np.polyfit(ry[keep], rx[keep], 2)
    return coef, ymin, ymax


def _raster_line(coef, ymin, ymax, H, W, y_vp, w_alpha, w_min):
    """Rasterize a centreline polynomial into a thin ribbon, width shrinking toward the horizon
    W(y) = max(w_min, w_alpha*(y - y_vp))."""
    out = np.zeros((H, W), np.uint8)
    for y in range(max(0, int(ymin)), min(H, int(ymax) + 1)):
        cx = float(np.polyval(coef, y))
        half = 0.5 * max(w_min, w_alpha * (y - y_vp))
        x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
        if x1 > x0:
            out[y, x0:x1] = 255
    return out


def _tight_line_mask(pix, H, W, y_vp, w_alpha, w_min, min_rows=12):
    """Phase 1 tightening: thin distance-adaptive ribbon along the robust centreline. Returns
    None (=> keep original) for components too small/horizontal to fit as a vertical line."""
    f = _fit_line(pix, min_rows)
    if f is None:
        return None
    return _raster_line(f[0], f[1], f[2], H, W, y_vp, w_alpha, w_min)


def surface_from_components(comps, lab, union, info, H, W, island_erode_frac=0.05, tighten=None):
    """Rasterize the violation surface from READABLE components whose STABLE class is a
    violation surface (solid_white_lane or eroded traffic_island). If `tighten` (a dict with
    y_vp / w_alpha / w_min) is given, solid_white_lane components are replaced by a thin
    distance-adaptive centreline ribbon; traffic_island stays an eroded area (NOT a line)."""
    surface = np.zeros((H, W), np.uint8)
    for c in comps:
        meta = info.get(c["id"])
        if not meta or not meta["readable"] or meta["stable_cls"] not in VIOLATION_CLASSES:
            continue
        pix = np.where(lab == c["id"], union, 0).astype(np.uint8)
        if meta["stable_cls"] == "traffic_island":
            er = max(1, int(island_erode_frac * np.sqrt(np.count_nonzero(lab == c["id"]))))
            pix = cv2.erode(pix, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
        elif tighten is not None and meta["stable_cls"] == "solid_white_lane":
            tm = _tight_line_mask(pix, H, W, tighten["y_vp"], tighten["w_alpha"], tighten["w_min"])
            if tm is not None:
                pix = tm
        surface = cv2.bitwise_or(surface, pix)
    return surface


# --------------------------------------------------------------------------- #
# B') PHASE 2 -- per-Line-ID vector memory (the ghost, generalized)
# --------------------------------------------------------------------------- #
class LaneMemory:
    """Remembers each solid line as its fitted CURVE (per LaneLineTracker line-ID) and re-injects
    it into the violation surface ONLY where a vehicle bbox occludes the road -- recovering the
    line under a car that is hiding it. Negative-update gate: a remembered line is HARD-KILLED
    the instant a large fraction of it is visible (not under a vehicle) yet absent from the live
    detection -- so junctions / dashed gaps / lines that truly ended are not hallucinated."""

    def __init__(self, H, W, ttl, y_vp, w_alpha, w_min, veh_dilate=4, kill_frac=0.5,
                 target_cls="solid_white_lane"):
        self.H, self.W, self.ttl0 = H, W, ttl
        self.y_vp, self.w_alpha, self.w_min = y_vp, w_alpha, w_min
        self.veh_dilate, self.kill_frac = veh_dilate, kill_frac
        self.target_cls = target_cls       # which stable_cls to remember
        self.mem = {}                      # line_id -> {coef, ymin, ymax, ttl, age}

    def right_edge_line(self, hood_frac=0.9):
        """Pick the INNERMOST right-side remembered line (the lane/shoulder boundary): among
        lines whose x at the hood row is right of frame-centre, the one with the smallest x.
        Returns {lid, coef, ymin, ymax, age} or None."""
        hood_y = hood_frac * self.H
        best = None
        for lid, m in self.mem.items():
            xl = float(np.polyval(m["coef"], hood_y))
            if xl > 0.5 * self.W and (best is None or xl < best[1]):
                best = ((lid, m), xl)
        if best is None:
            return None
        lid, m = best[0]
        return {"lid": lid, "coef": m["coef"], "ymin": m["ymin"], "ymax": m["ymax"], "age": m.get("age", 0)}

    def _ribbon(self, m):
        return _raster_line(m["coef"], m["ymin"], m["ymax"], self.H, self.W,
                            self.y_vp, self.w_alpha, self.w_min)

    def step(self, comps, info, lab, union, live_surface, vehicles, shift):
        dx, dy = shift
        vmask = np.zeros((self.H, self.W), np.uint8)
        for v in vehicles:
            x1, y1, x2, y2 = v["bbox"]
            cv2.rectangle(vmask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        if self.veh_dilate > 0:
            k = self.veh_dilate * 2 + 1
            vmask = cv2.dilate(vmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

        refreshed = set()
        for c in comps:
            meta = info.get(c["id"])
            if not meta or not meta["readable"] or meta["stable_cls"] != self.target_cls:
                continue
            f = _fit_line(np.where(lab == c["id"], union, 0).astype(np.uint8))
            if f is None:
                continue
            self.mem[meta["line_id"]] = {"coef": f[0], "ymin": f[1], "ymax": f[2],
                                         "ttl": self.ttl0, "age": meta.get("age", 0)}
            refreshed.add(meta["line_id"])

        live_d = cv2.dilate(live_surface, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        not_v, not_d = cv2.bitwise_not(vmask), cv2.bitwise_not(live_d)
        mem_surface = np.zeros((self.H, self.W), np.uint8)
        for lid in list(self.mem):
            m = self.mem[lid]
            if lid not in refreshed:                       # stale -> ego-shift, gate, decay
                m["coef"] = m["coef"].copy(); m["coef"][2] += dx
                m["ymin"] += dy; m["ymax"] += dy
                m["ttl"] -= 1
                if m["ttl"] <= 0 or m["ymax"] <= 0 or m["ymin"] >= self.H or m["ymin"] >= m["ymax"]:
                    del self.mem[lid]; continue
                ribbon = self._ribbon(m)
                area = int(np.count_nonzero(ribbon)) + 1
                vis_bare = cv2.bitwise_and(cv2.bitwise_and(ribbon, not_v), not_d)
                if np.count_nonzero(vis_bare) > self.kill_frac * area:
                    del self.mem[lid]; continue            # negative update: visibly gone
            else:
                ribbon = self._ribbon(m)
            mem_surface = cv2.bitwise_or(mem_surface, cv2.bitwise_and(ribbon, vmask))  # only under vehicles
        return mem_surface


# --------------------------------------------------------------------------- #
# Vehicle contact geometry
# --------------------------------------------------------------------------- #
def trigger_points(bbox):
    x1, _, x2, y2 = bbox
    w = x2 - x1
    return (int(x1 + 0.10 * w), int(y2)), (int(x1 + 0.90 * w), int(y2))


def verdict_segment(bbox):
    x1, _, x2, y2 = bbox
    w = x2 - x1
    return (int(x1 + 0.15 * w), int(y2)), (int(x1 + 0.85 * w), int(y2))


def _at(mask, x, y):
    H, W = mask.shape[:2]
    return mask[min(max(int(y), 0), H - 1), min(max(int(x), 0), W - 1)] > 0


def line_hits(mask, seg, samples: int = 48) -> bool:
    (x0, y), (x1, _) = seg
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    return any(_at(mask, x, y) for x in np.linspace(lo, hi, samples))


def verdict_fracs(mask, bbox, samples: int = 48, lo: float = 0.10, hi: float = 0.90):
    """Sample the bbox-bottom row across [lo, hi] of width and return the fractions where the
    mask is ON. This records the verdict-hit GEOMETRY once so the directional-shift (Idea A)
    and horizon (Idea B) knobs can post-filter cheaply without recomputing the surface."""
    x1, _, x2, y2 = bbox
    w = max(1, x2 - x1)
    return [float(f) for f in np.linspace(lo, hi, samples) if _at(mask, int(x1 + f * w), y2)]


def oncoming_by_side(mask, bbox, ego_x: float,
                     tol_frac: float = SIDE_TOL_FRAC, band: int = SIDE_BAND_PX,
                     margin_frac: float = SIDE_MARGIN_FRAC) -> bool:
    """True if the vehicle sits on the OPPOSITE side of the line from the ego car -> oncoming.

    At the vehicle's contact row we read the line's x-position (mean x of surface pixels in a
    small row band, within the vehicle's x-window). If the vehicle centre and the ego point
    (~frame centre) fall on opposite sides of that line_x -- and the vehicle is clearly off the
    line (> tol) -- the line is BETWEEN us and them, so it's opposing-lane traffic, not a
    same-lane violator. If no line pixels are found we cannot decide -> return False (keep, the
    recall-first default)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = mask.shape[:2]
    w = max(1, x2 - x1)
    m = int(margin_frac * w)
    y_lo, y_hi = max(0, y2 - band), min(H, y2 + band + 1)
    x_lo, x_hi = max(0, x1 - m), min(W, x2 + m)
    sub = mask[y_lo:y_hi, x_lo:x_hi]
    xs = np.where(sub > 0)[1]
    if xs.size == 0:
        return False                       # line not visible near the tires -> can't decide -> keep
    line_x = float(xs.mean()) + x_lo
    veh_rel = 0.5 * (x1 + x2) - line_x     # vehicle centre relative to the line
    ego_rel = ego_x - line_x               # ego (frame centre) relative to the line
    return abs(veh_rel) > tol_frac * w and veh_rel * ego_rel < 0


# --------------------------------------------------------------------------- #
# A) Ghost state machine
# --------------------------------------------------------------------------- #
class GhostMaskTracker:
    def __init__(self, H: int, W: int, ttl: int = 25, island_erode_frac: float = 0.05,
                 side_gate: bool = False):
        self.H, self.W = H, W
        self.ttl0 = ttl
        self.side_gate = side_gate          # suppress opposite-side (oncoming) verdicts
        self.ego_x = 0.5 * W                 # ego reference ~ frame centre (dashcam roughly centred)
        self.ghosts: dict[int, dict] = {}
        self.last_oncoming: dict[int, bool] = {}   # per-track opposite-direction flag from the last step()

    def _capture(self, surface, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = int(0.10 * (x2 - x1))
        rx1, rx2 = max(0, x1 - pad), min(self.W, x2 + pad)
        ry1, ry2 = max(0, y1), min(self.H, y2 + int(0.15 * (y2 - y1)))
        g = np.zeros((self.H, self.W), np.uint8)
        g[ry1:ry2, rx1:rx2] = surface[ry1:ry2, rx1:rx2]
        return g

    def step(self, vehicles, surface, shift, enrich: bool = False,
             record_oncoming: bool = False) -> dict:
        dx, dy = shift
        if self.ghosts:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            for g in self.ghosts.values():
                g["mask"] = cv2.warpAffine(g["mask"], M, (self.W, self.H))
                g["ttl"] -= 1
        hits = {}
        if record_oncoming:
            self.last_oncoming = {}
        for v in vehicles:
            tid, bbox = v["track_id"], v["bbox"]
            tl, tr = trigger_points(bbox)
            verdict = verdict_segment(bbox)
            if tid not in self.ghosts and (_at(surface, *tl) or _at(surface, *tr)):
                self.ghosts[tid] = {"mask": self._capture(surface, bbox), "ttl": self.ttl0}
            if tid in self.ghosts and self.ghosts[tid]["ttl"] > 0:
                jmask = self.ghosts[tid]["mask"]
                hit = line_hits(jmask, verdict)
                if not hit:
                    del self.ghosts[tid]
            else:
                jmask = surface
                hit = line_hits(jmask, verdict)
            # opposite-direction test: the line sits BETWEEN ego and the vehicle (oncoming lane).
            # Computed once, then used to gate (if enabled) and/or recorded for the confidence score.
            onc = (oncoming_by_side(jmask, bbox, self.ego_x)
                   if (self.side_gate or record_oncoming) else False)
            if record_oncoming:
                self.last_oncoming[tid] = onc
            if hit and self.side_gate and onc:
                hit = False
            if enrich:
                # record hit GEOMETRY (intersection fractions, contact-y, centre-x) for cheap
                # post-filtering by Idea A (directional shift) and Idea B (horizon proximity)
                hits[tid] = {"on": hit, "fracs": verdict_fracs(jmask, bbox) if hit else [],
                             "y": int(bbox[3]), "cx": int(0.5 * (bbox[0] + bbox[2]))}
            else:
                hits[tid] = hit
        self.ghosts = {t: g for t, g in self.ghosts.items() if g["ttl"] > 0}
        return hits


# --------------------------------------------------------------------------- #
# Clip driver (compute once) + K sweep (cheap)
# --------------------------------------------------------------------------- #
def compute_verdict_timeline(frames, shifts, H, W, fps, ttl_sec=1.0,
                             island_erode_frac=0.05, phantom_min_sec=1.0, side_gate=False,
                             return_oncoming=False):
    """Run line-tracking -> surface -> ghost/verdict over a clip. Returns
    {track_id: {frame_index: verdict_hit_bool}} (independent of K). With return_oncoming=True
    also returns {track_id: {frame_index: oncoming_bool}} (line is between ego and the vehicle),
    so the confidence score can down-weight opposite-direction traffic in the SAME single pass."""
    ttl = max(1, round(ttl_sec * fps))
    phantom = max(0, round(phantom_min_sec * fps))   # 0 => gate disabled (readable on first sight)
    lt = LaneLineTracker(H, W, phantom)
    gt = GhostMaskTracker(H, W, ttl, island_erode_frac, side_gate=side_gate)
    timeline = defaultdict(dict)
    oncoming = defaultdict(dict)
    for i, fr in enumerate(frames):
        comps, lab, union = reconcile_components(fr["lanes"], H, W)
        info = lt.update(comps)
        surface = surface_from_components(comps, lab, union, info, H, W, island_erode_frac)
        shift = tuple(shifts[i]) if i < len(shifts) and shifts[i] else (0.0, 0.0)
        hits = gt.step(fr["vehicles"], surface, shift, record_oncoming=return_oncoming)
        for tid, hit in hits.items():
            timeline[tid][fr["frame"]] = hit
        if return_oncoming:
            for tid, onc in gt.last_oncoming.items():
                oncoming[tid][fr["frame"]] = onc
    return (timeline, oncoming) if return_oncoming else timeline


def compute_enriched_timeline(frames, shifts, H, W, fps, ttl_sec=1.0,
                              island_erode_frac=0.05, phantom_min_sec=1.0, tighten=None,
                              memory=False, mem_ttl_sec=1.0):
    """Like compute_verdict_timeline but records hit GEOMETRY per (track, frame):
    {track_id: {frame: {"on": bool, "fracs": [..], "y": int, "cx": int}}}. Computed ONCE;
    Idea A / Idea B / dynamic-K then post-filter this cheaply (no surface recompute).
    `tighten` (dict y_vp/w_alpha/w_min) enables Phase-1 solid-line tightening; `memory` adds the
    Phase-2 per-Line-ID vector memory (requires tighten for the width params)."""
    ttl = max(1, round(ttl_sec * fps))
    phantom = max(0, round(phantom_min_sec * fps))
    lt = LaneLineTracker(H, W, phantom)
    gt = GhostMaskTracker(H, W, ttl, island_erode_frac, side_gate=False)
    lm = None
    if memory:
        assert tighten is not None, "memory requires tighten (width params)"
        lm = LaneMemory(H, W, max(1, round(mem_ttl_sec * fps)),
                        tighten["y_vp"], tighten["w_alpha"], tighten["w_min"])
    timeline = defaultdict(dict)
    for i, fr in enumerate(frames):
        comps, lab, union = reconcile_components(fr["lanes"], H, W)
        info = lt.update(comps)
        surface = surface_from_components(comps, lab, union, info, H, W, island_erode_frac, tighten)
        shift = tuple(shifts[i]) if i < len(shifts) and shifts[i] else (0.0, 0.0)
        if lm is not None:
            msurf = lm.step(comps, info, lab, union, surface, fr["vehicles"], shift)
            surface = cv2.bitwise_or(surface, msurf)
        for tid, rec in gt.step(fr["vehicles"], surface, shift, enrich=True).items():
            timeline[tid][fr["frame"]] = rec
    return timeline


def events_from_timeline(timeline, k_consec: int):
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
            if end - start + 1 >= k_consec:
                events.append((tid, start, end))
            i = j
    return events
