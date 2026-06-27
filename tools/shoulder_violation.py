"""
NEW VIOLATION (Israel-specific): a vehicle driving ON THE SHOULDER -- clearly to the RIGHT of
the solid yellow right-edge line, MOVING, and travelling in OUR direction.

Reuses the white-line stack: reconcile + LaneLineTracker (line IDs + age), _fit_line/_raster_line
(polynomial centreline), LaneMemory (occlusion persistence -- the car covers the yellow line, so
we evaluate against the REMEMBERED polynomial), MotionDirectionFilter (oncoming P), and the
K-consecutive / cooldown / evidence machinery.

Per-frame violation CANDIDATE for (vehicle, frame):
  right-edge yellow line valid >= 3 s                 (LaneLineTracker age, frozen in memory)
  AND vehicle CENTRE > line_x(y_contact) + k*bbox_w   (width-scaled, depth-adaptive side test)
  AND world_speed > MOVING_THRESHOLD_KMH              (T1: real speed gate, not "loom")
  AND bbox area >= AREA_MIN                            (near/mid field only -- far cars are noise)
  AND oncoming P < threshold                          (not the opposing carriageway)
Then K = K_SEC consecutive (excludes a legal pull-over-to-park) + per-track cooldown.

THE YELLOW LINE CONFIDENCE SCORE (T3), computed at the KEY (peak-crossing) frame of each
incident:  confidence = clamp( shoulder_overlap_frac * seg_conf , 0, 1 ), where
  shoulder_overlap_frac = (bbox area lying RIGHT of the remembered line) / (total bbox area)
  seg_conf              = the yellow line's lane-segmentation confidence (carried through
                          LaneMemory occlusion via the per-line-ID conf cache below).
Each incident is emitted as one violations.event.ViolationEvent(YELLOW_LINE_RIGHT), the same
record speeding + solid-line-crossing emit, so the LPR/evidence stage is identical for all rules.

SPEED SOURCE:  the REAL gate consumes a ``speed_lookup(track_id, frame) -> km/h`` callable
(wrap a World's ``vehicle.speed_per_frame``, same source as speed_estimation/overspeed.py). The
offline cache here has no world speed, so when no lookup is injected we fall back to an
ego-compensated contact-point motion PROXY (image_motion - ego_shift ~ world-velocity
projection) -- enough to separate parked from moving so the harness runs end-to-end. The proxy
scale is uncalibrated and is replaced wholesale the moment the World estimator is wired in.

  <gpu-python> tools/shoulder_violation.py [clip_prefix_to_render]
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossing_violation_test as cv  # noqa: E402
from ghost_mask import (LaneLineTracker, LaneMemory, _fit_line, _raster_line,  # noqa: E402
                        events_from_timeline, reconcile_components)
from motion_filter import MotionDirectionFilter, merge_events  # noqa: E402

sys.path.insert(0, cv.REPO)                                    # repo root, for the violations package
from violations.event import ViolationEvent, ViolationType  # noqa: E402

MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
TVP_FRAC, W_ALPHA, W_MIN = 0.45, 0.037, 5.0
K_MARGIN = 0.15        # side-test buffer as a fraction of bbox width (depth-adaptive)
P_THRESH = 0.5         # oncoming suppression
MIN_AGE_SEC = 3.0      # yellow line must persist this long
K_SEC = 3.0            # continuous shoulder-driving required (excludes legal pull-over)
COOLDOWN = 3.0
MOVING_THRESHOLD_KMH = 8.0    # T1: above the speed-estimator jitter floor; below it a car is "stopped"
AREA_MIN = 15_000             # only adjudicate cars close enough to judge (matches the evidence gate)
SPEED_WIN = 10                # frames of contact-point motion smoothed for the proxy speed
PROXY_KMH_PER_PXPS = 0.30     # UNCALIBRATED placeholder px/s -> km/h for the offline proxy only
YELLOW = "yellow_solid_lane"
HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")


def shoulder_overlap_frac(bbox, coef, step: int = 2) -> float:
    """Fraction of the bbox area lying to the RIGHT of the remembered line x=coef(y).

    Integrates row by row: at each scanline y the share of the bbox width right of the line is
    (x2 - clamp(line_x, x1, x2)) / w, averaged over the bbox height. Faithful to "how much of the
    car is over the line" and gives a spread-out, rankable score (the thin-line-pixel fraction
    would be near-zero). Computed against the REMEMBERED polynomial so the car occluding the live
    line does not zero it out.
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    rows = range(int(y1), int(y2) + 1, max(1, step))
    n = 0
    acc = 0.0
    for y in rows:
        xline = float(np.polyval(coef, y))
        right_w = max(0.0, min(w, x2 - max(x1, xline)))
        acc += right_w / w
        n += 1
    return acc / n if n else 0.0


class ShoulderEvaluator:
    """Stateful per-frame evaluator (so the timeline build and the renderer share one path)."""

    def __init__(self, H, W, fps, speed_lookup=None):
        self.H, self.W, self.fps = H, W, fps
        self.yvp, self.wa, self.wm = TVP_FRAC * H, W_ALPHA, W_MIN
        self.lt = LaneLineTracker(H, W, 0)
        self.mem = LaneMemory(H, W, max(1, round(1.0 * fps)), self.yvp, self.wa, self.wm,
                              target_cls=YELLOW)
        self.mf = MotionDirectionFilter(H, **MF)
        self.speed_lookup = speed_lookup           # (tid, frame) -> km/h  (None => proxy fallback)
        self.line_conf = {}                        # line_id -> last seen yellow seg conf (carried thru occlusion)
        self.resid = defaultdict(lambda: deque(maxlen=SPEED_WIN))   # tid -> ego-compensated contact motion px
        self.prev_contact = {}                     # tid -> (frame, cx, y_bottom)
        self.min_age = round(MIN_AGE_SEC * fps)

    def _speed_kmh(self, tid, frame, cx, ybot, dx, dy):
        """World speed for the moving gate. Uses the injected estimator if present; otherwise an
        ego-compensated contact-point motion proxy (image_motion - ego_shift ~ world velocity)."""
        if self.speed_lookup is not None:
            s = self.speed_lookup(tid, frame)
            return float(s) if s else 0.0
        prev = self.prev_contact.get(tid)
        self.prev_contact[tid] = (frame, cx, ybot)
        if prev is not None and prev[0] == frame - 1:
            self.resid[tid].append(math.hypot(cx - prev[1] - dx, ybot - prev[2] - dy))
        if not self.resid[tid]:
            return 0.0
        return float(np.median(self.resid[tid])) * self.fps * PROXY_KMH_PER_PXPS

    def step(self, fr, shift):
        frame = fr["frame"]
        dx, dy = shift
        comps, lab, union = reconcile_components(fr["lanes"], self.H, self.W)
        info = self.lt.update(comps)
        ysurf = np.zeros((self.H, self.W), np.uint8)           # live yellow surface for the memory gate
        n_yellow = 0
        for c in comps:
            m = info.get(c["id"])
            if m and m["readable"] and m["stable_cls"] == YELLOW:
                n_yellow += 1
                self.line_conf[m["line_id"]] = float(c["conf"])    # remember seg conf for the confidence score
                f = _fit_line(np.where(lab == c["id"], union, 0).astype(np.uint8))
                if f:
                    ysurf = cv2.bitwise_or(ysurf, _raster_line(f[0], f[1], f[2], self.H, self.W,
                                                               self.yvp, self.wa, self.wm))
        self.mem.step(comps, info, lab, union, ysurf, fr["vehicles"], shift)
        self.mf.update(fr["frame"], fr["vehicles"], shift[1])
        rel = self.mem.right_edge_line()
        seg_conf = self.line_conf.get(rel["lid"], 0.0) if rel is not None else 0.0
        res = {}
        for v in fr["vehicles"]:
            tid = v["track_id"]; x1, y1, x2, y2 = v["bbox"]
            cx = 0.5 * (x1 + x2); w = max(1, x2 - x1); area = w * max(1, y2 - y1)
            speed = self._speed_kmh(tid, frame, cx, y2, dx, dy)
            P = self.mf.oncoming_score(tid)
            xl = dist = None
            right = cand = False
            frac = conf = 0.0
            age = rel["age"] if rel is not None else 0
            if rel is not None and age >= self.min_age:
                xl = float(np.polyval(rel["coef"], y2))
                dist = cx - (xl + K_MARGIN * w)                # signed depth past the side-test line
                right = dist > 0
                if right:
                    frac = shoulder_overlap_frac((x1, y1, x2, y2), rel["coef"])
                    conf = max(0.0, min(1.0, frac * seg_conf))
                cand = right and (speed > MOVING_THRESHOLD_KMH) and (area >= AREA_MIN) and (P < P_THRESH)
            res[tid] = {"cand": cand, "right": right, "P": P, "speed": speed, "xl": xl, "dist": dist,
                        "frac": frac, "seg_conf": seg_conf, "conf": conf, "age": age, "w": w,
                        "bbox": (x1, y1, x2, y2)}
        return rel, res, ysurf, n_yellow


def evaluate(c, speed_lookup=None, once_per_vehicle=False):
    """Run the evaluator over a cached clip. Returns (incidents, recs, events, yellow_frames):
      incidents  -- merged (tid, start, end) per-track incidents (K-consec + cooldown)
      recs       -- recs[tid][frame] = the per-frame res dict (for the renderer / key-frame pick)
      events     -- one ViolationEvent per incident, scored at its peak-crossing key frame

    once_per_vehicle: shoulder-driving is reported STRICTLY ONCE per vehicle for the whole clip
      (Tal's baseline rule -- a car flagged on the shoulder is not re-reported). When True, only
      the EARLIEST incident per track id survives. The default (False) keeps the legacy 3 s-cooldown
      behaviour so the standalone sweep/render harness is unchanged.
    """
    fps = c["fps"]
    ev = ShoulderEvaluator(c["h"], c["w"], fps, speed_lookup)
    tl = defaultdict(dict)
    recs = defaultdict(dict)
    yellow_frames = 0
    for fr in c["frames"]:
        shift = tuple(fr.get("shift", [0.0, 0.0]))
        _, res, _, n_yellow = ev.step(fr, shift)
        yellow_frames += 1 if n_yellow else 0
        for tid, r in res.items():
            tl[tid][fr["frame"]] = r["cand"]
            recs[tid][fr["frame"]] = r

    kf = max(1, round(K_SEC * fps))
    incidents = merge_events(events_from_timeline(tl, kf), fps, COOLDOWN)

    if once_per_vehicle:
        # keep only the earliest incident per vehicle -> one shoulder report per car, ever
        earliest: dict[int, tuple] = {}
        for tid, s, e in incidents:
            if tid not in earliest or s < earliest[tid][1]:
                earliest[tid] = (tid, s, e)
        incidents = sorted(earliest.values(), key=lambda x: (x[0], x[1]))

    events = []
    for tid, s, e in incidents:
        # Key frame = peak crossing (max signed distance past the line) inside the incident window,
        # analogous to OverspeedEvent's peak-exceedance frame. Score the event ONCE, here.
        key = max(range(s, e + 1),
                  key=lambda f: (recs[tid].get(f, {}).get("dist") if recs[tid].get(f, {}).get("dist") is not None
                                 else -1e9))
        r = recs[tid][key]
        events.append(ViolationEvent(
            vehicle_id=tid,
            violation_type=ViolationType.YELLOW_LINE_RIGHT,
            key_frame=key,
            confidence=round(r["conf"], 3),
            details={
                "shoulder_overlap_frac": round(r["frac"], 3),
                "seg_conf": round(r["seg_conf"], 3),
                "margin_px": round(K_MARGIN * r["w"], 1),
                "est_speed_kmh": round(r["speed"], 1),
                "start_frame": s,
                "end_frame": e,
                "n_frames_over": e - s + 1,
                "oncoming_p": round(r["P"], 3),
                "line_age_s": round(r["age"] / fps, 2),
            },
        ))
    return incidents, recs, events, yellow_frames


def render(c, tag="shoulder", speed_lookup=None):
    """Draw, AT THE ORIGINAL VIDEO RESOLUTION:
        - the live yellow lane mask (translucent overlay) + the remembered right-edge line,
        - each vehicle's bbox (RED while it is a firing violator, amber if merely right-of-line),
        - a per-vehicle text tag with its speed (km/h) and overlap-confidence (%).
    """
    fps, W, h, prefix = c["fps"], c["w"], c["h"], c["prefix"]
    incidents, _, events, _ = evaluate(c, speed_lookup)
    active = defaultdict(set)
    for tid, s, e in incidents:
        for f in range(s, e + 1):
            active[f].add(tid)
    key_frames = {(e.vehicle_id, e.key_frame): e for e in events}

    ev = ShoulderEvaluator(h, W, fps, speed_lookup)            # fresh state for the render pass
    fr_by = {f["frame"]: f for f in c["frames"]}
    cap = cv2.VideoCapture(c["path"])
    out = os.path.join(cv.OUT_DIR, f"{prefix}_{tag}.mp4")
    os.makedirs(cv.OUT_DIR, exist_ok=True)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, h))   # EXACT input resolution
    proxy = speed_lookup is None
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rec = fr_by.get(fi)
        if rec is not None:
            rel, res, ysurf, _ = ev.step(rec, tuple(rec.get("shift", [0.0, 0.0])))
            if ysurf is not None and ysurf.any():              # translucent yellow MASK overlay
                tint = frame.copy()
                tint[ysurf > 0] = (0, 255, 255)
                cv2.addWeighted(tint, 0.35, frame, 0.65, 0, frame)
            if rel is not None:                                # the remembered right-edge decision line
                pts = [(int(np.polyval(rel["coef"], y)), y)
                       for y in range(max(0, int(rel["ymin"])), min(h, int(rel["ymax"])), 6)]
                col = (0, 255, 255) if rel["age"] >= round(MIN_AGE_SEC * fps) else (0, 140, 140)
                for p, q in zip(pts, pts[1:]):
                    cv2.line(frame, p, q, col, 3)
            firing = active.get(fi, set())
            for v in rec["vehicles"]:
                x1, y1, x2, y2 = v["bbox"]; tid = v["track_id"]; r = res.get(tid, {})
                hot = tid in firing
                col = (0, 0, 255) if hot else ((0, 165, 255) if r.get("right") else (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if hot else 2)
                cv2.putText(frame, f"v={r.get('speed', 0):.0f}km/h  ov={r.get('conf', 0) * 100:.0f}%",
                            (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                if hot:
                    label = f"SHOULDER #{tid}"
                    if (tid, fi) in key_frames:
                        label += "  <KEY>"
                    cv2.putText(frame, label, (x1, max(y1 - 26, 28)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if firing:
                cv2.rectangle(frame, (0, 0), (W - 1, h - 1), (0, 0, 255), 14)
        hud = f"{prefix} [{tag}] f{fi}" + ("  speed=PROXY" if proxy else "")
        cv2.putText(frame, hud, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        wr.write(frame); fi += 1
    cap.release(); wr.release()
    print(f"   video -> {out}  ({W}x{h})")


def main():
    with open(cv.LABELS_JSON) as fh:
        json.load(fh)                              # (kept: ensures the label file is present/valid)
    # smoke on the SMALL clips (fast) — proves it runs + checks it doesn't spuriously fire
    smoke = ["S9kQqWl05KU", "DiU52hnxSVY", "I3AcpHTrEXI", "yI0uJzyS-mQ", "B7-3EuQFAZM"]
    print(f"[shoulder] k_margin={K_MARGIN} move>{MOVING_THRESHOLD_KMH}km/h area>={AREA_MIN} "
          f"P<{P_THRESH} age>={MIN_AGE_SEC}s K={K_SEC}s  (speed source: injected World estimator "
          f"or offline PROXY)")
    print(f"{'clip':<16}{'frames':>7}{'yellowF':>9}{'incidents':>10}{'events':>7}")
    best = None
    for prefix in smoke:
        c = cv.load_cache(prefix)
        if c is None:
            print(f"{prefix:<16} no cache"); continue
        c = cv.ensure_shifts(c)
        incidents, _, events, yf = evaluate(c)
        print(f"{prefix:<16}{c['total']:>7}{yf:>9}{len(incidents):>10}{len(events):>7}")
        for e in events:
            print(f"    -> {e.violation_type} veh#{e.vehicle_id} key=f{e.key_frame} "
                  f"conf={e.confidence:.2f} {e.details}")
        if best is None or yf > best[1]:
            best = (prefix, yf)

    render_target = sys.argv[1] if len(sys.argv) > 1 else (best[0] if best else None)
    if render_target:
        print(f"\n[render] {render_target} (yellow mask + side test + moving/oncoming gates)")
        render(cv.ensure_shifts(cv.load_cache(render_target)))
    print("[done] logic runs; inject a real speed_lookup + validate/tune on real shoulder clips.")


if __name__ == "__main__":
    main()
