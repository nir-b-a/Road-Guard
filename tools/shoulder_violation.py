"""
NEW VIOLATION (Israel-specific): a vehicle driving ON THE SHOULDER -- clearly to the RIGHT of
the solid yellow right-edge line, MOVING, and travelling in OUR direction.

Reuses the white-line stack: reconcile + LaneLineTracker (line IDs + age), _fit_line/_raster_line
(polynomial centreline), LaneMemory (occlusion persistence -- the car covers the yellow line, so
we evaluate against the REMEMBERED polynomial), MotionDirectionFilter (oncoming P), and the
K-consecutive / cooldown / evidence machinery.

Decision per (vehicle, frame):
  right-edge yellow line valid >= 3 s              (LaneLineTracker age, frozen in memory)
  AND vehicle CENTRE > line_x(y_contact) + k*bbox_w   (width-scaled depth-adaptive margin)
  AND NOT violently looming (=> not a parked car we're approaching)   [interim moving gate]
  AND oncoming P < threshold                       (not opposing carriageway)
Then K = 3 s consecutive (excludes a legal pull-over-to-park) + 3 s per-track cooldown.

NOTE: no labeled shoulder/parked clips yet -> thresholds are placeholders; the moving gate's
real form is the sibling speed estimator at merge. This proves the logic runs and is ready.

  <gpu-python> tools/shoulder_violation.py [clip_prefix_to_render]
"""
from __future__ import annotations

import glob
import json
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

MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
TVP_FRAC, W_ALPHA, W_MIN = 0.45, 0.037, 5.0
K_MARGIN = 0.15        # side-test buffer as a fraction of bbox width (depth-adaptive)
LOOM_HIGH = 0.6        # bbox-height growth over the window above which we call it parked/stationary
P_THRESH = 0.5         # oncoming suppression
MIN_AGE_SEC = 3.0      # yellow line must persist this long
K_SEC = 3.0            # continuous shoulder-driving required (excludes legal pull-over)
COOLDOWN = 3.0
HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")
YELLOW = "yellow_solid_lane"


class ShoulderEvaluator:
    """Stateful per-frame evaluator (so the timeline build and the renderer share one path)."""

    def __init__(self, H, W, fps):
        self.H, self.W, self.fps = H, W, fps
        self.yvp, self.wa, self.wm = TVP_FRAC * H, W_ALPHA, W_MIN
        self.lt = LaneLineTracker(H, W, 0)
        self.mem = LaneMemory(H, W, max(1, round(1.0 * fps)), self.yvp, self.wa, self.wm,
                              target_cls=YELLOW)
        self.mf = MotionDirectionFilter(H, **MF)
        self.hist = defaultdict(lambda: deque(maxlen=MF["window"]))
        self.min_age = round(MIN_AGE_SEC * fps)

    def _loom(self, tid):
        h = self.hist[tid]
        return 0.0 if len(h) < 2 else (h[-1] - h[0]) / (h[0] + 1e-6)

    def step(self, fr, shift):
        comps, lab, union = reconcile_components(fr["lanes"], self.H, self.W)
        info = self.lt.update(comps)
        ysurf = np.zeros((self.H, self.W), np.uint8)           # live yellow surface for the memory gate
        n_yellow = 0
        for c in comps:
            m = info.get(c["id"])
            if m and m["readable"] and m["stable_cls"] == YELLOW:
                n_yellow += 1
                f = _fit_line(np.where(lab == c["id"], union, 0).astype(np.uint8))
                if f:
                    ysurf = cv2.bitwise_or(ysurf, _raster_line(f[0], f[1], f[2], self.H, self.W,
                                                               self.yvp, self.wa, self.wm))
        self.mem.step(comps, info, lab, union, ysurf, fr["vehicles"], shift)
        self.mf.update(fr["frame"], fr["vehicles"], shift[1])
        rel = self.mem.right_edge_line()
        res = {}
        for v in fr["vehicles"]:
            tid = v["track_id"]; x1, y1, x2, y2 = v["bbox"]; cx = 0.5 * (x1 + x2); w = max(1, x2 - x1)
            self.hist[tid].append(float(y2 - y1))
            loom, P = self._loom(tid), self.mf.oncoming_score(tid)
            xl, right, cand = None, False, False
            if rel is not None and rel["age"] >= self.min_age:
                xl = float(np.polyval(rel["coef"], y2))
                right = cx > xl + K_MARGIN * w
                cand = right and (loom < LOOM_HIGH) and (P < P_THRESH)
            res[tid] = {"cand": cand, "right": right, "P": P, "loom": loom, "xl": xl}
        return rel, res, n_yellow


def clip_timeline(c):
    ev = ShoulderEvaluator(c["h"], c["w"], c["fps"])
    tl = defaultdict(dict)
    yellow_frames = 0
    for fr in c["frames"]:
        shift = tuple(fr.get("shift", [0.0, 0.0]))
        _, res, n_yellow = ev.step(fr, shift)
        yellow_frames += 1 if n_yellow else 0
        for tid, r in res.items():
            tl[tid][fr["frame"]] = r["cand"]
    return tl, yellow_frames


def render(c, tag="shoulder"):
    fps, W, h, prefix = c["fps"], c["w"], c["h"], c["prefix"]
    ev = ShoulderEvaluator(h, W, fps)
    # firing frames (K=3s consecutive + cooldown) computed from a first pass
    tl, _ = clip_timeline(c)
    events = merge_events(events_from_timeline(tl, max(1, round(K_SEC * fps))), fps, COOLDOWN)
    active = defaultdict(set)
    for tid, s, e in events:
        for f in range(s, e + 1):
            active[f].add(tid)
    ev = ShoulderEvaluator(h, W, fps)                          # fresh state for the render pass
    fr_by = {f["frame"]: f for f in c["frames"]}
    cap = cv2.VideoCapture(c["path"]); pw, ph = 1280, int(1280 * h / W)
    out = os.path.join(cv.OUT_DIR, f"{prefix}_{tag}.mp4")
    os.makedirs(cv.OUT_DIR, exist_ok=True)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph)); fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rec = fr_by.get(fi)
        if rec is not None:
            rel, res, _ = ev.step(rec, tuple(rec.get("shift", [0.0, 0.0])))
            if rel is not None:                                # draw the right-edge yellow line
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
                cv2.putText(frame, f"P={r.get('P',0):.2f} loom={r.get('loom',0):+.2f}",
                            (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                if hot:
                    cv2.putText(frame, f"SHOULDER #{tid}", (x1, max(y1 - 26, 28)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if firing:
                cv2.rectangle(frame, (0, 0), (W - 1, h - 1), (0, 0, 255), 14)
        cv2.putText(frame, f"{prefix} [{tag}] f{fi}", (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        wr.write(cv2.resize(frame, (pw, ph))); fi += 1
    cap.release(); wr.release()
    print(f"   video -> {out}")


def main():
    with open(cv.LABELS_JSON) as fh:
        viol = list(json.load(fh)["clips"])
    hwy = [os.path.splitext(os.path.basename(f))[0]
           for f in sorted(glob.glob(os.path.join(HIGHWAY_DIR, "*.mp4")))
           if "_annotated" not in f and "_winner" not in f]
    # smoke on the SMALL clips (fast) — proves it runs + checks it doesn't spuriously fire
    smoke = ["S9kQqWl05KU", "DiU52hnxSVY", "I3AcpHTrEXI", "yI0uJzyS-mQ", "B7-3EuQFAZM"]
    print(f"[shoulder] thresholds: k_margin={K_MARGIN} loom_high={LOOM_HIGH} P<{P_THRESH} "
          f"age>={MIN_AGE_SEC}s K={K_SEC}s")
    print(f"{'clip':<16}{'frames':>7}{'yellowF':>9}{'rawEv':>7}{'incidents':>10}")
    best = None
    for prefix in smoke:
        c = cv.load_cache(prefix)
        if c is None:
            print(f"{prefix:<16} no cache"); continue
        c = cv.ensure_shifts(c)
        tl, yf = clip_timeline(c)
        kf = max(1, round(K_SEC * c["fps"]))
        raw = events_from_timeline(tl, kf)
        inc = merge_events(raw, c["fps"], COOLDOWN)
        print(f"{prefix:<16}{c['total']:>7}{yf:>9}{len(raw):>7}{len(inc):>10}")
        if best is None or yf > best[1]:
            best = (prefix, yf)
    render_target = sys.argv[1] if len(sys.argv) > 1 else (best[0] if best else None)
    if render_target:
        print(f"\n[render] {render_target} (yellow line + side test + moving/oncoming gates)")
        c = cv.ensure_shifts(cv.load_cache(render_target))
        render(c)
    print("[done] logic runs; validate/tune once real shoulder + parked-car clips exist.")


if __name__ == "__main__":
    main()
