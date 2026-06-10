"""
WRONG-WAY / oncoming-in-OUR-lane driver (additive; touches no existing violation logic).

A legitimate oncoming car is in the LEFT/opposing lane (usually with a divider line between us
and it). A WRONG-WAY driver is approaching INSIDE OUR OWN lane corridor. So per (vehicle, frame):

  oncoming P > P_HIGH                  (clearly approaching us; reuse MotionDirectionFilter)
  AND |veh_cx - ego_x| < CORRIDOR*W    (in OUR lane band, not the left opposing lane)  [primary]
  AND contact_y > MID*H                (close enough / actually in our path)
  AND NO solid/yellow divider line lies between ego and the vehicle                    [extra guard]

Then K consecutive frames (~1 s sustained approach) + per-track cooldown + evidence clips.

Standalone: imports shared primitives only; no edits to ghost_mask/motion_filter behaviour, so
the white-line / shoulder pipelines are unaffected. No labeled wrong-way clips yet -> thresholds
are conservative placeholders; this proves the logic runs and doesn't false-fire on clean clips.

  <gpu-python> tools/wrong_way.py [clip_prefix_to_render]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossing_violation_test as cv  # noqa: E402
from ghost_mask import LaneLineTracker, _fit_line, events_from_timeline, reconcile_components  # noqa: E402
from motion_filter import MotionDirectionFilter, merge_events  # noqa: E402

MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
P_HIGH = 0.5            # oncoming probability above which the vehicle is "approaching us"
CORRIDOR_FRAC = 0.18    # half-width of our lane band around ego centre (fraction of frame width)
MID_FRAC = 0.50         # vehicle contact must be in the lower half (close / in path)
DIVIDERS = ("solid_white_lane", "yellow_solid_lane")
K_SEC = 1.0             # sustained approach required
COOLDOWN = 3.0
HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")


class WrongWayEvaluator:
    def __init__(self, H, W, fps):
        self.H, self.W, self.fps = H, W, fps
        self.ego_x = 0.5 * W
        self.lt = LaneLineTracker(H, W, 0)
        self.mf = MotionDirectionFilter(H, **MF)

    def step(self, fr, shift):
        comps, lab, union = reconcile_components(fr["lanes"], self.H, self.W)
        info = self.lt.update(comps)
        dividers = []
        for c in comps:
            m = info.get(c["id"])
            if m and m["readable"] and m["stable_cls"] in DIVIDERS:
                f = _fit_line(np.where(lab == c["id"], union, 0).astype(np.uint8))
                if f is not None:
                    dividers.append(f[0])
        self.mf.update(fr["frame"], fr["vehicles"], shift[1])
        res = {}
        for v in fr["vehicles"]:
            tid = v["track_id"]; x1, y1, x2, y2 = v["bbox"]; cx = 0.5 * (x1 + x2)
            P = self.mf.oncoming_score(tid)
            in_corr = abs(cx - self.ego_x) < CORRIDOR_FRAC * self.W
            close = y2 > MID_FRAC * self.H
            div_between = False
            for coef in dividers:
                lx = float(np.polyval(coef, y2))
                if (lx - self.ego_x) * (lx - cx) < 0:        # line strictly between ego and vehicle
                    div_between = True
                    break
            cand = (P > P_HIGH) and in_corr and close and not div_between
            res[tid] = {"cand": cand, "P": P, "in_corr": in_corr, "div": div_between}
        return dividers, res


def clip_timeline(c):
    ev = WrongWayEvaluator(c["h"], c["w"], c["fps"])
    tl = defaultdict(dict)
    for fr in c["frames"]:
        _, res = ev.step(fr, tuple(fr.get("shift", [0.0, 0.0])))
        for tid, r in res.items():
            tl[tid][fr["frame"]] = r["cand"]
    return tl


def render(c, tag="wrongway"):
    fps, W, h, prefix = c["fps"], c["w"], c["h"], c["prefix"]
    events = merge_events(events_from_timeline(clip_timeline(c), max(1, round(K_SEC * fps))), fps, COOLDOWN)
    active = defaultdict(set)
    for tid, s, e in events:
        for f in range(s, e + 1):
            active[f].add(tid)
    ev = WrongWayEvaluator(h, W, fps)
    fr_by = {f["frame"]: f for f in c["frames"]}
    cap = cv2.VideoCapture(c["path"]); pw, ph = 1280, int(1280 * h / W)
    os.makedirs(cv.OUT_DIR, exist_ok=True)
    out = os.path.join(cv.OUT_DIR, f"{prefix}_{tag}.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph)); fi = 0
    ex = int(0.5 * W); band = int(CORRIDOR_FRAC * W)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rec = fr_by.get(fi)
        if rec is not None:
            _, res = ev.step(rec, tuple(rec.get("shift", [0.0, 0.0])))
            cv2.line(frame, (ex - band, 0), (ex - band, h), (120, 120, 120), 1)   # our-lane corridor
            cv2.line(frame, (ex + band, 0), (ex + band, h), (120, 120, 120), 1)
            firing = active.get(fi, set())
            for v in rec["vehicles"]:
                x1, y1, x2, y2 = v["bbox"]; tid = v["track_id"]; r = res.get(tid, {})
                hot = tid in firing
                col = (0, 0, 255) if hot else ((0, 165, 255) if r.get("in_corr") else (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if hot else 2)
                cv2.putText(frame, f"P={r.get('P',0):.2f}{' DIV' if r.get('div') else ''}",
                            (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
                if hot:
                    cv2.putText(frame, f"WRONG-WAY #{tid}", (x1, max(y1 - 26, 28)),
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
    smoke = ["S9kQqWl05KU", "DiU52hnxSVY", "I3AcpHTrEXI", "yI0uJzyS-mQ", "B7-3EuQFAZM", "Rj2yWbv1934"]
    print(f"[wrong-way] P>{P_HIGH} corridor=+/-{CORRIDOR_FRAC}W mid={MID_FRAC} K={K_SEC}s")
    print(f"{'clip':<16}{'frames':>7}{'rawEv':>7}{'incidents':>10}")
    best = None
    for prefix in smoke:
        c = cv.load_cache(prefix)
        if c is None:
            print(f"{prefix:<16} no cache"); continue
        c = cv.ensure_shifts(c)
        tl = clip_timeline(c)
        kf = max(1, round(K_SEC * c["fps"]))
        raw = events_from_timeline(tl, kf)
        inc = merge_events(raw, c["fps"], COOLDOWN)
        print(f"{prefix:<16}{c['total']:>7}{len(raw):>7}{len(inc):>10}")
        if best is None or len(inc) > best[1]:
            best = (prefix, len(inc))
    target = sys.argv[1] if len(sys.argv) > 1 else (best[0] if best else None)
    if target:
        print(f"\n[render] {target} (corridor band + P + divider flag)")
        render(cv.ensure_shifts(cv.load_cache(target)))
    print("[done] additive logic runs; validate/tune once a wrong-way clip exists.")


if __name__ == "__main__":
    main()
