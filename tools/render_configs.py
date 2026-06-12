"""
Render overlay videos for specific FP-reduction configs so the mechanism is VISIBLE.

For each clip we compute the enriched timeline + motion scores once, derive events via
events_configured for the chosen config, then render: lanes (stabilized class), red-tinted
active ghost masks, amber trigger dots, the (possibly shifted) GREEN verdict segment in its
ACTUAL position, each vehicle's oncoming P, and the red firing box + VIOLATION banner.

  <gpu-python> tools/render_configs.py
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
from ghost_mask import (GhostMaskTracker, LaneLineTracker, compute_enriched_timeline,  # noqa: E402
                        reconcile_components, surface_from_components, trigger_points)
from motion_filter import compute_motion_scores, events_configured  # noqa: E402

HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")
MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
KBASE = 0.10

# (tag, knobs, which set: "viol" | "hwy" | "both")
CONFIGS = [
    ("fullstack", dict(far_bias=0.75, alpha=5.0, horizon_gain=0.0), "both"),
    ("ideaA",     dict(far_bias=1.0,  alpha=0.0, horizon_gain=0.0), "hwy"),
]


def eff_window(cx, W, P, far_bias):
    """Accept interval (lo,hi) as fractions of bbox width -- the shifted verdict segment."""
    e = min(1.0, far_bias * P)
    if e <= 0:
        return 0.15, 0.85
    if cx < 0.5 * W:                       # far side = LEFT
        return 0.10, max(0.10, 0.90 - e * 0.80)
    return min(0.90, 0.10 + e * 0.80), 0.90


def render(cache, scores, events, windows, far_bias, tag, panel_w=1280):
    path, fps, W, h, prefix = cache["path"], cache["fps"], cache["w"], cache["h"], cache["prefix"]
    active = defaultdict(set)
    for tid, s, e in events:
        for f in range(max(0, s), e + 1):
            active[f].add(tid)
    in_win = np.zeros(cache["total"] + 1, dtype=bool)
    for wd in windows:
        in_win[wd["start"]:min(len(in_win), wd["end"] + 1)] = True

    fr_by = {f["frame"]: f for f in cache["frames"]}
    ttl = max(1, round(cv.TTL_SEC * fps))
    lt = LaneLineTracker(h, W, 0)
    gt = GhostMaskTracker(h, W, ttl=ttl, island_erode_frac=cv.ISLAND_ERODE_FRAC)

    cap = cv2.VideoCapture(path)
    pw, ph = panel_w, int(panel_w * h / W)
    os.makedirs(cv.OUT_DIR, exist_ok=True)
    out = os.path.join(cv.OUT_DIR, f"{prefix}_{tag}.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rec = fr_by.get(fi)
        firing = active.get(fi, set())
        if rec is not None:
            comps, lab, union = reconcile_components(rec["lanes"], h, W)
            info = lt.update(comps)
            surface = surface_from_components(comps, lab, union, info, h, W, cv.ISLAND_ERODE_FRAC)
            gt.step(rec["vehicles"], surface, tuple(rec.get("shift", [0.0, 0.0])))
            if gt.ghosts:
                gm = np.zeros((h, W), np.uint8)
                for g in gt.ghosts.values():
                    gm = cv2.bitwise_or(gm, g["mask"])
                tint = frame.copy(); tint[gm > 0] = (0, 0, 255)
                cv2.addWeighted(tint, 0.35, frame, 0.65, 0, frame)
            for c in comps:
                meta = info.get(c["id"], {})
                col = cv.CMAP.get(meta.get("stable_cls", c["cls"]), (180, 180, 180))
                cnts, _ = cv2.findContours((lab == c["id"]).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(frame, cnts, True, col, 2)
            for v in rec["vehicles"]:
                x1, y1, x2, y2 = v["bbox"]; w = max(1, x2 - x1); cx = 0.5 * (x1 + x2)
                tid = v["track_id"]
                P = scores.get(tid, {}).get(fi, 0.0)
                lo, hi = eff_window(cx, W, P, far_bias)
                hitting = tid in firing
                col = (0, 0, 255) if hitting else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if hitting else 2)
                tl, tr = trigger_points(v["bbox"])
                cv2.circle(frame, tl, 5, (0, 200, 255), -1); cv2.circle(frame, tr, 5, (0, 200, 255), -1)
                cv2.line(frame, (int(x1 + lo * w), y2), (int(x1 + hi * w), y2), (0, 255, 0), 3)
                cv2.putText(frame, f"P={P:.2f}", (x1, max(y1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                if hitting:
                    cv2.putText(frame, f"VIOLATION #{tid}", (x1, max(y1 - 28, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if firing:
            cv2.rectangle(frame, (0, 0), (W - 1, h - 1), (0, 0, 255), 14)
            cv2.rectangle(frame, (0, 14), (W - 1, 70), (0, 0, 0), -1)
            ids = ",".join(f"#{t}" for t in sorted(firing))
            cv2.putText(frame, f"VIOLATION  vehicle {ids}", (20, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        gtc = (0, 220, 0) if (fi < len(in_win) and in_win[fi]) else (60, 60, 60)
        cv2.rectangle(frame, (0, 0), (W - 1, 12), gtc, -1)
        cv2.putText(frame, f"{prefix} [{tag}] f{fi}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        wr.write(cv2.resize(frame, (pw, ph)))
        fi += 1
    cap.release(); wr.release()
    print(f"   -> {out}")


def load(prefix, windows):
    c = cv.load_cache(prefix)
    if c is None:
        return None
    c = cv.ensure_shifts(c)
    shifts = [fr.get("shift", [0.0, 0.0]) for fr in c["frames"]]
    tl = compute_enriched_timeline(c["frames"], shifts, c["h"], c["w"], c["fps"],
                                   ttl_sec=cv.TTL_SEC, island_erode_frac=cv.ISLAND_ERODE_FRAC,
                                   phantom_min_sec=cv.PHANTOM_MIN_SEC)
    sc = compute_motion_scores(c["frames"], c["h"], **MF)
    return c, tl, sc, windows


def main():
    with open(cv.LABELS_JSON) as fh:
        viol = {p: i.get("windows", []) for p, i in json.load(fh)["clips"].items()}
    hwy = {os.path.splitext(os.path.basename(f))[0]: [] for f in sorted(glob.glob(os.path.join(HIGHWAY_DIR, "*.mp4")))
           if "_annotated" not in f and "_winner" not in f}

    for setname, clips in (("viol", viol), ("hwy", hwy)):
        for prefix, windows in clips.items():
            todo = [c for c in CONFIGS if c[2] in (setname, "both")]
            if not todo:
                continue
            loaded = load(prefix, windows)
            if loaded is None:
                print(f"[skip] {prefix}: no cache"); continue
            c, tl, sc, win = loaded
            for tag, knobs, _ in todo:
                kf = max(1, round(KBASE * c["fps"]))
                ev = events_configured(tl, sc, kf, c["h"], c["w"], **knobs)
                print(f"[render] {prefix} [{tag}] events={len(ev)}")
                render(c, sc, ev, win, knobs["far_bias"], tag)
    print("[done]")


if __name__ == "__main__":
    main()
