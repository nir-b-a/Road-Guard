"""
PHASE 1 -- solid-line tightening (robust polyfit centreline + distance-adaptive width).

A/B the ship config (far_bias=0.75, alpha=5, horizon off, K_base=0.10) with tightening OFF vs
ON, reporting VIOL TP/FP and HWY FP on the existing caches. Then render ONE ~5-min video
(highway_5) with tightening ON so the thin, distance-tapered lines are visible.

  <gpu-python> tools/phase1.py
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
from render_configs import eff_window  # noqa: E402

HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")
MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
SHIP = dict(far_bias=0.75, alpha=5.0, horizon_gain=0.0)         # ship config (horizon off)
KBASE = 0.10
# Phase-1 tightening params (image-row proxy for depth; no calibration)
TVP_FRAC, W_ALPHA, W_MIN = 0.45, 0.035, 4.0
VIDEO_CLIP = "highway_5_trans_samaria"


def tighten_for(H):
    return {"y_vp": TVP_FRAC * H, "w_alpha": W_ALPHA, "w_min": W_MIN}


def enriched(c, tighten):
    shifts = [fr.get("shift", [0.0, 0.0]) for fr in c["frames"]]
    return compute_enriched_timeline(c["frames"], shifts, c["h"], c["w"], c["fps"],
                                     ttl_sec=cv.TTL_SEC, island_erode_frac=cv.ISLAND_ERODE_FRAC,
                                     phantom_min_sec=cv.PHANTOM_MIN_SEC, tighten=tighten)


def score_set(data, tl_key):
    tp = fp = fn = 0.0
    hfp = 0
    for d in data:
        kf = max(1, round(KBASE * d["c"]["fps"]))
        ev = events_configured(d[tl_key], d["sc"], kf, d["c"]["h"], d["c"]["w"], **SHIP)
        if d["windows"] is None:                       # highway: all events are FP
            hfp += len(ev)
        else:
            s = cv.score_clip(ev, d["windows"], d["c"]["fps"])
            tp += s["tp"]; fp += s["fp"]; fn += s["fn"]
    return tp, fp, fn, hfp


def render(c, sc, events, tighten, tag="phase1_tight"):
    path, fps, W, h, prefix = c["path"], c["fps"], c["w"], c["h"], c["prefix"]
    active = defaultdict(set)
    for tid, s, e in events:
        for f in range(max(0, s), e + 1):
            active[f].add(tid)
    fr_by = {f["frame"]: f for f in c["frames"]}
    ttl = max(1, round(cv.TTL_SEC * fps))
    lt = LaneLineTracker(h, W, 0)
    gt = GhostMaskTracker(h, W, ttl=ttl, island_erode_frac=cv.ISLAND_ERODE_FRAC)
    cap = cv2.VideoCapture(path)
    pw, ph = 1280, int(1280 * h / W)
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
            surf = surface_from_components(comps, lab, union, info, h, W, cv.ISLAND_ERODE_FRAC, tighten)
            gt.step(rec["vehicles"], surf, tuple(rec.get("shift", [0.0, 0.0])))
            if gt.ghosts:
                gm = np.zeros((h, W), np.uint8)
                for g in gt.ghosts.values():
                    gm = cv2.bitwise_or(gm, g["mask"])
                tint = frame.copy(); tint[gm > 0] = (0, 0, 255)
                cv2.addWeighted(tint, 0.35, frame, 0.65, 0, frame)
            # the TIGHTENED solid-line surface, drawn bright magenta so you can see it
            frame[surf > 0] = (255, 0, 255)
            for v in rec["vehicles"]:
                x1, y1, x2, y2 = v["bbox"]; w = max(1, x2 - x1); cx = 0.5 * (x1 + x2)
                tid = v["track_id"]; P = sc.get(tid, {}).get(fi, 0.0)
                lo, hi = eff_window(cx, W, P, SHIP["far_bias"])
                hot = tid in firing
                col = (0, 0, 255) if hot else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if hot else 2)
                tl, tr = trigger_points(v["bbox"])
                cv2.circle(frame, tl, 5, (0, 200, 255), -1); cv2.circle(frame, tr, 5, (0, 200, 255), -1)
                cv2.line(frame, (int(x1 + lo * w), y2), (int(x1 + hi * w), y2), (0, 255, 0), 3)
                cv2.putText(frame, f"P={P:.2f}", (x1, max(y1 - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                if hot:
                    cv2.putText(frame, f"VIOLATION #{tid}", (x1, max(y1 - 28, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if firing:
            cv2.rectangle(frame, (0, 0), (W - 1, h - 1), (0, 0, 255), 14)
        cv2.putText(frame, f"{prefix} [{tag}] f{fi}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        wr.write(cv2.resize(frame, (pw, ph)))
        fi += 1
    cap.release(); wr.release()
    print(f"[video] -> {out}")


def main():
    with open(cv.LABELS_JSON) as fh:
        viol_lbl = {p: i.get("windows", []) for p, i in json.load(fh)["clips"].items()}
    hwy_lbl = {os.path.splitext(os.path.basename(f))[0]: None
               for f in sorted(glob.glob(os.path.join(HIGHWAY_DIR, "*.mp4")))
               if "_annotated" not in f and "_winner" not in f}

    data, hwy_min = [], 0.0
    video_pack = None
    for prefix, windows in {**viol_lbl, **hwy_lbl}.items():
        c = cv.load_cache(prefix)
        if c is None:
            print(f"[skip] {prefix}"); continue
        c = cv.ensure_shifts(c)
        print(f"[timeline] {prefix} ({c['total']} frames) OFF+ON")
        tt = tighten_for(c["h"])
        d = {"c": c, "windows": windows, "sc": compute_motion_scores(c["frames"], c["h"], **MF),
             "off": enriched(c, None), "on": enriched(c, tt)}
        data.append(d)
        if windows is None:
            hwy_min += c["total"] / c["fps"] / 60.0
        if prefix == VIDEO_CLIP:
            video_pack = (c, d["sc"], d["on"], tt)

    print("\n" + "=" * 64)
    print(f"PHASE 1  solid-line tightening  (ship: {SHIP}, K_base={KBASE})")
    print(f"{'tightening':<12} | {'VIOL TP':>8} {'FP':>4} {'FN':>5} | {'HWY FP':>7} {'FP/min':>7}")
    print("-" * 64)
    for key, name in (("off", "OFF (base)"), ("on", "ON (phase1)")):
        tp, fp, fn, hfp = score_set(data, key)
        print(f"{name:<12} | {tp:>8.1f} {fp:>4.0f} {fn:>5.1f} | {hfp:>7} {hfp / hwy_min:>7.1f}")
    print("=" * 64)

    if video_pack:
        c, sc, tl, tt = video_pack
        kf = max(1, round(KBASE * c["fps"]))
        ev = events_configured(tl, sc, kf, c["h"], c["w"], **SHIP)
        print(f"[video] rendering {VIDEO_CLIP} (tighten ON, {len(ev)} events)...")
        render(c, sc, ev, tt)
    print("[done]")


if __name__ == "__main__":
    main()
