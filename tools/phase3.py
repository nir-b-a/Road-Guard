"""
PHASE 3 -- deployment polish:
  * ghost TTL lowered to 0.5 s (crossing_violation_test.TTL_SEC) -- no stale-ghost firing,
  * solid lines ~5% wider (w_alpha 0.037, w_min 5) -- a touch more reliable, more visible,
  * 3 s per-track cooldown -- one incident per vehicle,
  * ENVELOPE police clips: each merged incident exported +/-5 s, RAW (evidence) + OVERLAID (triage).

Re-scores the ship config (far=0.75, alpha=5, K=0.10) on all 12 caches with the new TTL +
widened lines, reports raw-vs-cooldown event counts, then exports evidence clips for one demo
violation clip (the +/-5s overlay doubles as the width eyeball).

  <gpu-python> tools/phase3.py [demo_clip_prefix]
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
from motion_filter import compute_motion_scores, events_configured, merge_events  # noqa: E402
from render_configs import eff_window  # noqa: E402

HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")
MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
SHIP = dict(far_bias=0.75, alpha=5.0, horizon_gain=0.0)
KBASE = 0.10
COOLDOWN = 3.0
PRE, POST = 5.0, 5.0
# widened tightening (~+5% wide end via w_alpha, +25% thin-end floor via w_min for visibility)
TVP_FRAC, W_ALPHA, W_MIN = 0.45, 0.037, 5.0
EVID_DIR = os.path.join(cv.REPO, "outputs", "evidence")
DEMO = sys.argv[1] if len(sys.argv) > 1 else "B7-3EuQFAZM"


def tighten_for(H):
    return {"y_vp": TVP_FRAC * H, "w_alpha": W_ALPHA, "w_min": W_MIN}


def events_for(c):
    tt = tighten_for(c["h"])
    shifts = [fr.get("shift", [0.0, 0.0]) for fr in c["frames"]]
    tl = compute_enriched_timeline(c["frames"], shifts, c["h"], c["w"], c["fps"],
                                   ttl_sec=cv.TTL_SEC, island_erode_frac=cv.ISLAND_ERODE_FRAC,
                                   phantom_min_sec=cv.PHANTOM_MIN_SEC, tighten=tt)
    sc = compute_motion_scores(c["frames"], c["h"], **MF)
    kf = max(1, round(KBASE * c["fps"]))
    raw = events_configured(tl, sc, kf, c["h"], c["w"], **SHIP)
    return tt, sc, raw, merge_events(raw, c["fps"], COOLDOWN)


def export_evidence(c, sc, tt, merged):
    """One RAW + one OVERLAID clip per merged incident, window = [start-PRE, end+POST]."""
    fps, W, h, prefix = c["fps"], c["w"], c["h"], c["prefix"]
    fr_by = {f["frame"]: f for f in c["frames"]}
    active = defaultdict(set)
    wins = []
    for i, (tid, s, e) in enumerate(merged):
        w0, w1 = max(0, int(s - PRE * fps)), min(c["total"], int(e + POST * fps))
        wins.append((i, tid, w0, w1))
        for f in range(s, e + 1):
            active[f].add(tid)
    os.makedirs(EVID_DIR, exist_ok=True)
    pw, ph = 1280, int(1280 * h / W)
    raw_w, ov_w = {}, {}

    def writer(name):
        return cv2.VideoWriter(os.path.join(EVID_DIR, name), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph))

    lt = LaneLineTracker(h, W, 0)
    gt = GhostMaskTracker(h, W, ttl=max(1, round(cv.TTL_SEC * fps)), island_erode_frac=cv.ISLAND_ERODE_FRAC)
    cap = cv2.VideoCapture(c["path"]); fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        in_any = [w for w in wins if w[2] <= fi <= w[3]]
        raw_small = cv2.resize(frame, (pw, ph)) if in_any else None
        rec = fr_by.get(fi)
        if rec is not None and in_any:
            comps, lab, union = reconcile_components(rec["lanes"], h, W)
            info = lt.update(comps)
            surf = surface_from_components(comps, lab, union, info, h, W, cv.ISLAND_ERODE_FRAC, tt)
            gt.step(rec["vehicles"], surf, tuple(rec.get("shift", [0.0, 0.0])))
            frame[surf > 0] = (255, 0, 255)
            firing = active.get(fi, set())
            for v in rec["vehicles"]:
                x1, y1, x2, y2 = v["bbox"]; w = max(1, x2 - x1); cx = 0.5 * (x1 + x2)
                tid = v["track_id"]; P = sc.get(tid, {}).get(fi, 0.0)
                lo, hi = eff_window(cx, W, P, SHIP["far_bias"])
                hot = tid in firing
                col = (0, 0, 255) if hot else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if hot else 2)
                tl_, tr_ = trigger_points(v["bbox"])
                cv2.circle(frame, tl_, 5, (0, 200, 255), -1); cv2.circle(frame, tr_, 5, (0, 200, 255), -1)
                cv2.line(frame, (int(x1 + lo * w), y2), (int(x1 + hi * w), y2), (0, 255, 0), 3)
                if hot:
                    cv2.putText(frame, f"VIOLATION #{tid}", (x1, max(y1 - 10, 28)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            if firing:
                cv2.rectangle(frame, (0, 0), (W - 1, h - 1), (0, 0, 255), 14)
        ov_small = cv2.resize(frame, (pw, ph)) if in_any else None
        for (i, tid, w0, w1) in in_any:
            if i not in raw_w:
                raw_w[i] = writer(f"{prefix}_incident{i}_raw.mp4")
                ov_w[i] = writer(f"{prefix}_incident{i}_overlay.mp4")
            raw_w[i].write(raw_small); ov_w[i].write(ov_small)
        fi += 1
    cap.release()
    for i in raw_w:
        raw_w[i].release(); ov_w[i].release()
        print(f"   incident {i}: {EVID_DIR}\\{prefix}_incident{i}_raw.mp4  (+ _overlay.mp4)")


def main():
    with open(cv.LABELS_JSON) as fh:
        viol = {p: i.get("windows", []) for p, i in json.load(fh)["clips"].items()}
    hwy = {os.path.splitext(os.path.basename(f))[0]: None
           for f in sorted(glob.glob(os.path.join(HIGHWAY_DIR, "*.mp4")))
           if "_annotated" not in f and "_winner" not in f}

    tp = fp = fn = 0.0; hfp = 0; hwy_min = 0.0; raw_tot = mrg_tot = 0
    demo_pack = None
    print(f"[phase3] ghost TTL={cv.TTL_SEC}s, width w_alpha={W_ALPHA}/w_min={W_MIN}, cooldown={COOLDOWN}s")
    for prefix, windows in {**viol, **hwy}.items():
        c = cv.load_cache(prefix)
        if c is None:
            print(f"[skip] {prefix}"); continue
        c = cv.ensure_shifts(c)
        tt, sc, raw, merged = events_for(c)
        raw_tot += len(raw); mrg_tot += len(merged)
        if windows is None:
            hfp += len(merged); hwy_min += c["total"] / c["fps"] / 60.0
        else:
            s = cv.score_clip(merged, windows, c["fps"])
            tp += s["tp"]; fp += s["fp"]; fn += s["fn"]
        if prefix == DEMO:
            demo_pack = (c, sc, tt, merged)

    print("\n" + "=" * 60)
    print("PHASE 3  (ghost TTL 0.5 + width +5% + 3s cooldown)  ship config")
    print(f"  VIOL  TP={tp:.1f} FP={fp:.0f} FN={fn:.1f}   HWY FP={hfp} ({hfp / hwy_min:.1f}/min)")
    print(f"  events: raw={raw_tot} -> after 3s cooldown={mrg_tot}")
    print("=" * 60)

    if demo_pack:
        c, sc, tt, merged = demo_pack
        print(f"\n[evidence] {DEMO}: {len(merged)} incident(s) -> +/-5s raw + overlay clips")
        export_evidence(c, sc, tt, merged)
    else:
        print(f"[evidence] demo clip {DEMO} not found")
    print("[done]")


if __name__ == "__main__":
    main()
