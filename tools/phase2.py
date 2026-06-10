"""
PHASE 2 -- per-Line-ID vector lane memory (the ghost, generalized).

Score-only A/B of the ship config (far=0.75, alpha=5, K=0.10) + Phase-1 tightening, with the
vector memory OFF vs ON, across all 12 cached clips. Goal: VIOL TP up / FN down (recall under
occlusion) while HWY FP stays flat (~16 -> the negative-update gate is holding). Then render
ALL 9 violation clips with memory ON (memory-injected line drawn ORANGE under occluding cars).

Baseline (memory OFF = Phase-1 tighten ON, ship config), from the prior run:
  VIOL TP=7.3 FP=70 FN=5.3 | HWY FP=16 (1.1/min)

  <gpu-python> tools/phase2.py
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
from ghost_mask import (GhostMaskTracker, LaneLineTracker, LaneMemory, compute_enriched_timeline,  # noqa: E402
                        reconcile_components, surface_from_components, trigger_points)
from motion_filter import compute_motion_scores, events_configured  # noqa: E402
from render_configs import eff_window  # noqa: E402

HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")
MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
SHIP = dict(far_bias=0.75, alpha=5.0, horizon_gain=0.0)
KBASE = 0.10
TVP_FRAC, W_ALPHA, W_MIN = 0.45, 0.035, 4.0


def tighten_for(H):
    return {"y_vp": TVP_FRAC * H, "w_alpha": W_ALPHA, "w_min": W_MIN}


def render(c, sc, events, tt, tag="phase2_mem"):
    path, fps, W, h, prefix = c["path"], c["fps"], c["w"], c["h"], c["prefix"]
    active = defaultdict(set)
    for tid, s, e in events:
        for f in range(max(0, s), e + 1):
            active[f].add(tid)
    fr_by = {f["frame"]: f for f in c["frames"]}
    ttl = max(1, round(cv.TTL_SEC * fps))
    lt = LaneLineTracker(h, W, 0)
    gt = GhostMaskTracker(h, W, ttl=ttl, island_erode_frac=cv.ISLAND_ERODE_FRAC)
    lm = LaneMemory(h, W, max(1, round(fps)), tt["y_vp"], tt["w_alpha"], tt["w_min"])
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
            surf = surface_from_components(comps, lab, union, info, h, W, cv.ISLAND_ERODE_FRAC, tt)
            msurf = lm.step(comps, info, lab, union, surf, rec["vehicles"], tuple(rec.get("shift", [0.0, 0.0])))
            full = cv2.bitwise_or(surf, msurf)
            gt.step(rec["vehicles"], full, tuple(rec.get("shift", [0.0, 0.0])))
            frame[surf > 0] = (255, 0, 255)          # live tightened solid line = magenta
            frame[msurf > 0] = (0, 165, 255)          # REMEMBERED line under a car = orange
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
            for wd in []:
                pass
        if firing:
            cv2.rectangle(frame, (0, 0), (W - 1, h - 1), (0, 0, 255), 14)
            cv2.rectangle(frame, (0, 14), (W - 1, 64), (0, 0, 0), -1)
            cv2.putText(frame, f"VIOLATION {','.join('#'+str(t) for t in sorted(firing))}",
                        (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, f"{prefix} [{tag}] f{fi}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        wr.write(cv2.resize(frame, (pw, ph)))
        fi += 1
    cap.release(); wr.release()
    print(f"   video -> {out}")


def main():
    with open(cv.LABELS_JSON) as fh:
        viol = {p: i.get("windows", []) for p, i in json.load(fh)["clips"].items()}
    hwy = {os.path.splitext(os.path.basename(f))[0]: None
           for f in sorted(glob.glob(os.path.join(HIGHWAY_DIR, "*.mp4")))
           if "_annotated" not in f and "_winner" not in f}

    tp = fp = fn = 0.0
    hfp = 0
    hwy_min = 0.0
    print("[phase2] computing memory-ON timelines + scoring...")
    print(f"{'clip':<16}{'TP':>5} {'FP':>4} {'FN':>5}")
    viol_packs = []
    for prefix, windows in {**viol, **hwy}.items():
        c = cv.load_cache(prefix)
        if c is None:
            print(f"[skip] {prefix}"); continue
        c = cv.ensure_shifts(c)
        tt = tighten_for(c["h"])
        shifts = [fr.get("shift", [0.0, 0.0]) for fr in c["frames"]]
        tl = compute_enriched_timeline(c["frames"], shifts, c["h"], c["w"], c["fps"],
                                       ttl_sec=cv.TTL_SEC, island_erode_frac=cv.ISLAND_ERODE_FRAC,
                                       phantom_min_sec=cv.PHANTOM_MIN_SEC, tighten=tt, memory=True)
        sc = compute_motion_scores(c["frames"], c["h"], **MF)
        kf = max(1, round(KBASE * c["fps"]))
        ev = events_configured(tl, sc, kf, c["h"], c["w"], **SHIP)
        if windows is None:
            hfp += len(ev); hwy_min += c["total"] / c["fps"] / 60.0
        else:
            s = cv.score_clip(ev, windows, c["fps"])
            tp += s["tp"]; fp += s["fp"]; fn += s["fn"]
            print(f"{prefix:<16}{s['tp']:>5.1f} {s['fp']:>4.0f} {s['fn']:>5.1f}")
            viol_packs.append((c, sc, ev, tt))

    print("\n" + "=" * 56)
    print("PHASE 2  vector lane memory  (ship config + tighten)")
    print(f"{'config':<26} | {'VIOL TP':>8} {'FP':>4} {'FN':>5} | {'HWY FP':>7} {'/min':>6}")
    print("-" * 56)
    print(f"{'baseline (Phase-1, mem OFF)':<26} | {7.3:>8.1f} {70:>4} {5.3:>5.1f} | {16:>7} {1.1:>6}")
    print(f"{'Phase 2 (memory ON)':<26} | {tp:>8.1f} {fp:>4.0f} {fn:>5.1f} | {hfp:>7} {hfp / hwy_min:>6.1f}")
    print("=" * 56)

    print("\n[render] all 9 violation clips with memory ON...")
    for c, sc, ev, tt in viol_packs:
        print(f"[render] {c['prefix']} events={len(ev)}")
        render(c, sc, ev, tt)
    print("[done]")


if __name__ == "__main__":
    main()
