"""
False-positive stress test on clean highway footage (NO violations expected).

Runs the exact same violation pipeline as crossing_violation_test.py (lane seg + vehicle
track -> spatial reconcile + LaneLineTracker -> ghost/verdict timeline) on a folder of
no-violation clips. EVERY event it produces is a false positive. Reports the FP count and
the FP-per-minute rate per clip, at both the render K (0.05s) and the score-winner K (0.40s),
and renders overlay videos so the false alarms can be eyeballed.

Cache-once: pass 1 (models) caches to outputs/violation_cache/<stem>.json, reused on reruns.

  C:/Users/talgx/miniconda3/envs/roadguard-dl/python.exe tools/fp_highway_test.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossing_violation_test as cv  # noqa: E402
from ghost_mask import events_from_timeline  # noqa: E402

HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")
K_REPORT = [0.05, 0.40]   # render-K (recall-first) and score-winner K


def main() -> None:
    ap = argparse.ArgumentParser(description="False-positive stress test on clean highway clips.")
    ap.add_argument("--dir", default=HIGHWAY_DIR)
    ap.add_argument("--lane-weights", default=cv.LANE_WEIGHTS)
    ap.add_argument("--vehicle-weights", default=cv.VEH_WEIGHTS)
    ap.add_argument("--lane-conf", type=float, default=0.25)
    ap.add_argument("--veh-conf", type=float, default=0.30)
    ap.add_argument("--render-k", type=float, default=0.05)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    clips = sorted(f for f in glob.glob(os.path.join(args.dir, "*.mp4"))
                   if "_annotated" not in os.path.basename(f) and "_winner" not in os.path.basename(f))
    print(f"[fp-test] {len(clips)} clean clips in {args.dir}")

    # ---- Pass 1: caches (models run here only) ----
    lane_model = None
    caches = {}
    for path in clips:
        stem = os.path.splitext(os.path.basename(path))[0]
        cache = None if args.refresh else cv.load_cache(stem)
        if cache is None:
            if lane_model is None:
                from ultralytics import YOLO
                lane_model = YOLO(args.lane_weights, task="segment")
                print(f"[pass1] lane model {args.lane_weights}")
            cache = cv.build_cache(stem, lane_model, args.vehicle_weights,
                                   args.lane_conf, args.veh_conf, path=path)
        else:
            print(f"[pass1] {stem}: cache hit ({cache['total']} frames)")
        if cache is not None:
            cache["path"] = path           # keep an absolute path for shift backfill / render
            caches[stem] = cache

    # ---- ghost verdict timelines (compute once per clip) ----
    for s in caches:
        caches[s] = cv.ensure_shifts(caches[s])
    print(f"\n[pass2] computing verdict timelines for {len(caches)} clips...")
    timelines = {s: cv.clip_timeline(c) for s, c in caches.items()}

    # ---- report: every event = a false positive ----
    print("\n" + "=" * 72)
    print("FALSE-POSITIVE STRESS TEST (clean highway, 0 violations expected)")
    print(f"{'clip':<34}{'min':>5} | " + " | ".join(f"K={k}s FP (FP/min)" for k in K_REPORT))
    print("-" * 72)
    totals = {k: 0 for k in K_REPORT}
    for s, c in caches.items():
        mins = c["total"] / c["fps"] / 60.0
        cells = []
        for k in K_REPORT:
            kf = max(1, round(k * c["fps"]))
            n = len(events_from_timeline(timelines[s], kf))
            totals[k] += n
            cells.append(f"{n:>3} ({n / mins:>4.1f}/min)")
        print(f"{s[:33]:<34}{mins:>5.1f} | " + " | ".join(cells))
    print("-" * 72)
    print(f"{'TOTAL':<34}{'':>5} | " + " | ".join(f"{totals[k]:>3} FP" for k in K_REPORT))
    print("=" * 72)

    # ---- render overlays at render-K so the false alarms are visible ----
    if not args.no_video:
        print(f"\n[render] FP overlay videos at K={args.render_k}s ...")
        for s, c in caches.items():
            kf = max(1, round(args.render_k * c["fps"]))
            ev = events_from_timeline(timelines[s], kf)
            cv.render_winner(c, ev, [])     # no GT windows -> all boxes are false alarms
    print("\n[done]")


if __name__ == "__main__":
    main()
