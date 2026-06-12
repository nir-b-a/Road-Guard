"""
A/B the motion-aware dynamic-K against the static baseline, on the existing caches.

For every clip we compute (once): the verdict timeline + the per-(track,frame) oncoming P.
Then we sweep (K_base, alpha) cheaply: alpha=0 is the static baseline; alpha>0 stretches the
consecutive-frame requirement for oncoming-looking tracks.  We report VIOLATION recall and
clean-HIGHWAY false positives together so the trade-off is visible in one table.

  <gpu-python> tools/eval_dynamic_k.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossing_violation_test as cv  # noqa: E402
from motion_filter import compute_motion_scores, events_from_timeline_dynamic  # noqa: E402

HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")

# MotionDirectionFilter params, set from the dumped V_y distributions:
#   real line-drivers: median V_y <= 0 ; highway oncoming: median ~+0.09, p90 ~+0.25
MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
K_BASE_SECS = [0.05, 0.10]
ALPHAS = [0.0, 3.0, 5.0, 8.0]            # 0.0 == static baseline
AGG = "median"


def load_set(highway: bool):
    if highway:
        stems = [os.path.splitext(os.path.basename(f))[0]
                 for f in sorted(glob.glob(os.path.join(HIGHWAY_DIR, "*.mp4")))
                 if "_annotated" not in f and "_winner" not in f]
        labels = {s: {"windows": []} for s in stems}
    else:
        with open(cv.LABELS_JSON) as fh:
            labels = json.load(fh)["clips"]
    data = {}
    for prefix, info in labels.items():
        c = cv.load_cache(prefix)
        if c is None:
            print(f"[skip] {prefix}: no cache"); continue
        c = cv.ensure_shifts(c)
        tl = cv.clip_timeline(c)
        sc = compute_motion_scores(c["frames"], c["h"], **MF)
        data[prefix] = {"cache": c, "tl": tl, "sc": sc, "windows": info.get("windows", [])}
    return data


def main():
    print("[load] violation clips..."); viol = load_set(False)
    print("[load] highway clips...");   hwy = load_set(True)
    hwy_min = sum(d["cache"]["total"] / d["cache"]["fps"] / 60.0 for d in hwy.values())

    print("\n" + "=" * 78)
    print(f"DYNAMIC-K A/B   (MF={MF}, agg={AGG})")
    print(f"{'K_base':>6} {'alpha':>6} | {'VIOL TP':>8} {'FP':>4} {'FN':>5} | "
          f"{'HWY FP':>7} {'FP/min':>7}")
    print("-" * 78)
    for kb in K_BASE_SECS:
        for a in ALPHAS:
            tp = fp = fn = 0.0
            for d in viol.values():
                kf = max(1, round(kb * d["cache"]["fps"]))
                ev = events_from_timeline_dynamic(d["tl"], d["sc"], kf, a, AGG)
                s = cv.score_clip(ev, d["windows"], d["cache"]["fps"])
                tp += s["tp"]; fp += s["fp"]; fn += s["fn"]
            hfp = 0
            for d in hwy.values():
                kf = max(1, round(kb * d["cache"]["fps"]))
                hfp += len(events_from_timeline_dynamic(d["tl"], d["sc"], kf, a, AGG))
            tag = "  (static)" if a == 0 else ""
            print(f"{kb:>6} {a:>6} | {tp:>8.1f} {fp:>4.0f} {fn:>5.1f} | "
                  f"{hfp:>7} {hfp / hwy_min:>7.1f}{tag}")
        print("-" * 78)
    print("=" * 78)
    print("Goal: keep VIOL TP at the static value while HWY FP/min drops sharply.")


if __name__ == "__main__":
    main()
