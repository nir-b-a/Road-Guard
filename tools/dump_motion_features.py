"""
Dump per-(track,frame) motion features to CSV so the oncoming P-score thresholds can be tuned
on REAL data, then print the separating distributions.

For each cached clip we run BOTH:
  * the existing verdict timeline  -> which (track,frame) pairs actually trip the solid line;
  * the MotionDirectionFilter      -> V_y, Y-origin anchor, and fused P for every track.
We emit one row per vehicle per frame with flags `hit` (verdict fired this frame) and
`in_window` (inside a labeled violation window).

Tune by comparing, among verdict-HIT rows:
  - TRUE POSITIVES : in_window==1 on the violation clips  (real same-direction line-drivers)
  - FALSE POSITIVES: every hit on the clean highway clips (oncoming traffic)
A good P/V_y threshold sits ABOVE the TP V_y distribution (so real violators keep K_base) and
BELOW the highway-FP V_y distribution (so oncoming traffic gets a stretched K).

  <gpu-python> tools/dump_motion_features.py            # violation clips  -> outputs/motion_features_violation.csv
  <gpu-python> tools/dump_motion_features.py --highway  # clean highway    -> outputs/motion_features_highway.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossing_violation_test as cv  # noqa: E402
from motion_filter import MotionDirectionFilter  # noqa: E402

HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")
COLS = ["clip", "frame", "track_id", "contact_y", "min_y", "bbox_h", "v_y", "anchor", "p",
        "k_dyn", "hit", "in_window"]


def clip_rows(prefix, cache, windows):
    """Per (track,frame) feature rows, joined with verdict-hit and in-window flags."""
    cache = cv.ensure_shifts(cache)
    timeline = cv.clip_timeline(cache)                       # {track_id: {frame: hit_bool}}
    H, W, fps = cache["h"], cache["w"], cache["fps"]
    mf = MotionDirectionFilter(H, k_base=max(1, round(0.5 * fps)))   # K_base ~ 0.5 s, in frames
    in_win = lambda f: any(wd["start"] <= f <= wd["end"] for wd in windows)
    rows = []
    for fr in cache["frames"]:
        f = fr["frame"]
        dy = fr.get("shift", [0.0, 0.0])[1]
        mf.update(f, fr["vehicles"], dy)
        for v in fr["vehicles"]:
            tid = v["track_id"]
            ft = mf.features(tid)
            hit = bool(timeline.get(tid, {}).get(f, False))
            rows.append([prefix, f, tid, round(ft["cy"], 1), round(ft["min_y"], 1),
                         round(ft["h"], 1), round(ft["v_y"], 4), int(ft["anchor"]),
                         round(ft["p"], 3), round(ft["k_dyn"], 1), int(hit), int(in_win(f))])
    return rows


def summarize(rows, label):
    """Print V_y / P percentiles for verdict-HIT rows, split TP (in_window) vs other."""
    hit = [r for r in rows if r[10] == 1]
    if not hit:
        print(f"  [{label}] no verdict-hit rows"); return
    tp = np.array([r[6] for r in hit if r[11] == 1], float)   # v_y of in-window hits
    fp = np.array([r[6] for r in hit if r[11] == 0], float)   # v_y of out-of-window hits
    def pct(a):
        if a.size == 0:
            return "   n=0"
        return (f"n={a.size:<4} median={np.median(a):+.3f} p90={np.percentile(a,90):+.3f} "
                f"p99={np.percentile(a,99):+.3f} max={a.max():+.3f}")
    print(f"  [{label}] V_y of verdict-hits:")
    print(f"     in-window (TP-ish): {pct(tp)}")
    print(f"     out-of-window     : {pct(fp)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--highway", action="store_true", help="use clean highway clips instead of violation clips")
    args = ap.parse_args()

    if args.highway:
        stems = [os.path.splitext(os.path.basename(f))[0] for f in sorted(glob.glob(os.path.join(HIGHWAY_DIR, "*.mp4")))
                 if "_annotated" not in f and "_winner" not in f]
        labels = {s: {"windows": []} for s in stems}
        out = os.path.join(cv.REPO, "outputs", "motion_features_highway.csv")
    else:
        with open(cv.LABELS_JSON) as fh:
            labels = json.load(fh)["clips"]
        out = os.path.join(cv.REPO, "outputs", "motion_features_violation.csv")

    all_rows = []
    for prefix, info in labels.items():
        cache = cv.load_cache(prefix)
        if cache is None:
            print(f"[skip] {prefix}: no cache"); continue
        print(f"[dump] {prefix} ({cache['total']} frames)")
        rows = clip_rows(prefix, cache, info.get("windows", []))
        summarize(rows, prefix)
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(COLS); w.writerows(all_rows)
    print(f"\n[done] {len(all_rows)} rows -> {out}")
    print("\n==== AGGREGATE ====")
    summarize(all_rows, "ALL")


if __name__ == "__main__":
    main()
