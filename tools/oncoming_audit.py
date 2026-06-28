"""
Opposite-direction (oncoming) AUDIT for the confidence pipeline (cache-only, CPU).

Question Tal asked: do cars driving the OPPOSITE direction currently fire crossing
violations (with a YOLOv8 bbox "violation zone") and get a confidence score?

The real geometric detector already exists -- oncoming_by_side() in ghost_mask -- but the
confidence path runs the timeline with side_gate=False, so it is OFF. Here we settle it by
computing each clip's verdict timeline TWICE:

    tl_off : side_gate=False  (exactly what confidence sees today)
    tl_on  : side_gate=True   (oncoming suppressed)

A (track, frame) that is a HIT in tl_off but NOT in tl_on was suppressed *because the line is
between us and the car* -> opposite-direction. We roll those up to events, cross-reference the
TP/FP label and the live two-sigmoid confidence, and print concrete examples.

Run:  python tools/oncoming_audit.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossing_violation_test as cvt                              # noqa: E402
from ghost_mask import compute_verdict_timeline, events_from_timeline  # noqa: E402
from sigmoid_search import conf_vec                                # noqa: E402
import violation_confidence as vc                                  # noqa: E402

ONCOMING_FRAC = 0.5      # event counts as opposite-direction if >=this share of its hit-frames are gated


def _timeline(cache, side_gate):
    shifts = [fr.get("shift", [0.0, 0.0]) for fr in cache["frames"]]
    return compute_verdict_timeline(cache["frames"], shifts, cache["h"], cache["w"], cache["fps"],
                                    ttl_sec=cvt.TTL_SEC, island_erode_frac=cvt.ISLAND_ERODE_FRAC,
                                    phantom_min_sec=cvt.PHANTOM_MIN_SEC, side_gate=side_gate)


def audit_clip(prefix, sig, windows):
    cache = cvt.load_cache(prefix)
    if cache is None:
        return []
    cache = cvt.ensure_shifts(cache)
    tl_off = _timeline(cache, side_gate=False)
    tl_on = _timeline(cache, side_gate=True)
    kf = max(1, round(vc.EVENT_K_SEC * cache["fps"]))

    rows = []
    for tid, s, e in events_from_timeline(tl_off, kf):
        on_map = tl_on.get(tid, {})
        frames = range(s, e + 1)
        gated = [f for f in frames if tl_off[tid].get(f) and not on_map.get(f)]
        frac = len(gated) / max(1, len(list(frames)))
        feat = vc.event_features(cache, (tid, s, e))
        if feat is None:
            continue
        conf = float(conf_vec(np.array([feat["distance"]]), np.array([feat["angle_off"]]),
                              sig["d50"], sig["k_d"], sig["a50"], sig["k_a"])[0])
        is_tp, _ = vc.label_event((tid, s, e), windows) if windows is not None else (False, 1.0)
        rows.append({"clip": prefix, "tid": tid, "s": s, "e": e, "oncoming_frac": frac,
                     "is_oncoming": frac >= ONCOMING_FRAC, "distance": feat["distance"],
                     "angle_off": feat["angle_off"], "conf": conf, "tp": is_tp})
    return rows


def main():
    with open(sig_path := "outputs/violation_compare/sigmoid_best.json") as fh:
        sig = json.load(fh)["params"]
    with open(cvt.LABELS_JSON) as fh:
        labels = json.load(fh)["clips"]

    all_rows = []
    for prefix, meta in labels.items():
        all_rows += audit_clip(prefix, sig, meta["windows"])
    for prefix in vc.FP_ONLY_CLIPS:
        all_rows += audit_clip(prefix, sig, windows=[])     # highway = pure FP

    onc = [r for r in all_rows if r["is_oncoming"]]
    onc_fp = [r for r in onc if not r["tp"]]
    onc_tp = [r for r in onc if r["tp"]]
    print(f"\n================  ONCOMING AUDIT  (sigmoid winner d50={sig['d50']} k_d={sig['k_d']} "
          f"a50={sig['a50']} k_a={sig['k_a']})  ================")
    print(f"total fired events : {len(all_rows)}")
    print(f"opposite-direction : {len(onc)}   (>= {int(ONCOMING_FRAC*100)}% of hit-frames have the line between ego & car)")
    print(f"   of those, FP    : {len(onc_fp)}   <- these are exactly the false alarms to suppress")
    print(f"   of those, TP    : {len(onc_tp)}   <- recall risk: real violations the gate would also kill")
    if onc:
        print(f"\nmean confidence  oncoming-FP={np.mean([r['conf'] for r in onc_fp]) if onc_fp else float('nan'):.3f}   "
              f"oncoming-TP={np.mean([r['conf'] for r in onc_tp]) if onc_tp else float('nan'):.3f}")
        print("\n--- concrete opposite-direction events (top by current confidence) ---")
        print(f"  {'clip':<16} {'tid':>4} {'frames':>13} {'onc%':>5} {'dist':>6} {'angle':>6} {'conf':>5}  label")
        for r in sorted(onc, key=lambda r: -r["conf"])[:15]:
            print(f"  {r['clip']:<16} {r['tid']:>4} {r['s']:>5}-{r['e']:<5} {r['oncoming_frac']*100:>4.0f}% "
                  f"{r['distance']:>6.1f} {r['angle_off']:>6.2f} {r['conf']:>5.2f}  {'TP' if r['tp'] else 'FP'}")
    out = "outputs/violation_compare/oncoming_audit.json"
    with open(out, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
