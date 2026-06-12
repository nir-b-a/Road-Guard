"""
Isolated ablation of the two FP-reduction ideas, on the existing caches.

Per clip we compute ONCE: the ENRICHED verdict timeline (hit geometry: fracs + y + cx) and the
per-(track,frame) oncoming P. Then every config is a cheap post-filter via events_configured:
  Idea A (far_bias)      -- require deep far-from-ego overlap, scaled by P.
  Idea B (horizon_gain)  -- stretch K by proximity-to-horizon.
  dyn-K  (alpha)         -- stretch K by oncoming motion P.
We report VIOL TP/FP/FN beside clean-HIGHWAY FP and FP/min. RECALL-FIRST: VIOL TP must stay 7.3.

  <gpu-python> tools/eval_geometry.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossing_violation_test as cv  # noqa: E402
from ghost_mask import compute_enriched_timeline  # noqa: E402
from motion_filter import compute_motion_scores, events_configured  # noqa: E402

HIGHWAY_DIR = os.path.join(cv.REPO, "tests_videos", "high_way_drive")
MF = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
KBASE = 0.10                          # all idea rows use K_base = 0.10 s
_data = {"viol": {}, "hwy": {}}


def load_set(highway: bool):
    if highway:
        stems = [os.path.splitext(os.path.basename(f))[0]
                 for f in sorted(glob.glob(os.path.join(HIGHWAY_DIR, "*.mp4")))
                 if "_annotated" not in f and "_winner" not in f]
        labels = {s: {"windows": []} for s in stems}
    else:
        with open(cv.LABELS_JSON) as fh:
            labels = json.load(fh)["clips"]
    out = {}
    for prefix, info in labels.items():
        c = cv.load_cache(prefix)
        if c is None:
            print(f"[skip] {prefix}: no cache"); continue
        c = cv.ensure_shifts(c)
        shifts = [fr.get("shift", [0.0, 0.0]) for fr in c["frames"]]
        tl = compute_enriched_timeline(c["frames"], shifts, c["h"], c["w"], c["fps"],
                                       ttl_sec=cv.TTL_SEC, island_erode_frac=cv.ISLAND_ERODE_FRAC,
                                       phantom_min_sec=cv.PHANTOM_MIN_SEC)
        sc = compute_motion_scores(c["frames"], c["h"], **MF)
        out[prefix] = {"cache": c, "tl": tl, "sc": sc, "windows": info.get("windows", [])}
    return out


def run(label, *, k_base=KBASE, **knobs):
    tp = fp = fn = 0.0
    for d in _data["viol"].values():
        kf = max(1, round(k_base * d["cache"]["fps"]))
        ev = events_configured(d["tl"], d["sc"], kf, d["cache"]["h"], d["cache"]["w"], **knobs)
        s = cv.score_clip(ev, d["windows"], d["cache"]["fps"])
        tp += s["tp"]; fp += s["fp"]; fn += s["fn"]
    hfp = 0
    for d in _data["hwy"].values():
        kf = max(1, round(k_base * d["cache"]["fps"]))
        hfp += len(events_configured(d["tl"], d["sc"], kf, d["cache"]["h"], d["cache"]["w"], **knobs))
    flag = "" if abs(tp - 7.3) < 0.05 else "  <-- TP DROPPED!"
    print(f"{label:<40} | {tp:>5.1f} {fp:>4.0f} {fn:>5.1f} | {hfp:>6} {hfp / _data['hwy_min']:>7.1f}{flag}")


def main():
    print("[load] violation clips..."); _data["viol"] = load_set(False)
    print("[load] highway clips...");   _data["hwy"] = load_set(True)
    _data["hwy_min"] = sum(d["cache"]["total"] / d["cache"]["fps"] / 60.0 for d in _data["hwy"].values())

    hdr = f"{'config':<40} | {'TP':>5} {'FP':>4} {'FN':>5} | {'HWYFP':>6} {'FP/min':>7}"
    print("\n" + "=" * len(hdr)); print(f"ISOLATED ABLATION  (K_base={KBASE}s, MF={MF}, agg=median)")
    print(hdr); print("-" * len(hdr))

    print("# CONTROL (no ideas)")
    run("static K=0.05 (baseline)", k_base=0.05)
    run("static K=0.10 (baseline)", k_base=0.10)

    print("# IDEA A only -- directional far-side shift (far_bias)")
    for fb in (0.25, 0.5, 0.75, 1.0):
        run(f"A far_bias={fb}", far_bias=fb)

    print("# IDEA B only -- horizon-proximity (horizon_gain x horizon_frac)")
    for hf in (0.40, 0.50):
        for hg in (1.0, 2.0, 4.0):
            run(f"B gain={hg} frac={hf}", horizon_gain=hg, horizon_frac=hf)

    print("# A + B  (no motion dyn-K)")
    for fb in (0.5, 0.75, 1.0):
        for hg in (2.0, 4.0):
            run(f"A+B far={fb} gain={hg} frac=0.45", far_bias=fb, horizon_gain=hg, horizon_frac=0.45)

    print("# FULL STACK  A + B + motion dyn-K (alpha=5)")
    for fb in (0.5, 0.75, 1.0):
        for hg in (2.0, 4.0):
            run(f"FULL far={fb} gain={hg} alpha=5", far_bias=fb, horizon_gain=hg,
                horizon_frac=0.45, alpha=5.0)
    run("dyn-K only alpha=5 (reference)", alpha=5.0)

    print("=" * len(hdr))
    print("Goal: VIOL TP stays 7.3 while HWY FP/min drops. Compare each block to the K=0.10 control.")


if __name__ == "__main__":
    main()
