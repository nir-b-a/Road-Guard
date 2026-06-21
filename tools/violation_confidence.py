"""
Per-violation CONFIDENCE score + calibration (cheap, cache-only).

For each detected crossing-violation EVENT we emit a float in [0,1] = P(real violation),
measured off the violating vehicle's bounding box (all from the harness cache -- no models,
no video). Two reliability features:

  distance   from bbox pixel-height (tools/distance.py) -- closer car => more reliable
  angle_off  |bbox-centre-x - frame-centre| / (W/2)     -- car near centre => more reliable
  oncoming   line sits BETWEEN ego & car (ghost_mask.oncoming_by_side) -- opposite-direction.
             Multiplies confidence by ONCOMING_FACTOR. This is the REAL direction signal; angle_off
             is only a centredness proxy and cannot down-weight a centred oncoming car.

The score is a CLASS-BALANCED LOGISTIC REGRESSION over these features (confidence_model.json):
confidence = sigmoid(w . standardize(features) + b). LR learns each feature's weight AND sign
and yields a calibrated probability.

WHY LR, not multiplicative target-shaped factors: the original plan multiplied hand-tuned
sigmoids toward TP=1/FP=0, but (a) three sub-1 factors can't reach ~1.0, and (b) the features
overlap, so 1/0 separation is unattainable. LR on distance+angle gives AUC ~0.78. A 3rd
feature, source RESOLUTION, was dropped: it appears to help (AUC ~0.86) only because the
pure-FP highway clips are all 1080p -- a dataset confound, not a reliability signal.

TP/FP come from violation_labels.json window overlap (+ highway clips as pure-FP sources).
Everything reads outputs/violation_cache/*.json, so this is a fast CPU run.

Run:  python tools/violation_confidence.py                 # build samples + fit + save model
      python tools/violation_confidence.py --reuse         # refit from cached features (instant)
      python tools/violation_confidence.py --reuse --render DeNnDugXxP0   # + overlay video
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghost_mask import events_from_timeline, compute_verdict_timeline  # noqa: E402
import crossing_violation_test as cvt                            # noqa: E402
from distance import distance_from_pixel_height                  # noqa: E402
import formula_search                                            # noqa: E402  (genetic-search scorer)
import sigmoid_search                                            # noqa: E402  (two-sigmoid scorer)

# Highway clips have caches but no labels -> every event on them is a FALSE positive.
FP_ONLY_CLIPS = ["highway_5_trans_samaria", "mitzpe_ramon_to_petah_tikva", "route_241_western_negev"]

EVENT_K_SEC = 0.05    # recall-first: surface many events (incl. FPs) so calibration sees both classes
PROD_FEATURES = ("distance", "angle_off")   # resolution excluded: confounded with FP-source clips
ONCOMING_FACTOR = 0.30   # confidence multiplier when the car is opposite-direction (line between ego & car)
SAMPLES_JSON = os.path.join(cvt.OUT_DIR, "confidence_samples.json")
MODEL_JSON = os.path.join(cvt.OUT_DIR, "confidence_model.json")


def clip_timeline_oncoming(cache: dict):
    """Single-pass verdict timeline PLUS the per-(track,frame) opposite-direction flag, so the
    confidence score can down-weight oncoming traffic without a second timeline pass."""
    shifts = [fr.get("shift", [0.0, 0.0]) for fr in cache["frames"]]
    return compute_verdict_timeline(cache["frames"], shifts, cache["h"], cache["w"], cache["fps"],
                                    ttl_sec=cvt.TTL_SEC, island_erode_frac=cvt.ISLAND_ERODE_FRAC,
                                    phantom_min_sec=cvt.PHANTOM_MIN_SEC, return_oncoming=True)


# --------------------------------------------------------------------------- #
# per-event feature extraction (off the violating track's bbox in its window)
# --------------------------------------------------------------------------- #
def _track_boxes(cache: dict, tid: int, start: int, end: int) -> list:
    fr_by_idx = {f["frame"]: f for f in cache["frames"]}
    out = []
    for fi in range(start, end + 1):
        rec = fr_by_idx.get(fi)
        if not rec:
            continue
        for v in rec["vehicles"]:
            if v["track_id"] == tid:
                out.append(v["bbox"])
                break
    return out


def event_features(cache: dict, event: tuple, oncoming: dict | None = None) -> dict | None:
    tid, s, e = event
    W, H = cache["w"], cache["h"]
    boxes = _track_boxes(cache, tid, s, e)
    if not boxes:
        return None
    heights = np.array([b[3] - b[1] for b in boxes], dtype=float)
    cxs = np.array([(b[0] + b[2]) / 2.0 for b in boxes], dtype=float)
    h_med = float(np.median(heights))                       # representative bbox height
    cx_med = float(np.median(cxs))
    feat = {
        "distance": distance_from_pixel_height(h_med),      # meters (calibrated power law)
        "pixel_height": h_med,
        "angle_off": abs(cx_med - W / 2.0) / (W / 2.0),     # 0 centred .. ~1 at frame edge
        "src_res": float(H),                                # source vertical resolution
    }
    if oncoming is not None:                                # majority vote over the event window
        flags = [bool(oncoming.get(tid, {}).get(f, False)) for f in range(s, e + 1)]
        feat["oncoming"] = bool(flags) and (sum(flags) / len(flags) >= 0.5)
    return feat


def label_event(event: tuple, windows: list) -> tuple[bool, float]:
    """TP if the event frame range overlaps any labelled window; weight from that window."""
    _, s, e = event
    for wd in windows:
        if e >= wd["start"] and s <= wd["end"]:
            return True, float(wd.get("weight", 1.0))
    return False, 1.0


# --------------------------------------------------------------------------- #
# dataset of (features, is_tp, weight) over all clips
# --------------------------------------------------------------------------- #
def build_dataset(labels: dict) -> list:
    samples = []
    # labelled clips (TP/FP by window overlap)
    for prefix, meta in labels.items():
        cache = cvt.load_cache(prefix)
        if cache is None:
            print(f"[skip] {prefix}: no cache")
            continue
        cache = cvt.ensure_shifts(cache)
        timeline, oncoming = clip_timeline_oncoming(cache)
        kf = max(1, round(EVENT_K_SEC * cache["fps"]))
        events = events_from_timeline(timeline, kf)
        n_tp = n_fp = 0
        for ev in events:
            feat = event_features(cache, ev, oncoming)
            if feat is None:
                continue
            is_tp, w = label_event(ev, meta["windows"])
            samples.append((feat, is_tp, w))
            n_tp += is_tp
            n_fp += (not is_tp)
        print(f"[data] {prefix:<14} events={len(events)}  TP={n_tp} FP={n_fp}")
    # pure-FP highway clips
    for prefix in FP_ONLY_CLIPS:
        cache = cvt.load_cache(prefix)
        if cache is None:
            continue
        cache = cvt.ensure_shifts(cache)
        timeline, oncoming = clip_timeline_oncoming(cache)
        kf = max(1, round(EVENT_K_SEC * cache["fps"]))
        events = events_from_timeline(timeline, kf)
        n = 0
        for ev in events:
            feat = event_features(cache, ev, oncoming)
            if feat is None:
                continue
            samples.append((feat, False, 1.0))
            n += 1
        print(f"[data] {prefix:<28} FP-only events={n}")
    return samples


def feature_diagnostic(samples: list) -> None:
    """Do the raw features even separate TP from FP? Print per-class distribution per feature."""
    def stats(vals):
        a = np.array(vals, dtype=float)
        return f"mean={a.mean():6.2f} med={np.median(a):6.2f} p10={np.percentile(a,10):6.2f} p90={np.percentile(a,90):6.2f}"
    print("\n--- raw feature separation (TP vs FP) ---")
    for key in ("distance", "angle_off", "pixel_height", "src_res"):
        tp = [f[key] for f, t, _ in samples if t]
        fp = [f[key] for f, t, _ in samples if not t]
        if tp and fp:
            print(f"  {key:12}  TP[{stats(tp)}]")
            print(f"  {'':12}  FP[{stats(fp)}]")


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank (Mann-Whitney U) statistic. Ranking quality, threshold-free."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_logreg(samples: list, feature_keys: tuple, verbose: bool = True) -> tuple[dict, float]:
    """Class-balanced logistic regression over the chosen features -> calibrated probability.
    Learns the optimal weight AND sign per feature. Returns (model, AUC); model carries the
    standardisation + coefficients so confidence_lr() can score any new event identically."""
    X = np.array([[f[k] for k in feature_keys] for f, _, _ in samples], dtype=float)
    y = np.array([1.0 if t else 0.0 for _, t, _ in samples])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    n = len(y)
    n_pos, n_neg = y.sum(), n - y.sum()
    w = np.where(y == 1, n / (2 * n_pos), n / (2 * n_neg))          # balance classes
    Xb = np.hstack([Xs, np.ones((n, 1))])
    theta = np.zeros(Xb.shape[1])
    for _ in range(5000):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ theta, -50, 50)))
        theta -= 0.1 * (Xb.T @ (w * (p - y)) / n)
    prob = 1.0 / (1.0 + np.exp(-np.clip(Xb @ theta, -50, 50)))
    auc = _auc(prob, y)
    model = {"features": list(feature_keys), "mu": mu.tolist(), "sd": sd.tolist(),
             "coef": theta[:-1].tolist(), "intercept": float(theta[-1])}
    if verbose:
        print(f"\n--- logistic regression  features={list(feature_keys)} ---")
        for k, c in zip(feature_keys, theta[:-1]):
            print(f"    coef[{k:12}] = {c:+.3f}  (standardized; sign = direction, |.| = strength)")
        print(f"    AUC={auc:.3f}   TP prob mean={prob[y==1].mean():.3f}   FP prob mean={prob[y==0].mean():.3f}"
              f"   sep={prob[y==1].mean()-prob[y==0].mean():+.3f}")
    return model, auc


def confidence_lr(feat: dict, model: dict) -> float:
    """Apply a saved logistic-regression model to one event's features -> confidence in [0,1]."""
    x = np.array([feat[k] for k in model["features"]], dtype=float)
    xs = (x - np.array(model["mu"])) / np.array(model["sd"])
    z = float(np.dot(xs, model["coef"]) + model["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def calibrate_for_fp(model: dict, samples: list, target_fp: float, gain: float) -> dict:
    """Return a copy of the LR model whose operating point is shifted so the MEAN false-positive
    confidence equals target_fp, keeping scores graded (smooth). `gain` scales the steepness:
    higher lets the clearest TPs reach higher while FPs stay centred at target_fp."""
    feats = model["features"]
    mu, sd = np.array(model["mu"]), np.array(model["sd"])
    coef = np.array(model["coef"]) * gain
    X = np.array([[f[k] for k in feats] for f, _, _ in samples], dtype=float)
    y = np.array([bool(t) for _, t, _ in samples])
    z0 = ((X - mu) / sd) @ coef
    lo, hi = -20.0, 20.0                                      # bisect the intercept for mean FP conf
    for _ in range(100):
        b = (lo + hi) / 2.0
        if (1.0 / (1.0 + np.exp(-(z0[~y] + b)))).mean() > target_fp:
            hi = b
        else:
            lo = b
    conf = 1.0 / (1.0 + np.exp(-(z0 + b)))
    print(f"\n[calibrate] FP-centred logistic  target_fp={target_fp}  gain={gain}")
    print(f"    intercept={b:+.3f}   TPconf={conf[y].mean():.3f}  FPconf={conf[~y].mean():.3f}"
          f"   max={conf.max():.3f}  frac>=0.99={(conf >= 0.99).mean():.2f}")
    tuned = dict(model)
    tuned["coef"] = coef.tolist()
    tuned["intercept"] = float(b)
    return tuned


def search_user_objective(samples: list, fp_w: float = 0.1, top: int = 8) -> tuple:
    """Try MANY confidence formulas and keep the one maximising Tal's objective:
        S = sum_TP (conf * 1)  -  fp_w * sum_FP (conf)
    (reward confidence on real violations, penalise it on false ones at weight fp_w).
    Raising fp_w forces the winner to actually push FP confidence DOWN (better separation).
    Formula = combine( f_distance, f_angle ); we sweep forms + params for both."""
    d = np.array([f["distance"] for f, _, _ in samples])
    a = np.array([f["angle_off"] for f, _, _ in samples])
    sign = np.array([1.0 if t else -fp_w for _, t, _ in samples])
    nTP, nFP = int((sign > 0).sum()), int((sign < 0).sum())

    def logd(x, x0, k):                    # 1 for small x -> 0 as x grows
        return 1.0 / (1.0 + np.exp(np.clip(k * (x - x0), -50, 50)))

    def lind(x, x0, s):                     # 1 until x0, linear ramp down
        return np.clip(1.0 - s * (x - x0), 0.0, 1.0)

    dist_funcs = [("dist=1", np.ones_like(d))]
    for x0 in (10, 15, 20, 25, 30, 40, 50, 70):
        for k in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
            dist_funcs.append((f"logistic(d50={x0},k={k})", logd(d, x0, k)))
        for s in (0.01, 0.02, 0.04):
            dist_funcs.append((f"linear(d0={x0},slope={s})", lind(d, x0, s)))

    ang_funcs = [("angle=1", np.ones_like(a))]
    for x0 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85):
        for k in (3, 5, 8, 12, 20):
            ang_funcs.append((f"logistic(a50={x0},k={k})", logd(a, x0, k)))

    combines = [("prod", lambda fd, fa: fd * fa),
                ("min", lambda fd, fa: np.minimum(fd, fa)),
                ("mean", lambda fd, fa: 0.5 * (fd + fa))]

    results = []
    for dn, dv in dist_funcs:
        for an, av in ang_funcs:
            for cn, cf in combines:
                conf = np.clip(cf(dv, av), 0.0, 1.0)
                results.append((float(conf @ sign), dn, an, cn, conf))
    results.sort(key=lambda r: r[0], reverse=True)

    print(f"\n=== USER-OBJECTIVE search   S = sum_TP conf  -  {fp_w} * sum_FP conf ===")
    print(f"  events TP={nTP} FP={nFP}   baseline(conf=1 all): S={sign.sum():.2f}   perfect(TP=1,FP=0): S={nTP:.2f}")
    print(f"  tried {len(results)} formulas; top {top}:")
    for S, dn, an, cn, conf in results[:top]:
        print(f"    S={S:6.2f}  [{cn:4}]  {dn:22} x {an:20}  TPconf={conf[sign>0].mean():.2f} FPconf={conf[sign<0].mean():.2f}")
    return results[0]


def render_confidence(prefix: str, scorer, k_sec: float = EVENT_K_SEC, panel_w: int = 1280,
                      hold_sec: float = 0.0) -> None:
    """Write <prefix>_confidence.mp4: each firing vehicle boxed in red with its confidence float.
    `scorer` is a callable feat-dict -> confidence in [0,1] (LR model or a genetic-search formula).
    hold_sec keeps the marker + a big top banner on screen for that many seconds AFTER the event
    (the box follows the car while it stays tracked), so the score is readable, not a one-frame flash."""
    cache = cvt.load_cache(prefix)
    if cache is None:
        print(f"[render] {prefix}: no cache")
        return
    cache = cvt.ensure_shifts(cache)
    timeline, oncoming = clip_timeline_oncoming(cache)
    kf = max(1, round(k_sec * cache["fps"]))
    hold_frames = max(0, round(hold_sec * cache["fps"]))
    active: dict = defaultdict(dict)                 # frame -> {track_id: confidence}
    for ev in events_from_timeline(timeline, kf):
        feat = event_features(cache, ev, oncoming)
        if feat is None:
            continue
        c = scorer(feat)
        tid, s, e = ev
        for f in range(s, e + 1 + hold_frames):      # hold the marker past the event
            active[f][tid] = max(active[f].get(tid, 0.0), c)

    path, fps, W, H = cache["path"], cache["fps"], cache["w"], cache["h"]
    fr_by_idx = {f["frame"]: f for f in cache["frames"]}
    cap = cv2.VideoCapture(path)
    pw, ph = panel_w, int(panel_w * H / W)
    os.makedirs(cvt.OUT_DIR, exist_ok=True)
    out_path = os.path.join(cvt.OUT_DIR, f"{prefix}_confidence.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph))
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    fscale = max(0.9, H / 600.0)                      # scale markers to source resolution
    thick = max(2, round(H / 240.0))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        firing = active.get(fi, {})
        rec = fr_by_idx.get(fi)
        if rec:
            for v in rec["vehicles"]:
                if v["track_id"] in firing:
                    x1, y1, x2, y2 = v["bbox"]
                    c = firing[v["track_id"]]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thick)
                    label = f"conf={c:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, FONT, fscale, thick)
                    ly = max(y1 - 8, th + 12)
                    cv2.rectangle(frame, (x1, ly - th - 10), (x1 + tw + 12, ly + 8), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1 + 6, ly), FONT, fscale, (255, 255, 255),
                                max(2, thick - 1), cv2.LINE_AA)
        if firing:                                    # big persistent banner so the score is readable
            cmax = max(firing.values())
            bh = int(H * 0.12)
            cv2.rectangle(frame, (0, 0), (W, bh), (0, 0, 255), -1)
            btext = f"VIOLATION   conf={cmax:.2f}"
            bscale = fscale * 1.7
            (tw, th), _ = cv2.getTextSize(btext, FONT, bscale, thick + 1)
            cv2.putText(frame, btext, (24, int(bh * 0.5 + th * 0.5)), FONT, bscale,
                        (255, 255, 255), thick + 1, cv2.LINE_AA)
        cv2.putText(frame, f"{prefix} f{fi}", (10, H - 20), FONT, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(cv2.resize(frame, (pw, ph)))
        fi += 1
    cap.release()
    writer.release()
    print(f"   confidence video -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-violation confidence calibration (cache-only).")
    ap.add_argument("--labels", default=cvt.LABELS_JSON)
    ap.add_argument("--reuse", action="store_true", help="reuse cached extracted features (skip timeline recompute)")
    ap.add_argument("--render", nargs="*", metavar="CLIP",
                    help="render confidence overlay video(s): give clip prefixes, or 'ALL' for every labeled clip")
    ap.add_argument("--search", action="store_true",
                    help="sweep many formulas to maximise S = sum_TP conf - 0.1*sum_FP conf")
    ap.add_argument("--hold-sec", type=float, default=0.0,
                    help="keep each violation marker + banner on screen this many seconds (readability)")
    ap.add_argument("--formula", metavar="JSON",
                    help="render with a saved genetic-search formula (best_formula.json) instead of the LR model")
    ap.add_argument("--sigmoid", metavar="JSON",
                    help="render with a saved two-sigmoid winner (sigmoid_best.json) instead of the LR model")
    ap.add_argument("--calibrate-fp", type=float, default=None, metavar="FP",
                    help="shift the logistic so mean false-positive confidence = this (e.g. 0.25); stays graded")
    ap.add_argument("--gain", type=float, default=2.0,
                    help="steepness for --calibrate-fp (higher = clearest TPs reach higher; default 2.0)")
    ap.add_argument("--oncoming-factor", type=float, default=ONCOMING_FACTOR,
                    help="multiply confidence by this for opposite-direction cars (1.0 disables; default 0.30)")
    args = ap.parse_args()

    if args.reuse and os.path.isfile(SAMPLES_JSON):
        with open(SAMPLES_JSON) as fh:
            samples = [(s["feat"], s["tp"], s["w"]) for s in json.load(fh)]
        print(f"[reuse] {len(samples)} cached samples from {SAMPLES_JSON}")
    else:
        with open(args.labels) as fh:
            labels = json.load(fh)["clips"]
        print(f"[labels] {len(labels)} clips + {len(FP_ONLY_CLIPS)} FP-only clips")
        samples = build_dataset(labels)
        if not samples:
            print("[abort] no samples (build caches first: crossing_violation_test.py --cache-only)")
            return
        os.makedirs(cvt.OUT_DIR, exist_ok=True)
        with open(SAMPLES_JSON, "w") as fh:
            json.dump([{"feat": f, "tp": t, "w": w} for f, t, w in samples], fh)
        print(f"[saved] {SAMPLES_JSON}")

    feature_diagnostic(samples)
    # learned-fusion comparisons (which features actually help) — resolution is confounded, excluded
    fit_logreg(samples, ("distance", "angle_off", "src_res"))
    fit_logreg(samples, ("distance",))
    print("\n>>> PRODUCTION MODEL <<<")
    model, auc = fit_logreg(samples, PROD_FEATURES)

    os.makedirs(cvt.OUT_DIR, exist_ok=True)
    with open(MODEL_JSON, "w") as fh:
        json.dump({"model": model, "auc": auc,
                   "note": "logistic regression; confidence = P(real violation). resolution "
                           "excluded (confounded with FP-source clips in the validation set)."}, fh, indent=2)
    print(f"[saved] {MODEL_JSON}   (AUC={auc:.3f})")

    if args.calibrate_fp is not None:
        model = calibrate_for_fp(model, samples, args.calibrate_fp, args.gain)
        tuned_path = os.path.join(cvt.OUT_DIR, "confidence_model_tuned.json")
        with open(tuned_path, "w") as fh:
            json.dump({"model": model, "target_fp": args.calibrate_fp, "gain": args.gain,
                       "note": "FP-centred logistic: mean false-positive confidence shifted to target_fp."}, fh, indent=2)
        print(f"[saved] {tuned_path}")

    if args.search:
        for fp_w in (0.1, 0.3, 0.5, 1.0):
            search_user_objective(samples, fp_w=fp_w)

    if args.render is not None:
        with open(args.labels) as fh:
            labeled = list(json.load(fh)["clips"])
        targets = labeled if args.render in ([], ["ALL"]) else args.render
        if args.formula:
            with open(args.formula) as fh:
                saved = json.load(fh)
            def scorer(feat):
                return float(formula_search.apply_saved(saved, [feat["distance"]], [feat["angle_off"]])[0])
            print(f"\n[scorer] WINNER formula (search S={saved['S']:.2f}, "
                  f"TPconf={saved['tp_conf']:.2f} FPconf={saved['fp_conf']:.2f}):\n         {saved['formula_str']}")
        elif args.sigmoid:
            with open(args.sigmoid) as fh:
                sig = json.load(fh)
            p = sig["params"]
            def scorer(feat):
                return float(sigmoid_search.conf_vec(np.array([feat["distance"]]), np.array([feat["angle_off"]]),
                                                     p["d50"], p["k_d"], p["a50"], p["k_a"])[0])
            print(f"\n[scorer] two-sigmoid winner  d50={p['d50']} k_d={p['k_d']} a50={p['a50']} k_a={p['k_a']}"
                  f"  (S={sig['S']:.2f}, TPconf={sig['tp_conf']:.2f} FPconf={sig['fp_conf']:.2f})")
        else:
            def scorer(feat):
                return confidence_lr(feat, model)
            print(f"\n[scorer] {'FP-centred graded logistic' if args.calibrate_fp is not None else 'logistic-regression production model'}")
        if args.oncoming_factor != 1.0:                  # geometric opposite-direction down-weight
            _base = scorer
            def scorer(feat, _b=_base, _of=args.oncoming_factor):
                return _b(feat) * (_of if feat.get("oncoming") else 1.0)
            print(f"[scorer] x{args.oncoming_factor:g} opposite-direction (oncoming) factor applied")
        print(f"[render] confidence overlays for {len(targets)} clip(s)...")
        for prefix in targets:
            render_confidence(prefix, scorer, hold_sec=args.hold_sec)


if __name__ == "__main__":
    main()
