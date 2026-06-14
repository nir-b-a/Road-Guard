"""
GPU-side cache rebuild + confidence re-validation + auto-refit (run in the roadguard-dl env).

We switched ByteTrack -> BoT-SORT, so every cached track id changed. The confidence model
is a logistic regression trained on ByteTrack-derived features, so it must be re-checked
against the new BoT-SORT features. This script:

  1. rebuilds each validation clip's cache with the BoT-SORT heavy pass (refresh),
  2. rebuilds the (features, is_tp) dataset from those caches via the existing build_dataset,
  3. loads the EXISTING LR weights (no refit) and scores the new features,
  4. reports the new ROC-AUC vs the stored ~0.78 and the delta,
  5. if the new AUC falls below a hard floor (default 0.75), AUTOMATICALLY refits a balanced
     LogisticRegression on the new BoT-SORT features and writes confidence_model_botsort.json
     (in the same standardize-then-linear format confidence_lr consumes).

Run:
  C:/Users/talgx/miniconda3/envs/roadguard-dl/python.exe tools/pipeline/rebuild_and_validate.py
  ...add --no-refresh to skip the (slow) rebuild and just re-score existing caches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_PIPELINE_DIR)
for _p in (_PIPELINE_DIR, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                       # noqa: E402
import crossing_violation_test as cvt    # noqa: E402
import violation_confidence as vc        # noqa: E402
import heavy_pass                        # noqa: E402

BOTSORT_MODEL_JSON = os.path.join(cvt.OUT_DIR, "confidence_model_botsort.json")


def compute_auc(y: list, scores: list) -> tuple[float, str]:
    """ROC-AUC via sklearn (the mandate); fall back to the repo's Mann-Whitney AUC if sklearn
    is not installed, so the script still runs on a lean env."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, scores)), "sklearn.roc_auc_score"
    except ImportError:
        return float(vc._auc(np.asarray(scores), np.asarray(y))), "fallback:mann-whitney"


def refit_confidence(samples: list, features: tuple | None = None,
                     out_path: str = BOTSORT_MODEL_JSON) -> tuple[dict, float, str]:
    """Refit a class-balanced logistic regression on the new BoT-SORT features and save it in the
    confidence_lr-compatible format ({features, mu, sd, coef, intercept}). Returns (model, auc,
    method). Uses sklearn.linear_model.LogisticRegression; falls back to the repo's own GD fit."""
    features = tuple(features) if features else tuple(vc.PROD_FEATURES)
    X = np.array([[f[k] for k in features] for f, _, _ in samples], dtype=float)
    y = np.array([1 if is_tp else 0 for _, is_tp, _ in samples])
    mu, sd = X.mean(0), X.std(0) + 1e-9            # standardize -> store mu/sd (matches fit_logreg)
    Xs = (X - mu) / sd

    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(class_weight="balanced", max_iter=5000)
        clf.fit(Xs, y)
        coef = clf.coef_[0].tolist()
        intercept = float(clf.intercept_[0])
        method = "sklearn.LogisticRegression"
    except ImportError:
        model_fb, _ = vc.fit_logreg(samples, features, verbose=False)  # GD fit, same output shape
        mu, sd = np.array(model_fb["mu"]), np.array(model_fb["sd"])
        coef, intercept = model_fb["coef"], float(model_fb["intercept"])
        method = "fallback:vc.fit_logreg"

    model = {"features": list(features), "mu": [float(v) for v in mu], "sd": [float(v) for v in sd],
             "coef": [float(c) for c in coef], "intercept": float(intercept)}
    auc, _ = compute_auc(list(y), [vc.confidence_lr(f, model) for f, _, _ in samples])

    os.makedirs(cvt.OUT_DIR, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"model": model, "auc": auc,
                   "note": f"BoT-SORT refit ({method}); confidence_lr-compatible "
                           f"(standardize-then-linear)."}, fh, indent=2)
    return model, auc, method


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild BoT-SORT caches + revalidate/refit confidence.")
    ap.add_argument("--labels", default=cvt.LABELS_JSON)
    ap.add_argument("--model", default=vc.MODEL_JSON)
    ap.add_argument("--no-refresh", action="store_true",
                    help="reuse existing caches (skip the slow BoT-SORT rebuild)")
    ap.add_argument("--auc-drop-warn", type=float, default=0.03,
                    help="warn if new AUC drops more than this below the stored AUC")
    ap.add_argument("--refit-threshold", type=float, default=0.75,
                    help="if new AUC < this floor, auto-refit the LR on the BoT-SORT features")
    ap.add_argument("--refit-out", default=BOTSORT_MODEL_JSON)
    args = ap.parse_args()

    with open(args.labels) as fh:
        labels = json.load(fh)["clips"]
    clips = list(labels) + vc.FP_ONLY_CLIPS
    print(f"[rebuild] {len(clips)} clips ({len(labels)} labeled + {len(vc.FP_ONLY_CLIPS)} FP-only)")

    # 1. rebuild caches under BoT-SORT
    if not args.no_refresh:
        for prefix in clips:
            try:
                heavy_pass.run_heavy_pass(prefix, refresh=True)
            except FileNotFoundError as e:
                print(f"  [skip] {e}")

    # 2. rebuild the labeled dataset from the (new) caches -- existing logic, unchanged
    samples = vc.build_dataset(labels)
    if not samples:
        print("[abort] no samples produced; were the caches built?")
        return

    # 3. load existing LR weights and score the new BoT-SORT features (no refit)
    with open(args.model) as fh:
        blob = json.load(fh)
    model = blob["model"] if "model" in blob else blob
    old_auc = blob.get("auc")
    scores = [vc.confidence_lr(feat, model) for feat, _, _ in samples]
    y = [1 if is_tp else 0 for _, is_tp, _ in samples]
    new_auc, method = compute_auc(y, scores)

    # 4. report
    print("\n" + "=" * 60)
    print(f"  confidence re-validation   ({method})")
    print(f"  samples={len(y)}  TP={sum(y)}  FP={len(y) - sum(y)}")
    if old_auc is not None:
        print(f"  old AUC (ByteTrack) = {old_auc:.3f}")
    print(f"  new AUC (BoT-SORT)  = {new_auc:.3f}")
    if old_auc is not None:
        delta = new_auc - old_auc
        print(f"  delta               = {delta:+.3f}")
        if delta < -args.auc_drop_warn:
            print(f"  [WARN] AUC dropped > {args.auc_drop_warn:.3f} below baseline.")

    # 5. auto-refit if the new AUC fell below the hard floor
    if new_auc < args.refit_threshold:
        print(f"\n[refit] new AUC {new_auc:.3f} < floor {args.refit_threshold:.3f} "
              f"-> retraining LR on BoT-SORT features...")
        model2, auc2, rmethod = refit_confidence(samples, out_path=args.refit_out)
        print(f"  method = {rmethod}")
        for k, c in zip(model2["features"], model2["coef"]):
            print(f"    coef[{k:10}] = {c:+.4f}")
        print(f"    intercept       = {model2['intercept']:+.4f}")
        print(f"    refit AUC       = {auc2:.3f}   (old {new_auc:.3f})")
        print(f"    saved -> {args.refit_out}")
    else:
        print(f"\n[keep] new AUC {new_auc:.3f} >= floor {args.refit_threshold:.3f} "
              f"-> existing model retained.")
    print("=" * 60)


if __name__ == "__main__":
    main()
