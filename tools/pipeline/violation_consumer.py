"""
Module C -- Violation Consumer wrapper.

Thin wrapper over the existing, validated execution path:
    clip_timeline  ->  events_from_timeline  ->  event_features  ->  confidence_lr
It does NOT re-implement the ghost-mask rules or the logistic-regression math; it drives them
over a (BoT-SORT) cache and reshapes the result into violation events for the joiner (Module D).

The cache schema is passed straight through unchanged (frame/shift/contact, track_id/bbox).
"""
from __future__ import annotations

import json
import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_PIPELINE_DIR)
for _p in (_PIPELINE_DIR, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import crossing_violation_test as cvt        # noqa: E402  -- clip_timeline / ensure_shifts / load_cache
import violation_confidence as vc            # noqa: E402  -- event_features / confidence_lr / paths
import motion_filter as mf                   # noqa: E402  -- motion-based oncoming direction
from ghost_mask import events_from_timeline  # noqa: E402  -- K-of-M event grouping

# Same K the confidence model was built/validated at (recall-first 0.05s -> frames per clip).
DEFAULT_K_SEC = vc.EVENT_K_SEC

# Oncoming-direction handling. An oncoming car CAN legitimately cross the solid line (a valid
# violation), so we KEEP it -- but the geometric line-intersection is noisier from this perspective,
# so we DOWN-WEIGHT its confidence by ONCOMING_FACTOR (the ~0.30 penalty defined in
# violation_confidence.py). Direction is decided by MOTION per track (ego-compensated vertical
# velocity + y-origin anchor), which is robust where the static lane-side flag fails for close/
# edge/front-facing cars. Params validated for oncoming sensitivity in tools/wrong_way.py.
ONCOMING_MF_KWARGS = dict(vy_scale=0.18, w_vy=0.85, w_anchor=0.15, window=10, fusion="sum")
DEFAULT_ONCOMING_P = 0.25                       # mean motion P at/above which a track is "oncoming"
DEFAULT_ONCOMING_FACTOR = vc.ONCOMING_FACTOR    # confidence multiplier for oncoming (0.30); 1.0 disables


def classify_track_direction(cache: dict, *, mf_kwargs: dict | None = None) -> dict:
    """Per-track mean oncoming-motion probability in [0,1] (0 = same-direction, 1 = oncoming).
    Motion-only, so it works where the static lane-side geometry fails."""
    scores = mf.compute_motion_scores(cache["frames"], cache["h"],
                                      **(mf_kwargs or ONCOMING_MF_KWARGS))
    out = {}
    for tid, by_frame in scores.items():
        vals = list(by_frame.values())
        out[int(tid)] = (sum(vals) / len(vals)) if vals else 0.0
    return out


def load_confidence_model(path: str | None = None) -> dict:
    """Load the trained logistic-regression weights. The on-disk file is
    {"model": {...}, "auc": ...}; confidence_lr wants the inner `model` dict."""
    path = path or vc.MODEL_JSON
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[violation_consumer] confidence model missing: {path}. "
            f"Build it first with tools/violation_confidence.py.")
    with open(path) as fh:
        blob = json.load(fh)
    return blob["model"] if "model" in blob else blob


def find_violations(cache: dict, model: dict, *, k_sec: float = DEFAULT_K_SEC,
                    prefix: str | None = None,
                    oncoming_factor: float = DEFAULT_ONCOMING_FACTOR,
                    oncoming_p_threshold: float = DEFAULT_ONCOMING_P,
                    mf_kwargs: dict | None = None) -> list[dict]:
    """Run the ghost-mask verdict timeline + confidence over one cache and emit events shaped
    for the joiner: {violation_id, track_id, start_frame, end_frame, confidence}.

    Oncoming cars crossing the line ARE valid violations and are KEPT, but their confidence is
    multiplied by oncoming_factor (default 0.30) because the geometry is noisier from this
    perspective. Direction is classified by MOTION per track (>= oncoming_p_threshold = oncoming).
    Set oncoming_factor=1.0 to disable the penalty."""
    cache = cvt.ensure_shifts(cache)                       # geometry needs per-frame ego-motion
    timeline = cvt.clip_timeline(cache)
    kf = max(1, round(k_sec * cache["fps"]))
    events = events_from_timeline(timeline, kf)
    prefix = prefix or cache.get("prefix", "clip")

    penalize = oncoming_factor != 1.0
    onc = classify_track_direction(cache, mf_kwargs=mf_kwargs) if penalize else {}

    out: list[dict] = []
    downweighted = []
    for ev in events:                                     # ev = (track_id, start_frame, end_frame)
        tid, s, e = ev
        feat = vc.event_features(cache, ev)
        if feat is None:                                  # track vanished within its window
            continue
        conf = float(vc.confidence_lr(feat, model))
        if penalize and onc.get(int(tid), 0.0) >= oncoming_p_threshold:
            conf *= oncoming_factor                       # oncoming: kept, but down-weighted
            downweighted.append(int(tid))
        out.append({"violation_id": len(out),
                    "track_id": int(tid),
                    "start_frame": int(s),
                    "end_frame": int(e),
                    "confidence": conf})
    if downweighted:
        uniq = sorted(set(downweighted))
        print(f"[violation_consumer] {prefix}: down-weighted {len(downweighted)} oncoming-direction "
              f"event(s) x{oncoming_factor:g} on track(s) {uniq} (motion P>={oncoming_p_threshold})")
    return out


def run_violation_consumer(prefix: str, *, k_sec: float = DEFAULT_K_SEC,
                           model_path: str | None = None) -> list[dict]:
    """Load a prefix's cache + the confidence model and return its violation events."""
    cache = cvt.load_cache(prefix)
    if cache is None:
        raise FileNotFoundError(
            f"[violation_consumer] no cache for {prefix!r}; run the heavy pass first.")
    return find_violations(cache, load_confidence_model(model_path), k_sec=k_sec, prefix=prefix)
