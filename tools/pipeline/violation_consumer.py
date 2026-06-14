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
from ghost_mask import events_from_timeline  # noqa: E402  -- K-of-M event grouping

# Same K the confidence model was built/validated at (recall-first 0.05s -> frames per clip).
DEFAULT_K_SEC = vc.EVENT_K_SEC


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
                    prefix: str | None = None) -> list[dict]:
    """Run the ghost-mask verdict timeline + confidence over one cache and emit events shaped
    for the joiner: {violation_id, track_id, start_frame, end_frame, confidence}."""
    cache = cvt.ensure_shifts(cache)                       # geometry needs per-frame ego-motion
    timeline = cvt.clip_timeline(cache)
    kf = max(1, round(k_sec * cache["fps"]))
    events = events_from_timeline(timeline, kf)
    prefix = prefix or cache.get("prefix", "clip")

    out: list[dict] = []
    for ev in events:                                     # ev = (track_id, start_frame, end_frame)
        feat = vc.event_features(cache, ev)
        if feat is None:                                  # track vanished within its window
            continue
        tid, s, e = ev
        out.append({"violation_id": len(out),
                    "track_id": int(tid),
                    "start_frame": int(s),
                    "end_frame": int(e),
                    "confidence": float(vc.confidence_lr(feat, model))})
    return out


def run_violation_consumer(prefix: str, *, k_sec: float = DEFAULT_K_SEC,
                           model_path: str | None = None) -> list[dict]:
    """Load a prefix's cache + the confidence model and return its violation events."""
    cache = cvt.load_cache(prefix)
    if cache is None:
        raise FileNotFoundError(
            f"[violation_consumer] no cache for {prefix!r}; run the heavy pass first.")
    return find_violations(cache, load_confidence_model(model_path), k_sec=k_sec, prefix=prefix)
