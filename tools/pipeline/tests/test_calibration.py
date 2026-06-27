"""
Unit tests for the handshake calibration grid-search (calibrate_handshake.py).

Pure-Python: builds in-memory caches in the legacy schema and a ground-truth links dict, then
asserts the (gap_frames x iou_threshold) evaluation maps outcomes to TP/FP/FN/TN correctly.
No cache files, no model files, no GPU.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import calibrate_handshake as cal   # noqa: E402

W, H = 1000, 500


def _switch_cache():
    """track 7 dies at frame 100; track 8 is reborn at 104, displaced by a (15,0)/frame ego shift
    summed across 101..104 == 60 px, so the ego-compensated box exactly overlaps (IoU 1.0).
    The true predecessor of 8 is therefore 7, recoverable only once gap_frames >= 4."""
    frames = []
    for f in (98, 99, 100):
        frames.append({"frame": f, "shift": [0, 0],
                       "vehicles": [{"track_id": 7, "bbox": [100, 100, 200, 200]}]})
    for f in (101, 102, 103):
        frames.append({"frame": f, "shift": [15, 0], "vehicles": []})
    for i, f in enumerate((104, 105, 106)):
        frames.append({"frame": f, "shift": [15, 0] if i == 0 else [0, 0],
                       "vehicles": [{"track_id": 8, "bbox": [160, 100, 260, 200]}]})
    return {"w": W, "h": H, "fps": 30.0, "frames": frames}


# --------------------------------------------------------------------------- #
# evaluate_clip: each cell of the confusion matrix
# --------------------------------------------------------------------------- #
def test_true_link_recovered_when_gap_large_enough():
    cache, gt = _switch_cache(), {"links": {"8": 7}}
    # gap (104-100)=4 -> recovered at gap_frames>=4 for any swept IoU (ego IoU == 1.0)
    assert cal.evaluate_clip(cache, gt, gap_frames=4, iou_threshold=0.8) == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}
    assert cal.evaluate_clip(cache, gt, gap_frames=5, iou_threshold=0.9) == {"tp": 1, "fp": 0, "fn": 0, "tn": 0}


def test_true_link_missed_when_gap_too_small():
    cache, gt = _switch_cache(), {"links": {"8": 7}}
    # gap window [1,2] excludes the real gap of 4 -> false negative
    assert cal.evaluate_clip(cache, gt, gap_frames=2, iou_threshold=0.8) == {"tp": 0, "fp": 0, "fn": 1, "tn": 0}


def test_false_assignment_counts_as_fp():
    cache, gt = _switch_cache(), {"no_link": [8]}
    # 8 should NOT inherit, but at gap>=4 the geometry links it to 7 -> false positive
    assert cal.evaluate_clip(cache, gt, gap_frames=4, iou_threshold=0.8) == {"tp": 0, "fp": 1, "fn": 0, "tn": 0}


def test_correct_rejection_counts_as_tn():
    cache, gt = _switch_cache(), {"no_link": [8]}
    # gap window too small to link -> correct rejection
    assert cal.evaluate_clip(cache, gt, gap_frames=2, iou_threshold=0.8) == {"tp": 0, "fp": 0, "fn": 0, "tn": 1}


def test_high_iou_threshold_blocks_weak_overlap():
    # shrink the shift so the ego-compensated overlap is partial (IoU ~0.74), below 0.9
    cache = _switch_cache()
    for fr in cache["frames"]:
        if fr["frame"] in (101, 102, 103, 104) and fr["shift"] != [0, 0]:
            fr["shift"] = [11, 0]            # sum 44 -> imperfect overlap
    gt = {"links": {"8": 7}}
    assert cal.evaluate_clip(cache, gt, gap_frames=4, iou_threshold=0.9)["fn"] == 1   # too strict
    assert cal.evaluate_clip(cache, gt, gap_frames=4, iou_threshold=0.5)["tp"] == 1   # lenient recovers


# --------------------------------------------------------------------------- #
# aggregation + grid search
# --------------------------------------------------------------------------- #
def test_evaluate_params_aggregates_across_clips():
    pair = (_switch_cache(), {"links": {"8": 7}})
    agg = cal.evaluate_params([pair, pair], gap_frames=4, iou_threshold=0.8)
    assert agg == {"tp": 2, "fp": 0, "fn": 0, "tn": 0}


def test_grid_search_recommends_recovering_combo():
    caches_gt = [(_switch_cache(), {"links": {"8": 7}})]
    rows = cal.grid_search(caches_gt)
    assert len(rows) == len(cal.GAP_SWEEP) * len(cal.IOU_SWEEP)
    best = rows[0]                                # sorted best-first by F1
    assert best["f1"] == 1.0 and best["tp"] == 1 and best["fp"] == 0
    assert best["gap_frames"] >= 4               # only large-enough gaps recover the link
    # a combo with gap too small must score F1 0
    small = next(r for r in rows if r["gap_frames"] <= 2)
    assert small["f1"] == 0.0
