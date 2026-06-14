"""
Hyperparameter calibration for the ego-compensated handshake (Module D).

Grid-searches (gap_frames x iou_threshold) for joiner.find_predecessor and scores each combo
against a GROUND-TRUTH LINKS file, because handshake correctness is about *identity* (did track
N correctly inherit the plate of the right predecessor?) -- something the violation-window labels
do not encode. The GT links file is small and hand-made from known ID-switch cases.

Pure-Python and model-free: it reads cached JSON (no cv2/torch) and reuses joiner +
build_track_index, so it runs in CI. Only the cache *files* must already exist (built on the GPU).

GT links file schema (gt_links.json):
  {
    "<clip_prefix>": {
        "links":   {"8": 7, "12": 10},   # new_track_id -> the CORRECT predecessor track_id
        "no_link": [15, 20]              # new_track_ids that must NOT inherit any plate
    },
    ...
  }

Cache schema consumed: the unchanged frame/shift/vehicles[track_id/bbox] layout.

Scoring per new_track:
  has a true predecessor (in "links"):  pred==truth -> TP | pred is None -> FN | pred==other -> FP
  must not link (in "no_link"):         pred is None -> TN | pred is not None    -> FP
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import joiner          # noqa: E402  find_predecessor (pure Python)
import lpr_consumer    # noqa: E402  build_track_index (pure Python)

_REPO = os.path.dirname(os.path.dirname(_PIPELINE_DIR))
DEFAULT_CACHE_DIR = os.path.join(_REPO, "outputs", "violation_cache")

GAP_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8]          # swept as the MAX gap (gap_min fixed at 1)
IOU_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9]


# --------------------------------------------------------------------------- #
# core evaluation (the part the unit tests exercise)
# --------------------------------------------------------------------------- #
def build_plate_map_all(cache: dict) -> dict:
    """Give every track a dummy plate so ANY track is an eligible predecessor -- the worst case
    for false assignments, which is exactly what calibration should stress."""
    return {tid: {"plate_candidate": f"plate_{tid}", "plate_confidence_score": 1.0}
            for tid in lpr_consumer.build_track_index(cache)}


def evaluate_clip(cache: dict, gt: dict, gap_frames: int, iou_threshold: float,
                  plate_map: dict | None = None) -> dict:
    """Confusion counts for one clip at one (gap_frames, iou_threshold) setting."""
    index = lpr_consumer.build_track_index(cache)
    plate_map = plate_map if plate_map is not None else build_plate_map_all(cache)
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    def predict(new_tid: int):
        pred = joiner.find_predecessor(cache, index, new_tid, plate_map,
                                       gap_min=1, gap_max=gap_frames, iou_threshold=iou_threshold)
        return pred[0] if pred else None

    for new_tid, true_old in {int(k): int(v) for k, v in gt.get("links", {}).items()}.items():
        pred_old = predict(new_tid)
        if pred_old == true_old:
            counts["tp"] += 1
        elif pred_old is None:
            counts["fn"] += 1
        else:
            counts["fp"] += 1                     # linked, but to the WRONG track

    for new_tid in (int(t) for t in gt.get("no_link", [])):
        counts["tn" if predict(new_tid) is None else "fp"] += 1

    return counts


def evaluate_params(caches_gt: list, gap_frames: int, iou_threshold: float) -> dict:
    """Aggregate confusion counts over all (cache, gt) pairs at one setting."""
    agg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for cache, gt in caches_gt:
        for k, v in evaluate_clip(cache, gt, gap_frames, iou_threshold).items():
            agg[k] += v
    return agg


def _prf(counts: dict) -> tuple[float, float, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def grid_search(caches_gt: list, gaps: list = GAP_SWEEP, ious: list = IOU_SWEEP) -> list:
    """Full grid; rows sorted best-first by F1, then fewest FP, then recall."""
    rows = []
    for gap in gaps:
        for thr in ious:
            counts = evaluate_params(caches_gt, gap, thr)
            precision, recall, f1 = _prf(counts)
            rows.append({"gap_frames": gap, "iou_threshold": thr, **counts,
                         "precision": round(precision, 4), "recall": round(recall, 4),
                         "f1": round(f1, 4)})
    rows.sort(key=lambda r: (r["f1"], -r["fp"], r["recall"]), reverse=True)
    return rows


# --------------------------------------------------------------------------- #
# IO + CLI
# --------------------------------------------------------------------------- #
def load_caches_with_gt(cache_dir: str, gt_path: str) -> list:
    with open(gt_path) as fh:
        gt_all = json.load(fh)
    pairs = []
    for prefix, gt in gt_all.items():
        path = os.path.join(cache_dir, f"{prefix}.json")
        if not os.path.isfile(path):
            print(f"[skip] no cache for {prefix!r} at {path}")
            continue
        with open(path) as fh:
            pairs.append((json.load(fh), gt))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the ego-compensated handshake (gap x IoU).")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    ap.add_argument("--gt-links", required=True, help="ground-truth ID-switch links JSON")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    caches_gt = load_caches_with_gt(args.cache_dir, args.gt_links)
    if not caches_gt:
        print("[abort] no (cache, gt) pairs found; build caches first and check --gt-links.")
        return

    rows = grid_search(caches_gt)
    n_links = sum(len(gt.get("links", {})) for _, gt in caches_gt)
    n_nolink = sum(len(gt.get("no_link", [])) for _, gt in caches_gt)

    print(f"\n=== handshake calibration   clips={len(caches_gt)} "
          f"true-links={n_links} must-not-link={n_nolink} ===")
    print(f"{'gap':>4} {'iou':>5} | {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3} | "
          f"{'prec':>5} {'rec':>5} {'F1':>5}")
    print("-" * 56)
    for r in rows[:args.top]:
        print(f"{r['gap_frames']:>4} {r['iou_threshold']:>5} | "
              f"{r['tp']:>3} {r['fp']:>3} {r['fn']:>3} {r['tn']:>3} | "
              f"{r['precision']:>5.2f} {r['recall']:>5.2f} {r['f1']:>5.2f}")

    best = rows[0]
    print("\n>>> RECOMMENDED sweet spot (max F1, then fewest false assignments):")
    print(f"    gap_frames = {best['gap_frames']}   iou_threshold = {best['iou_threshold']}")
    print(f"    -> TP={best['tp']} FP={best['fp']} FN={best['fn']}  "
          f"F1={best['f1']:.3f} (prec={best['precision']:.3f} rec={best['recall']:.3f})")
    print("    set joiner.GAP_MAX / joiner.IOU_THRESHOLD accordingly (GAP_MIN stays 1 unless "
          "you also sweep the lower bound).")


if __name__ == "__main__":
    main()
