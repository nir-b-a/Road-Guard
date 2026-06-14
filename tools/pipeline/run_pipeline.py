"""
Road Guard offline pipeline orchestrator.

Flow:
  heavy_pass (cache)  ->  lpr_consumer (plates)  +  violation_consumer (violations)
                      ->  joiner (handshake join)  ->  hard_negatives (mine FPs/edges)
                      ->  write <out>/<prefix>_violations.json

The heavy pass and OCR load real models, so run this in the GPU env:
  C:/Users/talgx/miniconda3/envs/roadguard-dl/python.exe tools/pipeline/run_pipeline.py CLIP --refresh
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_PIPELINE_DIR)
_REPO = os.path.dirname(_TOOLS_DIR)
for _p in (_PIPELINE_DIR, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import heavy_pass            # noqa: E402  Module A
import lpr_consumer          # noqa: E402  Module B
import violation_consumer    # noqa: E402  Module C
import joiner                # noqa: E402  Module D
import hard_negatives        # noqa: E402  Module E

DEFAULT_OUT = os.path.join(_REPO, "outputs", "pipeline")


def build_ocr_reader(kind: str = "fast_alpr"):
    """Construct an OCR callable: crop -> (plate | None, confidence), or None for kind="none"
    (skip LPR entirely -> every plate is PLATE_UNKNOWN). Heavy imports are lazy so importing this
    module (e.g. for tests) never pulls in the OCR stack."""
    if kind == "none":
        return None
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)                          # the `lpr` package lives at repo root
    try:
        from lpr.reader import FastALPRReader, PaddleOCRDetectorReader
    except ImportError as e:
        raise RuntimeError(
            f"[pipeline] OCR backend unavailable ({e}). Run in the roadguard-dl env with "
            f"fast-alpr / paddleocr installed, or pass --reader none.") from e
    reader = {"fast_alpr": FastALPRReader, "paddle": PaddleOCRDetectorReader}[kind]()
    return lambda crop: reader.read_plate_with_conf(crop)


def run(prefix: str, *, out_dir: str = DEFAULT_OUT, k_sec: float = violation_consumer.DEFAULT_K_SEC,
        reader_kind: str = "fast_alpr", refresh: bool = False,
        video_path: str | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # 1. heavy pass -> per-frame cache (BoT-SORT)
    cache = heavy_pass.run_heavy_pass(prefix, video_path=video_path, refresh=refresh)

    # 2a. plates: cache-driven LPR over the real video frames (skipped if no OCR backend)
    ocr_reader = build_ocr_reader(reader_kind)
    if ocr_reader is None:
        print("[pipeline] no OCR backend (--reader none) -> plates will be PLATE_UNKNOWN")
        plate_map = {}
    else:
        frame_provider = lpr_consumer.make_video_frame_provider(cache["path"])
        plate_map = lpr_consumer.run_lpr(cache, frame_provider, ocr_reader,
                                         lpr_consumer.laplacian_variance)

    # 2b. violations: ghost-mask verdict timeline + confidence
    model = violation_consumer.load_confidence_model()
    violations = violation_consumer.find_violations(cache, model, k_sec=k_sec, prefix=prefix)

    # 3. join plate <-> violation (own plate, else ego-compensated handshake, else null)
    joined = joiner.join_events(violations, plate_map, cache)

    # 4. mine hard negatives for the active-learning loop
    hard = hard_negatives.log_hard_negatives(joined, out_dir, clip_prefix=prefix)

    payload = {"prefix": prefix, "n_violations": len(joined),
               "n_with_plate": sum(1 for v in joined if v.get("plate_candidate")),
               "n_hard_negatives": len(hard), "violations": joined}
    out_path = os.path.join(out_dir, f"{prefix}_violations.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[pipeline] {prefix}: {payload['n_violations']} violations "
          f"({payload['n_with_plate']} with plate), {len(hard)} hard-negatives -> {out_path}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Road Guard offline violation + LPR pipeline.")
    ap.add_argument("prefix", help="clip prefix (matched under the clips dir) or cache name")
    ap.add_argument("--video", default=None, help="explicit video path (overrides prefix lookup)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--k-sec", type=float, default=violation_consumer.DEFAULT_K_SEC)
    ap.add_argument("--reader", default="fast_alpr", choices=["fast_alpr", "paddle", "none"])
    ap.add_argument("--refresh", action="store_true", help="rebuild the cache under BoT-SORT")
    args = ap.parse_args()
    run(args.prefix, out_dir=args.out, k_sec=args.k_sec, reader_kind=args.reader,
        refresh=args.refresh, video_path=args.video)


if __name__ == "__main__":
    main()
