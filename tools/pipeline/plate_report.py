"""
Violation-driven per-car plate report.

For each clip: find the violations (cache-only, fast), then read plates ONLY for the violating
cars -- each on its largest frames -- and print, per violating car, its plate. Never OCRs a
non-violating vehicle, so it is seconds-per-clip instead of minutes (the run_lpr-over-everything
path stalled on busy clips). Optionally renders a red-box proof clip per violation.

Run (roadguard-dl env):
  python tools/pipeline/plate_report.py B7-3EuQFAZM 0SmdindPVEY DeNnDugXxP0 --render
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

import crossing_violation_test as cvt        # noqa: E402  load_cache
import violation_consumer                    # noqa: E402
import lpr_consumer                           # noqa: E402
import joiner                                 # noqa: E402
import alert_and_visualize as av              # noqa: E402  render_snippet
from run_pipeline import build_ocr_reader, DEFAULT_OUT   # noqa: E402


def report_clip(prefix: str, ocr_reader, *, k_sec: float, max_frames: int,
                render: bool, out_dir: str, render_min_conf: float = 0.6) -> dict | None:
    cache = cvt.load_cache(prefix)
    if cache is None:
        print(f"[skip] {prefix}: no cache (run a heavy pass first)")
        return None
    model = violation_consumer.load_confidence_model()
    violations = violation_consumer.find_violations(cache, model, k_sec=k_sec, prefix=prefix)
    if not violations:
        print(f"\n=== {prefix}: 0 violations ===")
        return {"prefix": prefix, "violations": []}

    viol_tracks = sorted({v["track_id"] for v in violations})
    if ocr_reader is not None:
        fp = lpr_consumer.make_video_frame_provider(cache["path"])
        plate_map = lpr_consumer.run_lpr_for_tracks(cache, viol_tracks, fp, ocr_reader,
                                                    lpr_consumer.laplacian_variance,
                                                    max_frames_per_track=max_frames)
    else:
        plate_map = {}
    joined = joiner.join_events(violations, plate_map, cache)

    print(f"\n=== {prefix}: {len(joined)} violation(s) across {len(viol_tracks)} car(s) ===")
    for v in joined:
        plate = v.get("plate_candidate") or "PLATE_UNKNOWN"
        print(f"  car#{v['track_id']:<4} frames {v['start_frame']:>5}-{v['end_frame']:<5} "
              f"viol={v['confidence']:.2f}  ->  {plate:<12} "
              f"(ocr={v.get('plate_confidence_score', 0.0):.2f}, {v.get('plate_source')})")
    if render:
        # one clip per violating CAR (its strongest segment), only if plated or a confident violation
        by_car: dict = {}
        for v in joined:
            by_car.setdefault(v["track_id"], []).append(v)
        for tid, segs in by_car.items():
            best = max(segs, key=lambda x: x["confidence"])
            if not (best.get("plate_candidate") or best["confidence"] >= render_min_conf):
                continue
            av.render_snippet(cache["path"], cache, best, out_dir, prefix, pad=15)
    return {"prefix": prefix, "violations": joined}


def main() -> None:
    ap = argparse.ArgumentParser(description="Violation-driven per-car plate report.")
    ap.add_argument("prefixes", nargs="+", help="clip prefixes to report on")
    ap.add_argument("--reader", default="fast_alpr", choices=["fast_alpr", "paddle", "none"])
    ap.add_argument("--k-sec", type=float, default=violation_consumer.DEFAULT_K_SEC)
    ap.add_argument("--max-frames", type=int, default=8, help="largest frames OCR'd per violating car")
    ap.add_argument("--render", action="store_true", help="also write red-box proof clips")
    ap.add_argument("--render-min-conf", type=float, default=0.6,
                    help="render a car's clip only if plated or its best violation >= this")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Hebrew/UTF-8 safe on Windows
    except (AttributeError, ValueError):
        pass

    os.makedirs(args.out, exist_ok=True)
    ocr_reader = build_ocr_reader(args.reader)
    report = []
    for prefix in args.prefixes:
        r = report_clip(prefix, ocr_reader, k_sec=args.k_sec, max_frames=args.max_frames,
                        render=args.render, out_dir=args.out, render_min_conf=args.render_min_conf)
        if r is not None:
            report.append(r)

    out_json = os.path.join(args.out, "plate_report.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    n_viol = sum(len(r["violations"]) for r in report)
    n_plated = sum(1 for r in report for v in r["violations"] if v.get("plate_candidate"))
    print(f"\n[done] {len(report)} clip(s), {n_viol} violation(s), {n_plated} with a plate -> {out_json}")


if __name__ == "__main__":
    main()
