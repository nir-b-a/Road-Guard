"""
Stage-2 cascade validation runner (in-process).

Loads all models ONCE, then calls process_video_with_models() directly for each video
— no subprocess overhead, no per-video model reload.  Estimated total runtime: ~15-30 min
on GPU vs 60-150 min with the previous subprocess approach.

Usage:
    C:/Users/talgx/miniconda3/envs/roadguard-dl/python.exe tools/run_cascade_validation.py
    C:/Users/talgx/miniconda3/envs/roadguard-dl/python.exe tools/run_cascade_validation.py --no-stage2
    C:/Users/talgx/miniconda3/envs/roadguard-dl/python.exe tools/run_cascade_validation.py --out-dir D:/val_out

Categories:
    crossing_solid_line       (5 videos, all TP)
    solid_line_crossing_with_pl  (4 videos, all TP)
    normal_driving            (1 video truncated to 2 min, all FP for solid-line)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_TOOLS = str(REPO / "tools")
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

RAW = REPO / "tests_videos" / "raw_videos"

# ── Video selection (≤15 min total) ──────────────────────────────────────────
# (category, subdir, video_file, expected_solid_TPs, max_frames or 0)
VIDEOS = [
    # --- crossing_solid_line (5 videos) ---
    ("crossing_solid_line",         "58zh0ZyUTwU",  "58zh0ZyUTwU.mp4",   1, 0),
    ("crossing_solid_line",         "Aps9SxIsVl8",  "Aps9SxIsVl8.mp4",  11, 0),
    ("crossing_solid_line",         "mVvcgAA0rJ8",  "mVvcgAA0rJ8.mp4",   6, 0),
    ("crossing_solid_line",         "S9kQqWl05KU",  "S9kQqWl05KU.mp4",   1, 0),
    ("crossing_solid_line",         "yI0uJzyS-mQ",  "yI0uJzyS-mQ.mp4",   1, 0),
    # --- solid_line_crossing_with_pl (4 videos) ---
    ("solid_line_crossing_with_pl", "0fr98nSc3CA",  "0fr98nSc3CA.mp4",   2, 0),
    ("solid_line_crossing_with_pl", "7E35VSQbAH8",  "7E35VSQbAH8.mp4",   1, 0),
    ("solid_line_crossing_with_pl", "8JerQB_w8D4",  "8JerQB_w8D4.mp4",   4, 0),
    ("solid_line_crossing_with_pl", "loXvNaf7plw",  "loXvNaf7plw.mp4",   3, 0),
    # --- normal_driving (1 video, truncated to ~2 min) ---
    ("normal_driving",
     "3KpV5GRLkJc_Driving_out_of_Jerusalem_Israel_Dash_Cam",
     "3KpV5GRLkJc_Driving_out_of_Jerusalem_Israel_Dash_Cam.f137.mp4",
     0, 3600),
]

MIN_TP_RETENTION = 0.30


# ── Baseline helpers ──────────────────────────────────────────────────────────

def find_baseline(cat: str, subdir: str) -> Path | None:
    d = RAW / cat / subdir
    for f in sorted(d.glob("*_baseline.json")):
        return f
    return None


def count_baseline_tps(baseline_path: Path) -> int:
    with open(baseline_path) as fh:
        data = json.load(fh)
    return sum(1 for v in data.get("violations", [])
               if v["violation_type"] == "SOLID_LINE_CROSSING")


def count_detected_tps(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    count = 0
    with open(csv_path) as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue  # header
            if "SOLID_LINE_CROSSING" in line:
                count += 1
    return count


# ── Table printer ─────────────────────────────────────────────────────────────

def print_table(rows: list[dict]) -> None:
    cols = ["video", "category", "baseline_tp", "detected_tp", "retention%", "status"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    hdr = "  ".join(c.ljust(widths[c]) for c in cols)
    print(f"\n{hdr}")
    print(sep)
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 cascade validation (in-process)")
    ap.add_argument("--no-stage2", action="store_true",
                    help="Run without the tire model (Stage-1 only baseline)")
    ap.add_argument("--out-dir", default=str(REPO / "validation_output"),
                    help="Root output directory (one subdir per video)")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Load all models ONCE ──────────────────────────────────────────────────
    import cloud_env
    import main as _main_mod

    cloud_env.init()

    print("\n[val] ── Loading YOLO model ──")
    yolo_model = _main_mod.loadYoloModel()

    lane_weights = str(REPO / "weights" / "phase3_v3_yellowprotect.pt")
    if Path(lane_weights).exists():
        print(f"\n[val] ── Loading lane model: {Path(lane_weights).name} ──")
        lane_model = _main_mod.loadLaneModel(lane_weights)
    else:
        print(f"\n[val] Lane weights not found ({lane_weights}); yellow+crossing disabled")
        lane_model = None

    tire_model = None
    if not args.no_stage2:
        tire_weights = str(REPO / "models" / "tire_yolo11n.pt")
        if Path(tire_weights).exists():
            print(f"\n[val] ── Loading tire model: {Path(tire_weights).name} ──")
            try:
                from violations.cascade.tire_model import TireModel
                tire_model = TireModel(tire_weights)
                print("[val] Tire model ready (Stage-2 ENABLED)")
            except Exception as e:
                print(f"[val] Tire model failed to load ({e}); Stage-2 disabled")
        else:
            print(f"[val] Tire weights not found ({tire_weights}); Stage-2 disabled")
    else:
        print("\n[val] --no-stage2: running Stage-1 only")

    # ── Process each video ────────────────────────────────────────────────────
    rows: list[dict] = []
    total_baseline_tp = 0
    total_detected_tp = 0
    all_pass = True

    for cat, subdir, fname, expected_tps, max_frames in VIDEOS:
        video_path = RAW / cat / subdir / fname
        if not video_path.exists():
            print(f"\n[SKIP] {fname} — file not found")
            continue

        video_stem = video_path.stem
        out_dir    = out_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  Video : {fname}")
        print(f"  Cat   : {cat}")
        if max_frames:
            print(f"  Limit : {max_frames} frames")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            result = _main_mod.process_video_with_models(
                str(video_path), yolo_model, lane_model, tire_model,
                out_dir=str(out_dir), max_frames=max_frames, is_simulation=True,
                benchmark=False,
            )
            ok = True
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            result = {}
            ok = False
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s")

        violations_csv = Path(result.get("violations_csv") or
                              out_dir / f"{video_stem}_violations.csv")
        annotated_mp4  = Path(result.get("annotated_video") or
                              out_dir / f"{video_stem}_annotated.mp4")

        # For FP-only categories (expected_tps == 0) never read baseline — those files
        # contain Stage-1 FPs which would inflate the denominator and distort retention %.
        if expected_tps > 0:
            baseline_path = find_baseline(cat, subdir)
            baseline_tp   = count_baseline_tps(baseline_path) if baseline_path else expected_tps
        else:
            baseline_tp = 0
        detected_tp   = count_detected_tps(violations_csv)

        if baseline_tp > 0:
            retention = detected_tp / baseline_tp
            passed    = retention >= MIN_TP_RETENTION
            ret_str   = f"{retention*100:.0f}%"
        else:
            retention = None
            passed    = True   # normal_driving: FPs noted but not a failure
            ret_str   = "N/A (FP)"

        # per-video: FAIL is informational only — exit code driven by overall retention
        status = "PASS" if (ok and passed) else "FAIL"

        rows.append({
            "video":       fname[:40],
            "category":    cat[:25],
            "baseline_tp": baseline_tp,
            "detected_tp": detected_tp,
            "retention%":  ret_str,
            "status":      status,
        })

        if baseline_tp > 0:
            total_baseline_tp += baseline_tp
            total_detected_tp += detected_tp

        print(f"  baseline_tp={baseline_tp}  detected_tp={detected_tp}  "
              f"retention={ret_str}  [{status}]")
        if annotated_mp4.exists():
            print(f"  annotated -> {annotated_mp4}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_table(rows)

    overall_ok = True
    if total_baseline_tp > 0:
        overall = total_detected_tp / total_baseline_tp
        print(f"\nOverall TP retention: {total_detected_tp}/{total_baseline_tp} "
              f"= {overall*100:.1f}%  (threshold: {MIN_TP_RETENTION*100:.0f}%)")
        if overall < MIN_TP_RETENTION:
            print("OVERALL FAIL: overall TP retention below 30%")
            overall_ok = False
        else:
            print("OVERALL PASS")
    else:
        print("\nNo TP baseline found; skipping overall retention check")

    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    if n_fail:
        print(f"({n_fail} individual video(s) below 30% — check per-video breakdown above)")

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
