"""
Stage-2 cascade validation runner.

Picks 10 videos (~6-7 min total) across three categories, runs main.py on each,
compares SOLID_LINE_CROSSING detections against baseline JSONs, and checks that
TP retention is >= 30%.  Produces annotated output videos for visual inspection.

Usage:
    python tools/run_cascade_validation.py [--no-stage2] [--out-dir PATH]

Categories selected:
    crossing_solid_line  (5 videos, all TP)
    solid_line_crossing_with_pl  (4 videos, all TP)
    normal_driving  (1 video truncated to 2 min, all FP for solid-line)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW  = REPO / "tests_videos" / "raw_videos"

# ── Video selection (≤15 min total) ──────────────────────────────────────────
VIDEOS = [
    # (category, subdir, video_file, expected_solid_TPs, max_frames or 0)
    # --- crossing_solid_line (5 videos, ~3 min total) ---
    ("crossing_solid_line",
     "58zh0ZyUTwU",
     "58zh0ZyUTwU.mp4",
     1, 0),
    ("crossing_solid_line",
     "Aps9SxIsVl8",
     "Aps9SxIsVl8.mp4",
     11, 0),
    ("crossing_solid_line",
     "mVvcgAA0rJ8",
     "mVvcgAA0rJ8.mp4",
     6, 0),
    ("crossing_solid_line",
     "S9kQqWl05KU",
     "S9kQqWl05KU.mp4",
     1, 0),
    ("crossing_solid_line",
     "yI0uJzyS-mQ",
     "yI0uJzyS-mQ.mp4",
     1, 0),
    # --- solid_line_crossing_with_pl (4 videos, ~1.5 min total) ---
    ("solid_line_crossing_with_pl",
     "0fr98nSc3CA",
     "0fr98nSc3CA.mp4",
     2, 0),
    ("solid_line_crossing_with_pl",
     "7E35VSQbAH8",
     "7E35VSQbAH8.mp4",
     1, 0),
    ("solid_line_crossing_with_pl",
     "8JerQB_w8D4",
     "8JerQB_w8D4.mp4",
     4, 0),
    ("solid_line_crossing_with_pl",
     "loXvNaf7plw",
     "loXvNaf7plw.mp4",
     3, 0),
    # --- normal_driving (1 video, truncated to 2 min = ~3600 frames at 30fps) ---
    ("normal_driving",
     "3KpV5GRLkJc_Driving_out_of_Jerusalem_Israel_Dash_Cam",
     "3KpV5GRLkJc_Driving_out_of_Jerusalem_Israel_Dash_Cam.f137.mp4",
     0, 3600),
]

MIN_TP_RETENTION = 0.30


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


def count_detected_tps(out_dir: Path, video_stem: str) -> int:
    csv_path = out_dir / f"{video_stem}_violations.csv"
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


def run_video(video_path: Path, out_dir: Path, max_frames: int, extra_args: list[str]) -> bool:
    python = sys.executable
    cmd = [python, str(REPO / "main.py"), str(video_path),
           "--out-dir", str(out_dir), "--simulation"]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    cmd += extra_args
    print(f"\n{'='*70}")
    print(f"  Running: {video_path.name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s  (exit={result.returncode})")
    return result.returncode == 0


def print_table(rows: list[dict]) -> None:
    cols = ["video", "category", "baseline_tp", "detected_tp", "retention%", "status"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    sep = "  ".join("-" * widths[c] for c in cols)
    hdr = "  ".join(c.ljust(widths[c]) for c in cols)
    print(f"\n{hdr}")
    print(sep)
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 cascade validation")
    ap.add_argument("--no-stage2", action="store_true",
                    help="Pass --no-stage2 to main.py (compare Stage-1 only baseline)")
    ap.add_argument("--out-dir", default=str(REPO / "validation_output"),
                    help="Root output directory (one subdir per video)")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    extra = ["--no-stage2"] if args.no_stage2 else []

    rows: list[dict] = []
    total_baseline_tp = 0
    total_detected_tp = 0
    all_pass = True

    for cat, subdir, fname, expected_tps, max_frames in VIDEOS:
        video_path = RAW / cat / subdir / fname
        if not video_path.exists():
            print(f"[SKIP] {fname} — file not found at {video_path}")
            continue

        video_stem = video_path.stem
        out_dir = out_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        ok = run_video(video_path, out_dir, max_frames, extra)

        baseline_path = find_baseline(cat, subdir)
        baseline_tp   = count_baseline_tps(baseline_path) if baseline_path else expected_tps
        detected_tp   = count_detected_tps(out_dir, video_stem)

        if baseline_tp > 0:
            retention = detected_tp / baseline_tp
            passed    = retention >= MIN_TP_RETENTION
            ret_str   = f"{retention*100:.0f}%"
        else:
            retention = None
            passed    = True   # normal_driving: 0 expected, any FPs are noted but not a failure
            ret_str   = "N/A (FP category)"

        status = "PASS" if (ok and passed) else "FAIL"
        if status == "FAIL":
            all_pass = False

        annotated = out_dir / f"{video_stem}_annotated.mp4"
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

        print(f"\n  baseline_tp={baseline_tp}  detected_tp={detected_tp}  "
              f"retention={ret_str}  [{status}]")
        if annotated.exists():
            print(f"  annotated video -> {annotated}")

    print_table(rows)

    if total_baseline_tp > 0:
        overall = total_detected_tp / total_baseline_tp
        print(f"\nOverall TP retention: {total_detected_tp}/{total_baseline_tp} "
              f"= {overall*100:.1f}%  (threshold: {MIN_TP_RETENTION*100:.0f}%)")
        if overall < MIN_TP_RETENTION:
            print("OVERALL FAIL: overall TP retention below 30%")
            all_pass = False

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
