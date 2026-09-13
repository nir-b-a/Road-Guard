#!/usr/bin/env python3
"""
speed_gt_eval.py -- score the pipeline's per-vehicle speed estimates against a
SECOND phone's GPS, recorded in the car being filmed.

The experiment: phone A films from the chase car, phone B rides in the lead car
and logs its own GNSS speed. Phone A's pipeline estimates a speed for EVERY vehicle
it tracks; one of those tracks is the lead car, and phone B's GPS says how fast that
car really went. So every track is scored against phone B's speed, a graph and a
statistics row are produced for each, and you identify the lead car by its track ID
in the annotated video -- its row is the answer, the rest are other vehicles that
happened to be on the road (their rows are meaningless as accuracy figures, though a
suspiciously good one is worth a look: cars in a queue move alike).

Nothing here assumes the two recordings started together. Both captures are placed
on absolute UTC through the GNSS anchor in their session_meta.json:

    utc_ms = clock_epoch_unix_ms + timestamp_ns / 1e6

and every estimate is compared against phone B's GPS interpolated AT THAT INSTANT.

Inputs:
  --chase DIR   phone A's capture folder, after `python main.py <video> --speed-only`
                (it must contain <video>_vehicle_speeds.csv and frames.csv)
  --lead  DIR   phone B's capture folder (gps.csv + session_meta.json). Its video is
                not needed -- only its GPS.

Outputs (in --out, default <chase>/speed_eval):
  speed_eval_summary.csv     one row per tracked vehicle, sorted best RMSE first
  veh_<id>_speed_vs_gps.png  estimated vs. real speed + the error over time
  speed_eval_overview.png    phone B's real speed with the closest tracks overlaid

Run from the malshinon/ directory:
    python tools/speed_gt_eval.py --chase real_vids/chase --lead real_vids/lead
"""

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless: write PNGs without a display
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # malshinon/
sys.path.insert(0, ROOT)

from speed_estimation.session_clock import (  # noqa: E402
    MPS_TO_KMH, SessionClockError, load_frame_times, load_gps,
    load_session_clock, utc_ms_to_iso,
)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def find_speeds_csv(chase_dir: str, override: str | None) -> str:
    """Locate <video>_vehicle_speeds.csv.

    --chase is the CAPTURE folder (it is where the clock anchor lives), but a run
    with --out-dir writes the CSV somewhere else, so --speeds is resolved as given
    first and only then relative to the capture folder.
    """
    if override:
        for p in (override, os.path.join(chase_dir, override)):
            if os.path.exists(p):
                return os.path.abspath(p)
        sys.exit(f"[eval] --speeds not found: {override}")
    cands = sorted(glob.glob(os.path.join(chase_dir, "*_vehicle_speeds.csv")))
    if not cands:
        sys.exit(f"[eval] no *_vehicle_speeds.csv in {chase_dir}\n"
                 f"       run:  python main.py {chase_dir}/<video>.mp4 --speed-only\n"
                 f"       (if that run used --out-dir, point --speeds at the CSV it wrote)")
    if len(cands) > 1:
        sys.exit("[eval] several *_vehicle_speeds.csv; pass --speeds <name>:\n  "
                 + "\n  ".join(os.path.basename(c) for c in cands))
    return cands[0]


def load_estimates(speeds_csv: str, chase_dir: str) -> dict[int, dict]:
    """vehicle_id -> {utc_ms, kmh, std_kmh, frame, cx, w} as numpy arrays.

    Prefers the unix_ms column main.py already stamped. If it is blank (a capture
    whose anchor was unavailable at pipeline time) the UTC is recomputed here from
    frames.csv + the folder's anchor, so a re-run of main.py is not required.
    """
    rows_by_vid: dict[int, list] = {}
    need_recompute = False
    with open(speeds_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                vid = int(r["vehicle_id"])
                frame = int(r["frame"])
                kmh = float(r["speed_kmh"])
            except (KeyError, ValueError, TypeError):
                continue
            try:
                utc = float(r["unix_ms"])
            except (KeyError, ValueError, TypeError):
                utc = float("nan")
                need_recompute = True
            try:
                std = float(r.get("speed_std_kmh") or 0.0)
            except ValueError:
                std = 0.0
            try:
                x1, x2 = float(r["x1"]), float(r["x2"])
                cx, w = (x1 + x2) / 2.0, x2 - x1
            except (KeyError, ValueError, TypeError):
                cx = w = float("nan")
            rows_by_vid.setdefault(vid, []).append(
                (frame, utc, kmh, std, cx, w, r.get("vehicle_class", "")))

    if need_recompute:
        frames_csv = os.path.join(chase_dir, "frames.csv")
        if not os.path.exists(frames_csv):
            sys.exit(f"[eval] {speeds_csv} has no unix_ms and {frames_csv} is missing -- "
                     "the estimates cannot be placed on an absolute timeline")
        clock = load_session_clock(chase_dir)           # raises if there is no anchor
        ft = dict(load_frame_times(frames_csv))
        print(f"[eval] unix_ms column was blank; recomputed from {clock.anchor_file}")
        for vid, rows in rows_by_vid.items():
            rows_by_vid[vid] = [
                (fr, clock.to_utc_ms(ft[fr]) if fr in ft else utc, kmh, std, cx, w, cls)
                for fr, utc, kmh, std, cx, w, cls in rows]

    out: dict[int, dict] = {}
    for vid, rows in rows_by_vid.items():
        rows = [r for r in rows if not math.isnan(r[1])]
        if not rows:
            continue
        rows.sort()
        out[vid] = {
            "frame":   np.array([r[0] for r in rows], dtype=int),
            "utc_ms":  np.array([r[1] for r in rows], dtype=float),
            "kmh":     np.array([r[2] for r in rows], dtype=float),
            "std_kmh": np.array([r[3] for r in rows], dtype=float),
            "cx":      np.array([r[4] for r in rows], dtype=float),
            "w":       np.array([r[5] for r in rows], dtype=float),
            "class":   rows[0][6],
        }
    return out


def load_ground_truth(lead_dir: str, *, max_accuracy_m: float,
                      smooth_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(utc_ms, speed_kmh) from phone B's GPS.

    GNSS speed is Doppler-derived, not a difference of positions, so it is good to
    roughly 0.1-0.3 m/s and does NOT need the position accuracy to be tight -- hence
    accuracy filtering is off unless asked for. Optional moving-average smoothing is
    also off by default: it would hide real acceleration and flatter the estimator.
    """
    gps_csv = os.path.join(lead_dir, "gps.csv")
    if not os.path.exists(gps_csv):
        sys.exit(f"[eval] {gps_csv} not found -- --lead must be phone B's capture folder")
    clock = load_session_clock(lead_dir)
    fixes = load_gps(gps_csv)
    if max_accuracy_m > 0:
        before = len(fixes)
        fixes = [f for f in fixes if 0 < f.accuracy_m <= max_accuracy_m]
        print(f"[eval] GT fixes: kept {len(fixes)}/{before} with accuracy <= {max_accuracy_m} m")
    if not fixes:
        sys.exit("[eval] no usable GPS fixes in the lead capture")

    t = np.array([clock.to_utc_ms(f.timestamp_ns) for f in fixes], dtype=float)
    v = np.array([f.speed_mps * MPS_TO_KMH for f in fixes], dtype=float)

    if smooth_s > 0 and len(v) > 2:
        dt = np.median(np.diff(t)) / 1000.0 or 1.0
        k = max(1, int(round(smooth_s / dt)))
        if k > 1:
            kernel = np.ones(k) / k
            v = np.convolve(v, kernel, mode="same")
            print(f"[eval] GT smoothed with a {k}-sample ({k * dt:.1f} s) moving average")
    return t, v


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def paired(est_t: np.ndarray, est_v: np.ndarray,
           gt_t: np.ndarray, gt_v: np.ndarray, *,
           max_gap_ms: float, lag_ms: float = 0.0):
    """Estimates paired with GT interpolated at the SAME instants (+ optional lag).

    Returns (mask, gt_at_est). Samples where GT is missing are masked out, never
    bridged: outside the GPS span, or inside a gap longer than max_gap_ms (a lost
    lock is not ground truth, and a straight line drawn across it would be scored
    as if it were).
    """
    q = est_t + lag_ms
    inside = (q >= gt_t[0]) & (q <= gt_t[-1])
    idx = np.clip(np.searchsorted(gt_t, q), 1, len(gt_t) - 1)
    gap_ok = (gt_t[idx] - gt_t[idx - 1]) <= max_gap_ms
    mask = inside & gap_ok
    gt_at = np.where(mask, np.interp(q, gt_t, gt_v), np.nan)
    return mask, gt_at


def stats_for(est_v: np.ndarray, gt_v: np.ndarray) -> dict:
    """The accuracy figures, all in km/h (MSE in (km/h)^2).

    bias  -- the systematic part: monocular geometry tends to OVER-estimate, so a
             positive bias here is the expected failure mode, and it is the part a
             calibration constant could remove.
    std_err = sqrt(rmse^2 - bias^2) -- what is left after removing that constant
             offset, i.e. the noise the estimator genuinely cannot explain.
    """
    err = est_v - gt_v
    n = len(err)
    bias = float(np.mean(err))
    mse = float(np.mean(err ** 2))
    rmse = math.sqrt(mse)
    moving = gt_v > 5.0                       # MAPE is meaningless near standstill
    mape = float(np.mean(np.abs(err[moving] / gt_v[moving])) * 100.0) if moving.any() else float("nan")
    if n > 2 and np.std(est_v) > 1e-9 and np.std(gt_v) > 1e-9:
        corr = float(np.corrcoef(est_v, gt_v)[0, 1])
    else:
        corr = float("nan")
    return {
        "n": n,
        "est_mean_kmh": float(np.mean(est_v)),
        "gt_mean_kmh": float(np.mean(gt_v)),
        "bias_kmh": bias,
        "mae_kmh": float(np.mean(np.abs(err))),
        "rmse_kmh": rmse,
        "mse_kmh2": mse,
        "std_err_kmh": math.sqrt(max(mse - bias ** 2, 0.0)),
        "median_abs_err_kmh": float(np.median(np.abs(err))),
        "p95_abs_err_kmh": float(np.percentile(np.abs(err), 95)),
        "max_abs_err_kmh": float(np.max(np.abs(err))),
        "mape_pct": mape,
        "corr": corr,
        "within_5_kmh_pct": float(np.mean(np.abs(err) <= 5.0) * 100.0),
        "within_10_kmh_pct": float(np.mean(np.abs(err) <= 10.0) * 100.0),
    }


def best_lag(est_t, est_v, gt_t, gt_v, *, max_gap_ms, search_s, step_s=0.1):
    """The time shift of the ESTIMATE that minimises RMSE, searched over +/-search_s.

    A DIAGNOSTIC, never applied to the reported numbers. If the GNSS anchors are
    right this lands near 0; a consistent non-zero value across several vehicles
    means either the two clocks are offset (a sync problem) or the estimator's
    smoothing delays its output (a pipeline property). A per-vehicle value that
    jumps around is just noise on a short track.
    """
    if search_s <= 0:
        return float("nan"), float("nan")
    best = (float("inf"), float("nan"))
    for lag in np.arange(-search_s, search_s + 1e-9, step_s):
        mask, gt_at = paired(est_t, est_v, gt_t, gt_v, max_gap_ms=max_gap_ms, lag_ms=lag * 1000.0)
        if mask.sum() < 10:
            continue
        rmse = math.sqrt(float(np.mean((est_v[mask] - gt_at[mask]) ** 2)))
        if rmse < best[0]:
            best = (rmse, float(lag))
    return best[1], best[0]


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_vehicle(vid, cls, t_s, est_v, est_std, gt_v, st, lag_s, out_png, video_clock_note,
                 gt_pts=None):
    """Two stacked panels: speed (estimated vs real) and the error over time.

    `gt_pts` are the RAW GPS fixes (~1 Hz) behind the smooth orange line, drawn so
    it is visible that the ground truth between them is interpolated, not measured.
    """
    fig, (ax, ax_e) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    ax.plot(t_s, gt_v, color="tab:orange", linewidth=2.2, label="Real (phone B GNSS)")
    if gt_pts is not None and len(gt_pts[0]):
        ax.plot(gt_pts[0], gt_pts[1], "o", ms=3.5, color="tab:orange",
                markeredgecolor="white", markeredgewidth=0.5, label="GNSS fixes", zorder=6)
    ax.plot(t_s, est_v, color="tab:blue", linewidth=1.6, alpha=0.9, label=f"Estimated (track {vid})")
    if np.any(est_std > 0):
        ax.fill_between(t_s, est_v - est_std, est_v + est_std, color="tab:blue",
                        alpha=0.18, label="+/-1 std")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Vehicle {vid} ({cls}) -- estimated vs. real speed")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    box = (f"n={st['n']}  span={t_s[-1] - t_s[0]:.1f}s\n"
           f"RMSE={st['rmse_kmh']:.2f} km/h   MSE={st['mse_kmh2']:.1f} (km/h)^2\n"
           f"MAE={st['mae_kmh']:.2f}   bias={st['bias_kmh']:+.2f}\n"
           f"std(err)={st['std_err_kmh']:.2f}   MAPE={st['mape_pct']:.1f}%\n"
           f"r={st['corr']:.3f}   within 5/10 km/h: "
           f"{st['within_5_kmh_pct']:.0f}%/{st['within_10_kmh_pct']:.0f}%")
    if not math.isnan(lag_s):
        box += f"\nbest lag={lag_s:+.1f}s (0 = clocks agree)"
    ax.text(0.985, 0.03, box, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="0.7"))

    err = est_v - gt_v
    ax_e.axhline(0, color="0.4", linewidth=1)
    ax_e.plot(t_s, err, color="tab:red", linewidth=1.2)
    ax_e.fill_between(t_s, 0, err, color="tab:red", alpha=0.18)
    ax_e.axhline(st["bias_kmh"], color="tab:purple", linestyle="--", linewidth=1,
                 label=f"bias {st['bias_kmh']:+.2f} km/h")
    ax_e.set_ylabel("Error (km/h)")
    ax_e.set_xlabel(f"Time ({video_clock_note})")
    ax_e.grid(True, alpha=0.4)
    ax_e.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_overview(gt_t_s, gt_v, tracks, out_png, video_clock_note):
    """Phone B's real speed with the best-matching tracks overlaid, so the lead
    car's trace stands out from the traffic around it."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(gt_t_s, gt_v, color="tab:orange", linewidth=2.6, label="Real (phone B GNSS)", zorder=5)
    cmap = plt.get_cmap("tab10")
    for i, (vid, t_s, v, rmse) in enumerate(tracks):
        ax.plot(t_s, v, linewidth=1.2, alpha=0.85, color=cmap(i % 10),
                label=f"track {vid} (RMSE {rmse:.1f})")
    ax.set_xlabel(f"Time ({video_clock_note})")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Estimated vehicle speeds vs. the lead car's real GNSS speed")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score per-vehicle speed estimates against a second phone's GPS.")
    ap.add_argument("--chase", required=True, help="phone A capture folder (the camera car)")
    ap.add_argument("--lead", required=True, help="phone B capture folder (the filmed car)")
    ap.add_argument("--out", default=None, help="output folder (default <chase>/speed_eval)")
    ap.add_argument("--speeds", default=None,
                    help="path to *_vehicle_speeds.csv when the run used --out-dir")
    ap.add_argument("--min-overlap-s", type=float, default=3.0,
                    help="skip tracks seen for less than this inside the GT window")
    ap.add_argument("--min-samples", type=int, default=15, help="skip tracks with fewer pairs")
    ap.add_argument("--gt-max-gap-s", type=float, default=2.5,
                    help="do not interpolate GPS across gaps longer than this")
    ap.add_argument("--gt-max-accuracy-m", type=float, default=0.0,
                    help="drop GPS fixes worse than this (0 = keep all)")
    ap.add_argument("--gt-smooth-s", type=float, default=0.0,
                    help="moving-average the GPS speed (0 = raw, recommended)")
    ap.add_argument("--lag-search-s", type=float, default=3.0,
                    help="+/- range for the sync diagnostic (0 = off)")
    ap.add_argument("--ids", default=None, help="only these track ids, comma separated")
    ap.add_argument("--top", type=int, default=8, help="tracks drawn on the overview plot")
    ap.add_argument("--no-plots", action="store_true", help="statistics only")
    args = ap.parse_args()

    chase_dir = os.path.abspath(args.chase)
    lead_dir = os.path.abspath(args.lead)
    out_dir = os.path.abspath(args.out or os.path.join(chase_dir, "speed_eval"))
    os.makedirs(out_dir, exist_ok=True)

    # ── clocks: both captures onto one absolute timeline ─────────────────────
    try:
        chase_clock = load_session_clock(chase_dir)
        lead_clock = load_session_clock(lead_dir)
    except SessionClockError as e:
        sys.exit(f"[eval] {e}")

    print("\n=== clocks ===")
    for tag, c in (("chase (A)", chase_clock), ("lead  (B)", lead_clock)):
        print(f"  {tag}: {c.describe()}")
        for w in c.warnings():
            print(f"      !! {w}")
    if not (chase_clock.is_gnss and lead_clock.is_gnss):
        print("  !! at least one anchor is NOT satellite-derived: the two recordings may be "
              "misaligned by up to a second or two, and every statistic below inherits that.")

    # ── data ─────────────────────────────────────────────────────────────────
    speeds_csv = find_speeds_csv(chase_dir, args.speeds)
    print(f"\n[eval] estimates : {speeds_csv}")
    est = load_estimates(speeds_csv, chase_dir)
    if not est:
        sys.exit("[eval] no vehicles with a speed series in that CSV")
    gt_t, gt_v = load_ground_truth(lead_dir, max_accuracy_m=args.gt_max_accuracy_m,
                                   smooth_s=args.gt_smooth_s)
    print(f"[eval] ground truth: {len(gt_t)} GPS fixes, "
          f"{utc_ms_to_iso(gt_t[0])} -> {utc_ms_to_iso(gt_t[-1])} "
          f"({(gt_t[-1] - gt_t[0]) / 1000.0:.1f} s)")

    # x-axis zero = the chase video's first frame, so a time on the graph is the
    # time to scrub to in <video>_annotated.mp4.
    frames_csv = os.path.join(chase_dir, "frames.csv")
    if os.path.exists(frames_csv):
        ft = load_frame_times(frames_csv)
        t0 = chase_clock.to_utc_ms(ft[0][1]) if ft else min(e["utc_ms"][0] for e in est.values())
    else:
        t0 = min(e["utc_ms"][0] for e in est.values())
    note = "s into the chase video"
    print(f"[eval] chase video starts {utc_ms_to_iso(t0)}; graphs are seconds from there")

    only = {int(x) for x in args.ids.split(",")} if args.ids else None
    max_gap_ms = args.gt_max_gap_s * 1000.0

    # ── per-vehicle scoring ──────────────────────────────────────────────────
    rows, overview = [], []
    skipped = 0
    for vid in sorted(est):
        if only and vid not in only:
            continue
        e = est[vid]
        mask, gt_at = paired(e["utc_ms"], e["kmh"], gt_t, gt_v, max_gap_ms=max_gap_ms)
        n = int(mask.sum())
        if n < args.min_samples:
            skipped += 1
            continue
        t_ms = e["utc_ms"][mask]
        span_s = (t_ms[-1] - t_ms[0]) / 1000.0
        if span_s < args.min_overlap_s:
            skipped += 1
            continue

        est_v, est_std, gt_paired = e["kmh"][mask], e["std_kmh"][mask], gt_at[mask]
        st = stats_for(est_v, gt_paired)
        lag_s, lag_rmse = best_lag(e["utc_ms"][mask], est_v, gt_t, gt_v,
                                   max_gap_ms=max_gap_ms, search_s=args.lag_search_s)

        t_s = (t_ms - t0) / 1000.0
        rows.append({
            "vehicle_id": vid, "vehicle_class": e["class"],
            "overlap_s": round(span_s, 2),
            "first_frame": int(e["frame"][mask][0]), "last_frame": int(e["frame"][mask][-1]),
            "video_t_start_s": round(float(t_s[0]), 2), "video_t_end_s": round(float(t_s[-1]), 2),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()},
            "best_lag_s": round(lag_s, 2) if not math.isnan(lag_s) else "",
            "rmse_at_best_lag_kmh": round(lag_rmse, 3) if not math.isnan(lag_rmse) else "",
            "mean_bbox_cx": round(float(np.nanmean(e["cx"][mask])), 1),
            "mean_bbox_w": round(float(np.nanmean(e["w"][mask])), 1),
        })
        overview.append((vid, t_s, est_v, st["rmse_kmh"]))

        if not args.no_plots:
            w = (gt_t >= t_ms[0]) & (gt_t <= t_ms[-1])
            plot_vehicle(vid, e["class"], t_s, est_v, est_std, gt_paired, st, lag_s,
                         os.path.join(out_dir, f"veh_{vid}_speed_vs_gps.png"), note,
                         gt_pts=((gt_t[w] - t0) / 1000.0, gt_v[w]))

    if not rows:
        sys.exit(f"[eval] no track overlapped the lead car's GPS for >= {args.min_overlap_s}s "
                 f"with >= {args.min_samples} samples. Do the two captures overlap in time? "
                 f"Run tools/sync_sessions.py to check.")

    rows.sort(key=lambda r: r["rmse_kmh"])

    # ── summary csv ──────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "speed_eval_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    if not args.no_plots:
        keep = {r["vehicle_id"] for r in rows[:max(args.top, 0)]}
        # Crop the GT trace to the window the tracks actually cover (+5 s of context),
        # so a 20-minute drive does not squash a 40-second comparison into one pixel.
        t_lo = t0 + min(o[1][0] for o in overview) * 1000.0 - 5000
        t_hi = t0 + max(o[1][-1] for o in overview) * 1000.0 + 5000
        gmask = (gt_t >= t_lo) & (gt_t <= t_hi)
        plot_overview((gt_t[gmask] - t0) / 1000.0, gt_v[gmask],
                      [o for o in overview if o[0] in keep],
                      os.path.join(out_dir, "speed_eval_overview.png"), note)

    # ── console table ────────────────────────────────────────────────────────
    print(f"\n=== {len(rows)} track(s) scored ({skipped} skipped: too short or no GPS overlap) ===")
    hdr = (f"{'id':>5} {'class':<10} {'n':>5} {'span_s':>7} {'video_t':>9} "
           f"{'RMSE':>7} {'MAE':>7} {'bias':>7} {'MSE':>8} {'r':>6} {'<5km/h':>7} {'lag_s':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['vehicle_id']:>5} {str(r['vehicle_class'])[:10]:<10} {r['n']:>5} "
              f"{r['overlap_s']:>7.1f} {r['video_t_start_s']:>9.1f} "
              f"{r['rmse_kmh']:>7.2f} {r['mae_kmh']:>7.2f} {r['bias_kmh']:>+7.2f} "
              f"{r['mse_kmh2']:>8.1f} {r['corr']:>6.3f} {r['within_5_kmh_pct']:>6.0f}% "
              f"{str(r['best_lag_s']):>6}")

    top = rows[0]
    print(f"\n[eval] closest match: track {top['vehicle_id']} ({top['vehicle_class']}), "
          f"RMSE {top['rmse_kmh']:.2f} km/h over {top['overlap_s']:.0f} s "
          f"(video {top['video_t_start_s']:.0f}-{top['video_t_end_s']:.0f} s). "
          f"A HINT only -- confirm the lead car's ID in the annotated video.")
    print(f"[eval] summary -> {csv_path}")
    if not args.no_plots:
        print(f"[eval] plots   -> {out_dir}")


if __name__ == "__main__":
    main()
