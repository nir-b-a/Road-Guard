#!/usr/bin/env python3
"""
check_heading_accuracy.py -- how accurate is the gyro heading the pipeline computes?

WHAT IT CHECKS
    speed_estimation/ego_yaw.ego_heading_from_android integrates the gravity-projected
    gyro yaw rate into a per-frame heading (anchored to 0 at frame 0), and main.py feeds
    that heading into the world-frame reconstruction. This tool calls the SAME function
    with the same arguments and scores its output against GPS. Nothing is reimplemented.

    Truth is the GPS course over ground, usable only while moving (>= --min-speed):
      bearing  the provider's reported bearing_deg (0.0 in gps.csv means "no bearing")
      track    course from the fix positions (central difference over both neighbours)
    Both are drawn on the timeline; --truth picks the one that is scored.

    The gyro heading has an arbitrary start, so exactly ONE constant offset is removed
    (median over the first --align-seconds of scored fixes, or over the whole drive with
    --align all). Nothing else is fitted: no scale, no sign flip, no time lag unless you
    pass --gps-lag-s yourself.

    Four views, from the most to the least demanding:
      1. absolute heading error over time  -> accumulated drift (deg/min)
      2. heading CHANGE over short windows  -> turn direction, turn size, local noise;
         drift-free, and the horizon the speed estimator actually differentiates over
      3. drift while the car is stopped     -> gyro bias, needs no GPS course at all
      4. gyro stream health                 -> sample rate, gaps, and frames that got no
                                               gyro sample (ego_yaw gives those yaw rate 0)
    Plus a GPS-latency search, reported only.

SIGN CONVENTION -- everything shown is COMPASS degrees: 0 = north, clockwise positive
    heading error = gyro - GPS; + means the gyro points clockwise (right) of the GPS course.
    With a correct Constants.HEADING_SIGN the heading-change scale is ~ +1.

OUTPUTS (--out, default <session>/heading_check/)
    heading_timeline.png   heading / error / turn rate / speed, one shared time axis
    heading_changes.png    gyro vs GPS heading change, one panel per --windows entry
    heading_map.png        GPS track in metres with GPS-course and gyro-heading arrows
    heading_per_fix.csv    one row per GPS fix, with the nearest video frame
    heading_report.json    every number printed to the console

RUN (from the malshinon/ directory)
    python tools/check_heading_accuracy.py test_videos/chase_vid
    python tools/check_heading_accuracy.py SESSION --truth track --align all
    python tools/check_heading_accuracy.py SESSION --gps-lag-s 0.5 --show
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # malshinon/
sys.path.insert(0, ROOT)

import Constants                                    # noqa: E402
from speed_estimation import ego_yaw                # noqa: E402

EARTH_R_M = 6371008.8
MIN_VALID_FIXES = 5          # fewer scored fixes than this -> GPS scoring is skipped
MAX_FIX_GAP_S = 2.5          # scored fixes further apart than this start a new segment
TRACK_MIN_CHORD_M = 3.0      # a position-derived course needs at least this much travel
LAG_SEARCH_S = 2.0           # GPS latency searched over +/- this
LAG_STEP_S = 0.05
STATIONARY_SPEED_MPS = 0.5   # interpolated GPS speed below this counts as stopped
STATIONARY_TRIM_S = 1.0      # dropped at both ends of a stop (GPS speed lags the car)
MIN_STATIONARY_S = 5.0       # shorter stops (after trimming) are ignored
GAP_FACTOR = 5.0             # a gyro gap is dt > GAP_FACTOR x median dt ...
GAP_MIN_MS = 10.0            # ... and at least this long
ERR_FLAG_DEG = 10.0          # the map rings fixes whose |heading error| reaches this


# ============================================================================
# Small helpers
# ============================================================================

def wrap180(a):
    """Wrap degrees to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def unwrap_deg(a):
    return np.degrees(np.unwrap(np.radians(np.asarray(a, dtype=float))))


def nearest_index(sorted_vals: np.ndarray, queries: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(sorted_vals, queries), 1, len(sorted_vals) - 1)
    left_closer = np.abs(queries - sorted_vals[idx - 1]) <= np.abs(sorted_vals[idx] - queries)
    return np.where(left_closer, idx - 1, idx)


def _maybe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _f(v):
    """float or None -- JSON-safe."""
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def _p(v, fmt=".2f"):
    return "n/a" if v is None or not np.isfinite(v) else format(float(v), fmt)


# ============================================================================
# Loading
# ============================================================================

def pipeline_heading(session: str, heading_sign: int) -> dict:
    """Per-frame heading from the pipeline's own ego_heading_from_android, in compass deg."""
    frames_csv = os.path.join(session, "frames.csv")
    gyro_csv = os.path.join(session, "gyro.csv")
    gravity_csv = os.path.join(session, "gravity.csv")
    for p in (frames_csv, gyro_csv, gravity_csv, os.path.join(session, "gps.csv")):
        if not os.path.exists(p):
            raise SystemExit(f"[heading] missing required stream: {p}")

    heading = ego_yaw.ego_heading_from_android(frames_csv, gyro_csv, gravity_csv,
                                               sign=heading_sign)
    frame_ts = ego_yaw.load_frame_timestamps(frames_csv)
    frames = sorted(frame_ts)
    if len(frames) < 2:
        raise SystemExit("[heading] frames.csv has fewer than 2 usable rows")

    t_ns = np.array([frame_ts[f] for f in frames], dtype=np.int64)
    math_rad = np.array([heading.get(f, 0.0) for f in frames], dtype=float)
    # ego_yaw's heading is counter-clockwise positive when the sign is right (the world
    # frame uses x = cos, y = sin). Compass is clockwise from north: compass = 90 - math.
    # It is a running sum, so it is already continuous (never wrapped).
    return {
        "frames": np.asarray(frames, dtype=int),
        "t_ns": t_ns,
        "t_s": (t_ns - t_ns[0]) / 1e9,
        "compass": 90.0 - np.degrees(math_rad),
        "gyro_csv": gyro_csv,
    }


def load_gps(path: str, t0_ns: int) -> dict:
    """gps.csv -> time-sorted arrays; t_s is relative to the first video frame."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((int(r["timestamp_ns"]), float(r["lat"]), float(r["lon"]),
                             float(r["speed_mps"]), _maybe_float(r.get("bearing_deg")),
                             _maybe_float(r.get("accuracy_m"))))
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit(f"[heading] {path}: no usable GPS rows")
    rows.sort(key=lambda x: x[0])
    t_ns = np.array([x[0] for x in rows], dtype=np.int64)
    col = lambda i: np.array([x[i] for x in rows], dtype=float)   # noqa: E731
    return {"t_ns": t_ns, "t_s": (t_ns - t0_ns) / 1e9, "lat": col(1), "lon": col(2),
            "speed": col(3), "bearing": col(4), "accuracy": col(5)}


def to_local_en(lat, lon, lat0: float, lon0: float):
    """Equirectangular metres around (lat0, lon0) -- ample for directions over a few km."""
    e = EARTH_R_M * math.cos(math.radians(lat0)) * np.radians(np.asarray(lon) - lon0)
    n = EARTH_R_M * np.radians(np.asarray(lat) - lat0)
    return e, n


# ============================================================================
# GPS truth courses (compass deg, NaN where unusable)
# ============================================================================

def course_from_bearing(gps: dict, min_speed: float) -> np.ndarray:
    # Location.getBearing() returns exactly 0.0 when the fix has no bearing, so 0.0 is
    # treated as missing (a genuine due-north bearing of exactly 0.0 is lost, rarely).
    b = gps["bearing"]
    ok = np.isfinite(b) & (b != 0.0) & (gps["speed"] >= min_speed)
    return np.where(ok, b % 360.0, np.nan)


def course_from_track(gps: dict, e: np.ndarray, n: np.ndarray, min_speed: float) -> np.ndarray:
    t = gps["t_s"]
    c = np.full(len(t), np.nan)
    for k in range(1, len(t) - 1):
        if t[k + 1] - t[k - 1] > 2 * MAX_FIX_GAP_S or gps["speed"][k] < min_speed:
            continue
        de, dn = e[k + 1] - e[k - 1], n[k + 1] - n[k - 1]
        if math.hypot(de, dn) >= TRACK_MIN_CHORD_M:
            c[k] = math.degrees(math.atan2(de, dn)) % 360.0     # compass = atan2(east, north)
    return c


# ============================================================================
# Scoring
# ============================================================================

def evaluate(gyro: dict, gps: dict, truth_compass: np.ndarray, truth_name: str, args):
    """Absolute heading error vs one GPS course series. None if too few usable fixes."""
    ft = gyro["t_s"]
    gt = gps["t_s"] - args.gps_lag_s              # the instant each fix describes
    valid = (gt >= ft[0]) & (gt <= ft[-1]) & np.isfinite(truth_compass)
    vi = np.flatnonzero(valid)
    if len(vi) < MIN_VALID_FIXES:
        return None

    tv = gt[vi]
    truth = truth_compass[vi]
    gyro_at = np.interp(tv, ft, gyro["compass"])

    # gyro - truth, unwrapped over the drive so a slow drift is never folded back at +/-180
    e = unwrap_deg(wrap180(gyro_at - truth))
    if args.align == "all":
        win = np.ones(len(tv), dtype=bool)
    else:
        win = tv <= tv[0] + args.align_seconds
        win[:3] = True
    offset = float(np.median(e[win]))
    err = e - offset

    # Whole-turn shift that puts the aligned gyro on a readable 0-360 branch at the first
    # fix. It differs from `offset` only by multiples of 360, so err is unaffected.
    shift = offset + 360.0 * math.floor((gyro_at[0] - offset) / 360.0)
    gyro_at_aligned = gyro_at - shift

    # Truth course unwrapped step by step (each step spans ~1 s, so it can never really
    # exceed 180 deg) -- independent of the gyro. Segments break at gaps in scored fixes.
    seg = np.concatenate([[0], np.cumsum(np.diff(tv) > MAX_FIX_GAP_S)])
    truth_unw = np.concatenate([[0.0], np.cumsum(wrap180(np.diff(truth)))])

    ae = np.abs(err)
    k_max = int(np.argmax(ae))
    return {
        "truth": truth_name,
        "vi": vi, "t_s": tv, "truth_raw": truth, "seg": seg, "truth_unw": truth_unw,
        "gyro_at": gyro_at, "gyro_at_aligned": gyro_at_aligned,
        "gyro_aligned": gyro["compass"] - shift,      # per frame
        "truth_plot": gyro_at_aligned - err,          # truth on the gyro's 360 branch
        "err": err,
        "alignment": {
            "mode": args.align,
            "window_fixes": int(win.sum()),
            "window_s": float(tv[win][-1] - tv[win][0]),
            "offset_removed_deg": float(offset % 360.0),
        },
        "abs_error": {
            "n_fixes": int(len(tv)),
            "median_abs_deg": float(np.median(ae)),
            "p95_abs_deg": float(np.percentile(ae, 95)),
            "rms_deg": float(np.sqrt(np.mean(err ** 2))),
            "max_abs_deg": float(ae[k_max]),
            "max_at_t_s": float(tv[k_max]),
            "final_deg": float(err[-1]),
        },
        "drift": drift_fit(tv, err),
    }


def drift_fit(t_s: np.ndarray, err: np.ndarray) -> dict:
    """Straight line through the error: a steady gyro bias shows up as a clean ramp."""
    if len(t_s) < 3 or np.ptp(t_s) <= 0:
        return {"deg_per_min": None, "intercept_deg": None, "r2": None}
    t_min = t_s / 60.0
    slope, icpt = np.polyfit(t_min, err, 1)
    resid = err - (slope * t_min + icpt)
    ss = float(np.sum((err - err.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else float("nan")
    return {"deg_per_min": float(slope), "intercept_deg": float(icpt), "r2": _f(r2)}


def change_stats(ev: dict, window_s: float, min_turn_deg: float) -> dict:
    """Gyro heading change vs GPS course change over ~window_s, for every scored fix."""
    tv, seg = ev["t_s"], ev["seg"]
    tol = max(0.3, 0.25 * window_s)
    ks, js = [], []
    for k in range(len(tv)):
        cand = np.flatnonzero((seg == seg[k]) & (tv > tv[k]))
        if len(cand) == 0:
            continue
        j = int(cand[np.argmin(np.abs(tv[cand] - (tv[k] + window_s)))])
        if abs(tv[j] - tv[k] - window_s) <= tol:
            ks.append(k)
            js.append(j)

    out = {"window_s": window_s, "n_pairs": len(ks), "n_turning": 0, "fit_on": None,
           "scale": None, "corr": None, "rms_all_deg": None, "rms_turning_deg": None,
           "median_abs_deg": None, "p95_abs_deg": None, "median_turn_error_pct": None,
           "d_gps": None, "d_gyro": None, "turn": None}
    if len(ks) < 3:
        out["verdict"] = sign_verdict(out)
        return out

    ks, js = np.asarray(ks), np.asarray(js)
    d_gyro = ev["gyro_at"][js] - ev["gyro_at"][ks]
    d_gps = ev["truth_unw"][js] - ev["truth_unw"][ks]
    resid = d_gyro - d_gps
    turn = np.abs(d_gps) >= min_turn_deg
    fit_turning = turn.sum() >= 3
    use = turn if fit_turning else np.ones(len(ks), dtype=bool)
    x, y = d_gps[use], d_gyro[use]
    sxx = float(np.sum(x * x))
    out.update({
        "n_turning": int(turn.sum()),
        "fit_on": "turning" if fit_turning else "all",
        # through the origin: a heading change of 0 must read 0
        "scale": float(np.sum(x * y) / sxx) if sxx > 0 else None,
        "corr": float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else None,
        "rms_all_deg": float(np.sqrt(np.mean(resid ** 2))),
        "rms_turning_deg": float(np.sqrt(np.mean(resid[turn] ** 2))) if turn.any() else None,
        "median_abs_deg": float(np.median(np.abs(resid))),
        "p95_abs_deg": float(np.percentile(np.abs(resid), 95)),
        "median_turn_error_pct": (float(np.median(np.abs(resid[turn]) / np.abs(d_gps[turn])))
                                  * 100.0 if turn.any() else None),
        "d_gps": d_gps, "d_gyro": d_gyro, "turn": turn,
    })
    out["verdict"] = sign_verdict(out)
    return out


def sign_verdict(st: dict) -> str:
    s, c = st["scale"], st["corr"]
    if st["n_pairs"] < 3 or s is None or c is None:
        return "inconclusive: not enough heading-change pairs"
    if st["n_turning"] < 3:
        return (f"inconclusive: only {st['n_turning']} turning window(s) -- "
                f"this needs a drive with turns")
    if abs(c) < 0.8:
        return f"WEAK: correlation {c:+.2f} is too low to judge turn direction or size"
    if s < 0:
        return (f"MIRRORED: the gyro turns the opposite way to the car (scale {s:+.2f}). "
                f"HEADING_SIGN is wrong for this data, and LAT_SIGN must be re-derived with it")
    if 0.9 <= s <= 1.1:
        return f"GOOD: gyro turns match GPS in direction and size (scale {s:.2f}, corr {c:.2f})"
    return (f"SCALE OFF: gyro turns are {s:.2f}x the GPS turns (corr {c:.2f}) -- check the "
            f"GPS lag, the gravity axis, or a phone moving in its mount")


def lag_search(gyro: dict, gps: dict, truth_compass: np.ndarray, args):
    """RMS of fix-to-fix heading-change residuals vs an assumed GPS latency. Report only.

    Positive lag = each GPS fix describes the car that many seconds BEFORE its timestamp.
    """
    ft, gt = gyro["t_s"], gps["t_s"]
    ok = (np.isfinite(truth_compass)
          & (gt >= ft[0] + LAG_SEARCH_S) & (gt <= ft[-1] - LAG_SEARCH_S))
    vi = np.flatnonzero(ok)
    if len(vi) < MIN_VALID_FIXES:
        return None
    t = gt[vi]
    pair = np.diff(t) <= MAX_FIX_GAP_S
    d_gps = wrap180(np.diff(truth_compass[vi]))
    turn = pair & (np.abs(d_gps) >= args.min_turn_deg)
    use = turn if turn.sum() >= 5 else pair
    if use.sum() < 3:
        return None

    lags = np.round(np.arange(-LAG_SEARCH_S, LAG_SEARCH_S + LAG_STEP_S / 2, LAG_STEP_S), 3)
    rms = np.empty(len(lags))
    for i, lag in enumerate(lags):
        g = np.interp(t - lag, ft, gyro["compass"])
        r = np.diff(g)[use] - d_gps[use]
        rms[i] = math.sqrt(float(np.mean(r * r)))
    best, zero = int(np.argmin(rms)), int(np.argmin(np.abs(lags)))
    return {"best_lag_s": float(lags[best]), "rms_at_best_deg": float(rms[best]),
            "rms_at_zero_deg": float(rms[zero]), "n_intervals": int(use.sum()),
            "on": "turning intervals" if use is turn else "all intervals",
            "lags_s": lags, "rms_deg": rms}


def stationary_drift(gyro: dict, gps: dict) -> dict:
    """Heading change while GPS says the car is stopped: pure gyro bias, no course needed."""
    ft = gyro["t_s"]
    speed = np.interp(ft, gps["t_s"], gps["speed"])
    still = (speed < STATIONARY_SPEED_MPS).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], still, [0]])))
    segs = []
    for a, b in zip(edges[0::2], edges[1::2]):                 # still frames a .. b-1
        t_a, t_b = ft[a] + STATIONARY_TRIM_S, ft[b - 1] - STATIONARY_TRIM_S
        if t_b - t_a < MIN_STATIONARY_S:
            continue
        m = (ft >= t_a) & (ft <= t_b)
        if m.sum() < 3:
            continue
        h = gyro["compass"][m]
        segs.append({
            "t_start_s": float(t_a), "t_end_s": float(t_b), "duration_s": float(t_b - t_a),
            "drift_deg_per_min": float(np.polyfit(ft[m], h, 1)[0] * 60.0),
            "net_change_deg": float(h[-1] - h[0]),
        })
    out = {"speed_threshold_mps": STATIONARY_SPEED_MPS, "n_segments": len(segs),
           "segments": segs}
    if segs:
        rates = np.array([s["drift_deg_per_min"] for s in segs])
        dur = np.array([s["duration_s"] for s in segs])
        out.update({
            "total_still_s": float(dur.sum()),
            "median_deg_per_min": float(np.median(rates)),
            "duration_weighted_mean_abs_deg_per_min": float(np.sum(np.abs(rates) * dur)
                                                            / dur.sum()),
            "max_abs_deg_per_min": float(np.max(np.abs(rates))),
        })
    return out


def gyro_health(gyro_csv: str, frame_t_ns: np.ndarray) -> dict:
    """Sample rate, gaps, and frames that received no gyro sample at all."""
    g_ts, _ = ego_yaw._read_xyz(gyro_csv, "timestamp_ns", ("gx", "gy", "gz"))
    out = {"n_samples": int(len(g_ts))}
    if len(g_ts) < 2:
        return out
    dt_ms = np.diff(g_ts) / 1e6
    pos = dt_ms[dt_ms > 0]
    med = float(np.median(pos)) if len(pos) else float("nan")
    thr = max(GAP_FACTOR * med, GAP_MIN_MS) if np.isfinite(med) else GAP_MIN_MS
    gaps = dt_ms[dt_ms > thr]

    # frame windows built exactly as ego_yaw_rate_from_android builds them
    t = frame_t_ns.astype(np.int64)
    mids = (t[:-1] + t[1:]) // 2
    lo, hi = np.empty_like(t), np.empty_like(t)
    lo[0] = t[0] - (mids[0] - t[0])
    lo[1:] = mids
    hi[-1] = t[-1] + (t[-1] - mids[-1])
    hi[:-1] = mids
    per_frame = np.searchsorted(g_ts, hi, side="left") - np.searchsorted(g_ts, lo, side="left")

    out.update({
        "rate_hz": _f(1000.0 / med) if np.isfinite(med) and med > 0 else None,
        "median_dt_ms": _f(med),
        "gap_threshold_ms": float(thr),
        "n_gaps": int(len(gaps)),
        "longest_dt_ms": float(dt_ms.max()),
        "time_in_gaps_s": float(gaps.sum() / 1000.0),
        "samples_per_frame_median": float(np.median(per_frame)),
        "frames_with_no_sample": int(np.sum(per_frame == 0)),
        "first_sample_after_first_frame_ms": float((g_ts[0] - t[0]) / 1e6),
        "last_sample_before_last_frame_ms": float((t[-1] - g_ts[-1]) / 1e6),
    })
    return out


# ============================================================================
# Plots
# ============================================================================
