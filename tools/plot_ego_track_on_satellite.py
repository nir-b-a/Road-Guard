#!/usr/bin/env python3
"""
plot_ego_track_on_satellite.py -- georeference the reconstructed world-frame ego
pose and score it against GPS on a satellite basemap.

WHAT THIS ANSWERS
    speed_estimation/ego_yaw.py reconstructs a world-frame ego pose by integrating
    the gravity-projected gyro yaw rate into a heading and dead-reckoning speed
    along it. That reconstruction NEVER reads GPS lat/lon -- only speed_mps -- so
    its position has no feedback of any kind and its error is unbounded by
    construction. This tool asks: how far does it actually wander, and how fast
    does that error grow with time and distance?

    GPS lat/lon is the ground truth. It is imperfect (this repo's sample clip has a
    median accuracy of 3.8 m and a worst fix of 34 m), so every metric is reported
    alongside the reported accuracy of the fix it was measured against.

WHAT IT PLOTS -- and what it deliberately does NOT do
    The drawn track is the pipeline's OWN output: ego_yaw.ego_heading_from_android
    + ego_yaw.ego_position_from_android, called with the same arguments main.py
    uses, at the sign Constants.HEADING_SIGN actually holds. Nothing about the
    geometry is "fixed up" to look better:

      * alignment is a PROPER rotation only (det = +1) plus a translation anchor.
        A reflection is NEVER applied to the drawn track. The reflected fit's
        residual is computed and reported as a NUMBER (see the handedness
        diagnostic below) precisely so an inverted handedness shows up as a big
        error instead of being silently absorbed by the fit.
      * scale is LOCKED to 1. The track is already metric (it comes from GPS
        speed), so letting a Umeyama scale float would absorb genuine speed-bias
        drift into the alignment and hide it. Rotation + translation only.
      * --compare-heading-sign does NOT mirror the drawn track. It re-runs the
        real reconstruction with sign=-HEADING_SIGN, i.e. it shows what the
        PIPELINE would produce if the constant were flipped.

GEOREFERENCING
    Local ENU tangent plane at the anchor, using the WGS84 radii of curvature
    (accurate to well under a metre over several km):
        E = R_normal * cos(lat0) * dlon_rad        N = R_meridional * dlat_rad

    Rotation:    Kabsch/Umeyama with scale fixed at 1, fitted over the first
                 --align-seconds of MOTION (GPS fixes with speed >= --min-speed).
                 Only the early window is used: fitting over the whole drive would
                 let late drift rotate the start and smear the error uniformly,
                 which is the opposite of what we want to see.
    Translation: by default the track is pinned so that its position at the first
                 usable GPS fix coincides with that fix (--anchor first-fix), so
                 error starts at ~0 and every metre after that is accumulated
                 drift. --anchor kabsch instead best-fits the translation over the
                 window, which spreads the error (better-looking, less honest --
                 provided for comparison, not as the default).
    Overrides:   --origin-lat / --origin-lon / --heading-offset-deg.

HANDEDNESS DIAGNOSTIC
    The alignment reports the RMS residual of the proper-rotation fit AND of the
    reflected (det = -1) fit. If the reflected fit is dramatically better, the
    reconstruction's handedness is inverted and no rotation can ever line it up.
    The tool says so and points at --compare-heading-sign; it does not act on it.
    It also reports how collinear the fit window is: on a dead-straight window the
    two fits are indistinguishable and the diagnostic is meaningless.

OUTPUTS (written to --out, default: the session folder)
    ego_track_map.html      Leaflet + Esri World Imagery with roads and street
                            names on top (Google-Maps-style hybrid; satellite-only
                            and OpenStreetMap are one click away). No API key, no
                            dependency -- the page is generated from a template
                            here. Three trails: the free-running reconstruction,
                            the raw GPS track, and the GPS-anchored trail (below).
                            Plus time markers every --marker-interval-s and error
                            spokes joining reconstruction to truth at each marker.

THE GPS-ANCHORED TRAIL  (green)
    The reconstruction restarted at EVERY GPS fix: GPS supplies the anchor points,
    the reconstruction draws the ~1 s of shape between consecutive fixes. This
    removes accumulated drift by construction, which is the point -- the pipeline
    never uses the absolute track either. speed_estimator differentiates over a run
    of max(SMOOTH_WINDOW, DERIV_WINDOW) frames, so a per-interval error is far
    closer to its real error budget than the free-running final drift is.

    Only the POSITION is re-anchored. The rotation stays the one global solution;
    segments are NOT re-fitted onto their own GPS chords, because that would
    least-squares an inverted handedness away and make a broken sign look perfect.
    With position-only anchoring a wrong sign still shows: every segment bows the
    same way off its anchor, and mean_signed_cross_m stops being ~0.
    ego_track_metrics.csv   one row per GPS fix: position error, along/cross-track
                            split, heading error, distance travelled, GPS accuracy.
    ego_track_report.json   config, alignment, growth fits, per-decile error table.
    ego_track_error.png     error vs time, error vs distance (with growth fits),
                            heading error vs time.

RUN (from the malshinon/ directory)
    python tools/plot_ego_track_on_satellite.py real_vids/aba_video/vid1_undistort_calibrated
    python tools/plot_ego_track_on_satellite.py SESSION --compare-heading-sign
    python tools/plot_ego_track_on_satellite.py SESSION --heading-offset-deg 137 --anchor kabsch

NOTE the page fetches Leaflet and the Esri tiles from the network; everything
computed here is fully offline.
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

# WGS84
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


# ============================================================================
# Local ENU tangent plane
# ============================================================================

class LocalENU:
    """Metric east/north plane tangent to WGS84 at (lat0, lon0).

    Uses the two radii of curvature at the anchor rather than a single spherical
    radius: over a few km the difference is metres, which is the same order as the
    drift we are trying to measure.
    """

    def __init__(self, lat0_deg: float, lon0_deg: float):
        self.lat0 = float(lat0_deg)
        self.lon0 = float(lon0_deg)
        s = math.sin(math.radians(self.lat0))
        w = math.sqrt(1.0 - _WGS84_E2 * s * s)
        self.r_normal = _WGS84_A / w                                # prime vertical
        self.r_meridional = _WGS84_A * (1.0 - _WGS84_E2) / (w ** 3)  # meridional
        self.cos_lat0 = math.cos(math.radians(self.lat0))

    def to_enu(self, lat_deg, lon_deg):
        lat = np.asarray(lat_deg, dtype=float)
        lon = np.asarray(lon_deg, dtype=float)
        e = self.r_normal * self.cos_lat0 * np.radians(lon - self.lon0)
        n = self.r_meridional * np.radians(lat - self.lat0)
        return e, n

    def to_latlon(self, e, n):
        e = np.asarray(e, dtype=float)
        n = np.asarray(n, dtype=float)
        lat = self.lat0 + np.degrees(n / self.r_meridional)
        lon = self.lon0 + np.degrees(e / (self.r_normal * self.cos_lat0))
        return lat, lon


# ============================================================================
# Loading
# ============================================================================

def load_gps(path: str) -> dict:
    """gps.csv -> sorted arrays. bearing_deg is kept raw; validity is judged later."""
    ts, lat, lon, spd, brg, acc = [], [], [], [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                ts.append(int(row["timestamp_ns"]))
                lat.append(float(row["lat"]))
                lon.append(float(row["lon"]))
                spd.append(float(row["speed_mps"]))
            except (KeyError, ValueError):
                continue
            brg.append(_maybe_float(row.get("bearing_deg")))
            acc.append(_maybe_float(row.get("accuracy_m")))
    if not ts:
        raise SystemExit(f"[ego-map] {path}: no usable GPS rows")
    order = np.argsort(np.asarray(ts))
    return {
        "t_ns": np.asarray(ts, dtype=np.int64)[order],
        "lat": np.asarray(lat, dtype=float)[order],
        "lon": np.asarray(lon, dtype=float)[order],
        "speed": np.asarray(spd, dtype=float)[order],
        "bearing": np.asarray(brg, dtype=float)[order],
        "accuracy": np.asarray(acc, dtype=float)[order],
    }


def _maybe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def reconstruct(session: str, heading_sign: int, speed_source: str) -> dict:
    """Call the PIPELINE's own reconstruction. No reimplementation, no adjustment.

    speed_source: "auto" reproduces main.py exactly (fused accel+GPS when
    linacc.csv exists, GPS-only otherwise); "gps" forces the GPS-only path;
    "fused" requires linacc.csv.
    """
    frames_csv = os.path.join(session, "frames.csv")
    gyro_csv = os.path.join(session, "gyro.csv")
    gravity_csv = os.path.join(session, "gravity.csv")
    gps_csv = os.path.join(session, "gps.csv")
    linacc_csv = os.path.join(session, "linacc.csv")

    for p in (frames_csv, gyro_csv, gravity_csv, gps_csv):
        if not os.path.exists(p):
            raise SystemExit(f"[ego-map] missing required stream: {p}")

    has_linacc = os.path.exists(linacc_csv)
    if speed_source == "fused" and not has_linacc:
        raise SystemExit(f"[ego-map] --speed-source fused needs {linacc_csv}")
    use_linacc = linacc_csv if (speed_source in ("auto", "fused") and has_linacc) else None

    heading = ego_yaw.ego_heading_from_android(frames_csv, gyro_csv, gravity_csv,
                                               sign=heading_sign)
    pos = ego_yaw.ego_position_from_android(frames_csv, gps_csv, heading,
                                            linacc_csv=use_linacc,
                                            gravity_csv=gravity_csv if use_linacc else None)
    frame_ts = ego_yaw.load_frame_timestamps(frames_csv)
    frames = sorted(frame_ts)
    if len(frames) < 2:
        raise SystemExit("[ego-map] frames.csv has fewer than 2 usable rows")

    t_ns = np.array([frame_ts[f] for f in frames], dtype=np.int64)
    xy = np.array([pos.get(f, (0.0, 0.0)) for f in frames], dtype=float)
    h = np.array([heading.get(f, 0.0) for f in frames], dtype=float)
    return {
        "frames": np.asarray(frames, dtype=int),
        "t_ns": t_ns,
        "x": xy[:, 0], "y": xy[:, 1],
        "heading": np.unwrap(h),
        "speed_source": "fused" if use_linacc else "gps",
        "heading_sign": heading_sign,
    }


def sample_track(track: dict, t_ns) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstructed (x, y, heading) at arbitrary times, linearly interpolated.

    At 30 Hz the interpolation error is sub-centimetre, i.e. far below anything we
    are measuring. Heading is already unwrapped, so plain interpolation is safe.
    """
    t = np.asarray(t_ns, dtype=float)
    ft = track["t_ns"].astype(float)
    return (np.interp(t, ft, track["x"]),
            np.interp(t, ft, track["y"]),
            np.interp(t, ft, track["heading"]))


# ============================================================================
# Alignment  (rotation only -- scale locked, reflection reported not applied)
# ============================================================================

def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def kabsch_2d(P: np.ndarray, Q: np.ndarray) -> dict:
    """Best-fit 2-D rigid map from P (reconstruction) to Q (GPS ENU), scale = 1.

    Returns the PROPER rotation (det = +1) and, separately, the residual the
    REFLECTED fit would have achieved -- the handedness diagnostic. The reflected
    rotation is reported only; it is never returned as the transform to use.
    """
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    A, B = P - Pc, Q - Qc
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    V, Ut = Vt.T, U.T

    d = np.sign(np.linalg.det(V @ Ut)) or 1.0
    R_proper = V @ np.diag([1.0, d]) @ Ut
    R_reflect = V @ np.diag([1.0, -d]) @ Ut

    def rms(R):
        """RMS residual after mapping the centred reconstruction through R."""
        return float(np.sqrt(np.mean(np.sum((A @ R.T - B) ** 2, axis=1))))

    # Collinearity of the fit window: with a dead-straight window the two fits are
    # indistinguishable (a line is symmetric about itself) and the diagnostic below
    # carries no information. S[1]/S[0] near 0 means "straight".
    aniso = float(S[1] / S[0]) if S[0] > 0 else 0.0
    return {
        "R": R_proper,
        "theta_rad": float(math.atan2(R_proper[1, 0], R_proper[0, 0])),
        "rms_proper_m": rms(R_proper),
        "rms_reflected_m": rms(R_reflect),
        "centroid_src": Pc, "centroid_dst": Qc,
        "window_anisotropy": aniso,
        "n_points": int(len(P)),
    }


def solve_alignment(track: dict, gps: dict, enu, args, manual_origin: bool = False) -> dict:
    """Rotation (Kabsch over the first --align-seconds of motion) + translation."""
    gt = gps["t_ns"].astype(float)
    inside = (gt >= float(track["t_ns"][0])) & (gt <= float(track["t_ns"][-1]))
    moving = gps["speed"] >= args.min_speed
    usable = inside & moving
    if usable.sum() < 2:
        raise SystemExit(
            f"[ego-map] only {int(usable.sum())} GPS fixes are inside the video span "
            f"AND above --min-speed {args.min_speed} m/s -- cannot solve an alignment. "
            f"Lower --min-speed, or pass --heading-offset-deg explicitly.")

    t0 = gt[usable][0]
    window = usable & (gt >= t0) & (gt <= t0 + args.align_seconds * 1e9)
    widened = False
    if window.sum() < 2:                       # too few in the window -> use all motion
        window = usable
        widened = True

    ge, gn = enu.to_enu(gps["lat"][window], gps["lon"][window])
    px, py, _ = sample_track(track, gps["t_ns"][window])
    P = np.column_stack([px, py])
    Q = np.column_stack([ge, gn])

    if args.heading_offset_deg is not None:
        theta = math.radians(args.heading_offset_deg)
        fit = {"R": _rot(theta), "theta_rad": theta,
               "rms_proper_m": float("nan"), "rms_reflected_m": float("nan"),
               "window_anisotropy": float("nan"), "n_points": int(window.sum()),
               "source": "manual (--heading-offset-deg)"}
    else:
        fit = kabsch_2d(P, Q)
        fit["source"] = f"kabsch over first {args.align_seconds:.0f}s of motion"
    fit["window_fixes"] = int(window.sum())
    fit["window_widened"] = widened

    # ── translation ──────────────────────────────────────────────────────────
    R = fit["R"]
    if manual_origin:
        # "Override the start point": the reconstruction's FIRST FRAME is placed at
        # the supplied lat/lon. The tangent plane is centred there, so its ENU is
        # (0, 0) and the ego pose at frame 0 is (0, 0) by construction.
        x0, y0, _ = sample_track(track, np.array([track["t_ns"][0]], dtype=float))
        t_vec = -R @ np.array([x0[0], y0[0]])
        anchor_note = "reconstruction's first frame pinned to --origin-lat/--origin-lon"
    elif args.anchor == "kabsch" and args.heading_offset_deg is None:
        t_vec = fit["centroid_dst"] - R @ fit["centroid_src"]
        anchor_note = "kabsch best-fit translation over the alignment window"
    else:
        # Pin the track to the first usable fix: error starts at ~0 and everything
        # after it is accumulated drift, which is the quantity of interest. The
        # first fix overall is deliberately NOT used when the vehicle was parked --
        # a stationary fix has no course and is usually the worst fix of the drive
        # (this repo's sample clip opens with a 34 m one).
        i0 = int(np.flatnonzero(usable)[0])
        ax, ay, _ = sample_track(track, np.array([gps["t_ns"][i0]], dtype=float))
        ae, an = enu.to_enu(gps["lat"][i0], gps["lon"][i0])
        t_vec = np.array([float(ae), float(an)]) - R @ np.array([ax[0], ay[0]])
        skipped = "" if i0 == 0 else (f"; fixes #0-#{i0 - 1} were below --min-speed "
                                      f"{args.min_speed} m/s")
        anchor_note = (f"pinned to GPS fix #{i0} "
                       f"(accuracy {gps['accuracy'][i0]:.1f} m{skipped})")
        fit["anchor_fix_index"] = i0
        fit["anchor_fix_accuracy_m"] = float(gps["accuracy"][i0])
    fit["t"] = t_vec
    fit["anchor_note"] = anchor_note
    fit["usable_mask"] = usable
    return fit


def apply_alignment(fit: dict, x, y):
    P = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    W = P @ fit["R"].T + fit["t"]
    return W[:, 0], W[:, 1]


def gps_anchored_track(track: dict, gps: dict, enu: LocalENU, fit: dict,
                       mode: str = "position") -> dict | None:
    """GPS supplies the anchor points; the reconstruction supplies the shape between them.

    The dead reckoning is restarted at EVERY GPS fix, so position drift cannot
    accumulate and what remains is the error over ONE fix interval (~1 s here). That
    is much closer to the horizon speed_estimator actually integrates over -- it
    differentiates a run of max(SMOOTH_WINDOW, DERIV_WINDOW) frames and never sees the
    absolute track at all.

    mode="position"        re-anchor POSITION only; the heading stays whatever the
                           single global rotation gives it. Literal reading of "draw
                           the reconstruction between each pair of fixes", and it adds
                           no GPS truth beyond the anchor points -- but accumulated
                           HEADING drift still leaks in, so on a long drive every
                           segment can point the wrong way even though its start is
                           pinned. Read it as "local shape AND global heading drift".
    mode="position+course" additionally rotate each segment about its anchor so it
                           STARTS along the GPS course at that fix. Removes the
                           accumulated heading offset too, leaving the purely local
                           question: over one interval, does the reconstruction bend
                           the right way by the right amount?

    Neither mode ever fits a segment onto its own GPS chord. Re-fitting per segment
    would least-squares a mirrored handedness away and make a broken sign look
    perfect. Anchoring the START course does not: a mirrored yaw rate still curves the
    segment the wrong way over the interval, which is exactly what should stay visible.
    """
    gt = gps["t_ns"].astype(float)
    inside = (gt >= float(track["t_ns"][0])) & (gt <= float(track["t_ns"][-1]))
    idx = np.flatnonzero(inside)
    if len(idx) < 2:
        return None

    ft = track["t_ns"].astype(float)
    R, theta_g = fit["R"], fit["theta_rad"]
    ge, gn = enu.to_enu(gps["lat"][idx], gps["lon"][idx])

    # GPS course at each anchor, by central difference of the truth positions
    course = np.full(len(idx), np.nan)
    for k in range(len(idx)):
        a, b = max(0, k - 1), min(len(idx) - 1, k + 1)
        de_, dn_ = ge[b] - ge[a], gn[b] - gn[a]
        if math.hypot(de_, dn_) >= 0.5:
            course[k] = math.atan2(dn_, de_)

    segments, rows, n_course = [], [], 0
    for k in range(len(idx) - 1):
        t0, t1 = gt[idx[k]], gt[idx[k + 1]]
        # exact fix instants as endpoints, every video frame in between for the shape
        ts = np.concatenate([[t0], ft[(ft > t0) & (ft < t1)], [t1]])
        sx, sy, sh = sample_track(track, ts)
        P = np.column_stack([sx, sy]) @ R.T

        if mode == "position+course" and np.isfinite(course[k]):
            # extra spin about the anchor so the segment departs along the GPS course
            P = P @ _rot(course[k] - (sh[0] + theta_g)).T
            n_course += 1
        P = P - P[0] + np.array([ge[k], gn[k]])        # pin the START to this fix

        lat, lon = enu.to_latlon(P[:, 0], P[:, 1])
        segments.append(_latlon_pairs(lat, lon))

        # how far the propagation lands from where GPS says the next fix was
        de, dn = P[-1, 0] - ge[k + 1], P[-1, 1] - gn[k + 1]
        ce, cn = ge[k + 1] - ge[k], gn[k + 1] - gn[k]     # GPS chord for this interval
        chord = math.hypot(ce, cn)
        if chord >= 0.5:
            cu, su = ce / chord, cn / chord
            along, cross = de * cu + dn * su, -de * su + dn * cu
        else:
            along = cross = float("nan")
        rows.append({
            "fix_index": int(idx[k]),
            "t_s": float((gt[idx[k]] - track["t_ns"][0]) / 1e9),
            "dt_s": float((t1 - t0) / 1e9),
            "gps_chord_m": float(chord),
            "endpoint_error_m": float(math.hypot(de, dn)),
            "along_m": _f(along), "cross_m": _f(cross),
            "gps_accuracy_m": _f(gps["accuracy"][idx[k + 1]]),
        })

    err = np.array([r["endpoint_error_m"] for r in rows])
    chords = np.array([r["gps_chord_m"] for r in rows])
    dts = np.array([r["dt_s"] for r in rows])
    cross = np.array([r["cross_m"] if r["cross_m"] is not None else np.nan for r in rows])
    along = np.array([r["along_m"] if r["along_m"] is not None else np.nan for r in rows])
    moved = chords >= 0.5
    pct = 100.0 * err[moved] / chords[moved] if moved.any() else np.array([np.nan])

    return {
        "segments": segments,
        "per_segment": rows,
        "stats": {
            "mode": mode,
            "n_segments_course_anchored": n_course,
            "n_segments": int(len(rows)),
            "mean_interval_s": float(dts.mean()),
            "mean_gps_chord_m": float(chords.mean()),
            "endpoint_error_m": {
                "median": float(np.median(err)), "mean": float(err.mean()),
                "p95": float(np.percentile(err, 95)), "max": float(err.max()),
            },
            "endpoint_error_pct_of_chord_median": float(np.nanmedian(pct)),
            "mean_abs_along_m": float(np.nanmean(np.abs(along))),
            "mean_abs_cross_m": float(np.nanmean(np.abs(cross))),
            # a consistently one-signed cross error over many segments is a handedness
            # tell: a mirrored track bows the same wrong way off every anchor
            "mean_signed_cross_m": float(np.nanmean(cross)),
        },
    }


# ============================================================================
# Metrics
# ============================================================================

def _wrap_deg(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def _fit_through_origin(x, y, power: int):
    """Least-squares y = k * x**power through the origin, plus its R^2.

    Error is anchored at ~0 at t=0, so a through-origin fit is the right model.
    Comparing power=1 against power=2 tests the prediction that a constant gyro
    bias produces QUADRATIC position drift (error ~ 0.5*v*bias*t^2) while a speed
    scale error produces linear drift.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan")
    b = x ** power
    denom = float(np.sum(b * b))
    if denom <= 0:
        return float("nan"), float("nan")
    k = float(np.sum(b * y) / denom)
    resid = y - k * b
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return k, r2


def evaluate(track: dict, gps: dict, enu: LocalENU, fit: dict, args) -> dict:
    """Per-GPS-fix error series. Truth is the RAW fixes -- never interpolated."""
    gt = gps["t_ns"].astype(float)
    inside = (gt >= float(track["t_ns"][0])) & (gt <= float(track["t_ns"][-1]))
    n_outside = int((~inside).sum())
    idx = np.flatnonzero(inside)
    if len(idx) < 2:
        raise SystemExit("[ego-map] fewer than 2 GPS fixes fall inside the video span")

    t_ns = gps["t_ns"][idx]
    t_s = (t_ns - track["t_ns"][0]) / 1e9
    ge, gn = enu.to_enu(gps["lat"][idx], gps["lon"][idx])
    rx, ry, rh = sample_track(track, t_ns)
    re, rn = apply_alignment(fit, rx, ry)

    err_e, err_n = re - ge, rn - gn
    err = np.hypot(err_e, err_n)

    # GPS course from central differences of the truth positions; fall back to the
    # reported bearing when consecutive fixes are too close for a direction.
    course = np.full(len(idx), np.nan)
    for k in range(len(idx)):
        a, b = max(0, k - 1), min(len(idx) - 1, k + 1)
        de, dn = ge[b] - ge[a], gn[b] - gn[a]
        if math.hypot(de, dn) >= 0.5:
            course[k] = math.atan2(dn, de)
        else:
            brg = gps["bearing"][idx[k]]
            if np.isfinite(brg) and gps["speed"][idx[k]] >= args.min_speed and brg != 0.0:
                course[k] = math.radians(90.0 - brg)
    valid_course = np.isfinite(course)
    course_f = np.where(valid_course, course, 0.0)

    # along = along the direction of travel (+ = reconstruction runs ahead of truth)
    # cross = to the LEFT of the direction of travel
    cu, su = np.cos(course_f), np.sin(course_f)
    along = np.where(valid_course, err_e * cu + err_n * su, np.nan)
    cross = np.where(valid_course, -err_e * su + err_n * cu, np.nan)

    # distance travelled, measured on the TRUTH track
    step = np.hypot(np.diff(ge), np.diff(gn))
    dist = np.concatenate([[0.0], np.cumsum(step)])

    # ── heading drift ────────────────────────────────────────────────────────
    # Both series are UNWRAPPED before differencing, and the result is NOT wrapped
    # back. Wrapping to +/-180 would fold a large accumulated drift back toward
    # zero and make a badly drifting run look fine -- the exact opposite of what
    # this tool is for. An unbounded number here is the honest answer.
    recon_hdg = np.degrees(rh + fit["theta_rad"])
    gps_hdg = np.degrees(course)
    valid_h = valid_course & (gps["speed"][idx] >= args.min_speed)
    hdg_drift = np.full(len(idx), np.nan)
    hdg_bias = 0.0
    hdg_unwrap_gap_s = float("nan")
    vi = np.flatnonzero(valid_h)
    if len(vi) >= 2:
        gps_unw = np.degrees(np.unwrap(course[vi]))     # continuous GPS course
        recon_unw = np.degrees(rh[vi] + fit["theta_rad"])  # rh is already unwrapped
        raw = recon_unw - gps_unw
        # The absolute offset is arbitrary (heading is anchored to 0 at frame 0), so
        # it is removed over the alignment window; everything after that is drift.
        wmask = t_s[vi] <= t_s[vi][0] + args.align_seconds
        hdg_bias = float(np.median(raw[wmask])) if np.any(wmask) else float(raw[0])
        hdg_drift[vi] = raw - hdg_bias
        # Unwrapping assumes no >180 deg turn happened between two VALID fixes; a
        # long gap in usable course (a stop, a tunnel) could break that assumption.
        hdg_unwrap_gap_s = float(np.max(np.diff(t_s[vi]))) if len(vi) > 1 else 0.0

    probe = handedness_probe(course, valid_h, rh, args)

    lat_r, lon_r = enu.to_latlon(re, rn)
    return {
        "idx": idx, "t_ns": t_ns, "t_s": t_s,
        "gps_lat": gps["lat"][idx], "gps_lon": gps["lon"][idx],
        "gps_e": ge, "gps_n": gn,
        "gps_speed": gps["speed"][idx], "gps_accuracy": gps["accuracy"][idx],
        "recon_lat": lat_r, "recon_lon": lon_r, "recon_e": re, "recon_n": rn,
        "error_m": err, "along_m": along, "cross_m": cross,
        "dist_m": dist,
        "recon_heading_deg": _wrap_deg(recon_hdg),
        "gps_heading_deg": gps_hdg,
        "heading_err_deg": hdg_drift,
        "heading_bias_deg": hdg_bias,
        "heading_unwrap_max_gap_s": hdg_unwrap_gap_s,
        "handedness_probe": probe,
        "n_fixes_outside_video": n_outside,
    }


def handedness_probe(course: np.ndarray, valid: np.ndarray, recon_heading: np.ndarray,
                     args) -> dict:
    """Is the reconstruction's turn direction the same as reality's?

    Alignment-independent, and independent of where the drawn track ends up: it
    compares the CHANGE in reconstructed heading between consecutive usable GPS
    fixes against the change in the GPS course over the same interval, across the
    WHOLE drive. A best-fit scale near +1 means the reconstruction turns the same
    way as the vehicle; near -1 means it turns the opposite way, i.e. the
    handedness is inverted and the track is a mirror image that no rotation can
    align.

    This exists because the Kabsch reflection residual -- the other handedness
    check -- is undecidable on a straight alignment window (a line is symmetric
    about itself), which is common at the start of a drive. This probe only needs
    the drive to contain SOME turn, anywhere.

    Reports, never acts.
    """
    vi = np.flatnonzero(valid)
    out = {"n_turn_intervals": 0, "correlation": None, "best_scale": None,
           "verdict": "inconclusive: not enough usable GPS course samples"}
    if len(vi) < 3:
        return out

    d_gps = np.diff(np.unwrap(course[vi]))
    d_rec = np.diff(recon_heading[vi])          # recon_heading is already unwrapped
    m = np.abs(d_gps) > math.radians(args.min_turn_deg)
    out["n_turn_intervals"] = int(m.sum())
    if m.sum() < 3:
        out["verdict"] = (f"inconclusive: the drive contains only {int(m.sum())} interval(s) "
                          f"turning more than --min-turn-deg {args.min_turn_deg} deg -- "
                          f"drive a route with a turn in it to decide handedness")
        return out

    a, b = d_rec[m], d_gps[m]
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
    scale = float(np.linalg.lstsq(a[:, None], b, rcond=None)[0][0])
    out["correlation"] = None if not np.isfinite(corr) else corr
    out["best_scale"] = scale

    # A confident verdict needs BOTH a strong correlation AND a scale near +/-1. A
    # scale far from unit magnitude means the gyro's turn SIZE disagrees with GPS,
    # which is a different fault from a flipped sign -- and on a handful of
    # intervals it is just as likely to be noise. Say so rather than pick a side.
    ev = f"scale {scale:+.2f}, corr {corr:+.3f} over {int(m.sum())} turning intervals"
    strong = np.isfinite(corr) and abs(corr) >= 0.8
    unit_ish = 0.5 <= abs(scale) <= 2.0
    if strong and unit_ish and scale < 0:
        out["verdict"] = (f"INVERTED: reconstructed turns run OPPOSITE to reality ({ev}). "
                          f"The track is a mirror image -- no rotation can align it. "
                          f"Re-run with --compare-heading-sign to see what flipping "
                          f"Constants.HEADING_SIGN would produce.")
    elif strong and unit_ish and scale > 0:
        out["verdict"] = f"CORRECT: reconstructed turns follow reality ({ev})"
    elif not strong:
        out["verdict"] = (f"WEAK EVIDENCE ({ev}): the correlation is too low to call the "
                          f"turn direction. Sign is {'negative' if scale < 0 else 'positive'}, "
                          f"which HINTS at "
                          f"{'inverted' if scale < 0 else 'correct'} handedness, but a drive "
                          f"with more/cleaner turns is needed to decide.")
    else:
        out["verdict"] = (f"MAGNITUDE MISMATCH ({ev}): the sign says "
                          f"{'inverted' if scale < 0 else 'correct'}, but the gyro's turn "
                          f"size disagrees with GPS by {abs(scale):.1f}x -- that is a "
                          f"separate fault from a flipped sign (mount movement, a bad "
                          f"gravity axis, or too few turns to fit).")
    out["confident"] = bool(strong and unit_ish)
    return out


def summarize(ev: dict, track: dict, fit: dict, args) -> dict:
    """Headline numbers + the growth structure. Deliberately NOT a single average.

    A mean error over a whole drive hides exactly the thing we are looking for: an
    open-loop dead-reckoning track is nearly perfect at the start and arbitrarily
    bad at the end, and the average of those two says nothing useful. So the
    summary reports the terminal and peak error, growth-rate fits, and a per-decile
    table showing the error climbing bucket by bucket.
    """
    err, t_s, dist = ev["error_m"], ev["t_s"], ev["dist_m"]
    duration = float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0
    total_dist = float(dist[-1])

    # Long-horizon projections are only meaningful once the handedness is CONFIRMED
    # CORRECT. If it is inverted, the dominant error is a mirror -- a fixed function
    # of the route's shape, not something that accumulates. If it merely could not
    # be established, the error cannot be attributed to drift at all yet. Either way
    # the extrapolations below are withheld rather than guessed.
    probe = ev["handedness_probe"]
    inverted = str(probe["verdict"]).startswith("INVERTED")
    hand_confirmed = bool(probe.get("confident")) and str(probe["verdict"]).startswith("CORRECT")
    if inverted:
        no_projection = ("handedness is inverted -- the dominant error is a mirror, not "
                         "accumulated drift; resolve the sign before extrapolating")
    elif not hand_confirmed:
        no_projection = ("handedness could not be confirmed on this drive, so the error "
                         "cannot be attributed to accumulated drift yet")
    else:
        no_projection = None

    # Growth models are fitted THROUGH THE ORIGIN, which is only meaningful when the
    # error genuinely starts at ~0 -- i.e. under --anchor first-fix. With --anchor
    # kabsch the translation is a best fit, error does not start at zero, and these
    # R^2 values go negative; the extrapolation is suppressed when that happens.
    k_lin, r2_lin = _fit_through_origin(t_s, err, 1)
    k_quad, r2_quad = _fit_through_origin(t_s, err, 2)
    k_dist, r2_dist = _fit_through_origin(dist, err, 1)
    better_quad = (np.isfinite(r2_quad) and np.isfinite(r2_lin) and r2_quad > r2_lin)
    best_r2 = r2_quad if better_quad else r2_lin
    weak_fit = not (np.isfinite(best_r2) and best_r2 >= 0.3)
    if no_projection is None and weak_fit:
        why = ("--anchor kabsch best-fits the translation, so the error does not start "
               "at zero and a through-origin model does not apply"
               if args.anchor == "kabsch" else
               "the error does not grow monotonically over this sample -- it is dominated "
               "by GPS noise and route geometry rather than by accumulated drift, which "
               "a clip this short cannot separate")
        no_projection = (f"the through-origin growth model fits poorly "
                         f"(R2 {best_r2:.2f}): {why}")

    # per-decile accumulation table (time-bucketed)
    deciles = []
    if duration > 0:
        edges = np.linspace(t_s[0], t_s[-1], 11)
        for i in range(10):
            upper = (t_s <= edges[i + 1]) if i == 9 else (t_s < edges[i + 1])
            m = (t_s >= edges[i]) & upper
            if not np.any(m):
                continue
            deciles.append({
                "bucket": i + 1,
                "t_start_s": float(edges[i]), "t_end_s": float(edges[i + 1]),
                "dist_end_m": float(np.max(dist[m])),
                "mean_error_m": float(np.mean(err[m])),
                "max_error_m": float(np.max(err[m])),
                "mean_abs_cross_m": float(np.nanmean(np.abs(ev["cross_m"][m])))
                if np.any(np.isfinite(ev["cross_m"][m])) else None,
                "mean_abs_along_m": float(np.nanmean(np.abs(ev["along_m"][m])))
                if np.any(np.isfinite(ev["along_m"][m])) else None,
                "mean_abs_heading_err_deg": float(np.nanmean(np.abs(ev["heading_err_deg"][m])))
                if np.any(np.isfinite(ev["heading_err_deg"][m])) else None,
            })

    # Heading drift rate. The R^2 is reported with it because a linear ramp is the
    # right model for a gyro BIAS but the wrong model for, say, a one-off step at a
    # turn (which is what an inverted handedness produces) -- a big rate with a poor
    # R^2 means "something jumped", not "it drifts this fast", and the 10-minute
    # projection derived from it should be ignored.
    hd = ev["heading_err_deg"]
    hd_ok = np.isfinite(hd)
    hdg_rate = hdg_rate_r2 = float("nan")
    if hd_ok.sum() >= 3:
        slope, intercept = np.polyfit(t_s[hd_ok], hd[hd_ok], 1)
        hdg_rate = float(slope)
        resid = hd[hd_ok] - (slope * t_s[hd_ok] + intercept)
        ss_tot = float(np.sum((hd[hd_ok] - hd[hd_ok].mean()) ** 2))
        hdg_rate_r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    recon_len = float(np.sum(np.hypot(np.diff(track["x"]), np.diff(track["y"]))))
    len_ratio = recon_len / total_dist if total_dist > 0 else float("nan")

    # ── Handedness: two independent checks, both reported, neither acted on. ──
    # The yaw-increment probe is the reliable one (it only needs a turn SOMEWHERE
    # in the drive); the Kabsch reflection residual is a second opinion that is
    # undecidable whenever the alignment window is straight.
    # The residual ratio decides; a straight window is reported as a caveat rather
    # than used as a veto, because a near-straight window can still separate the two
    # fits decisively when the residuals differ by a wide margin.
    reflect_note = "not evaluated (manual --heading-offset-deg)"
    rp, rr = fit.get("rms_proper_m"), fit.get("rms_reflected_m")
    if rp is not None and np.isfinite(rp) and np.isfinite(rr):
        res_ratio = max(rp, rr) / max(min(rp, rr), 1e-9)
        aniso = fit.get("window_anisotropy", float("nan"))
        caveat = (f" (note: the alignment window is nearly straight, anisotropy "
                  f"{aniso:.3f}, which makes this comparison fragile)"
                  if np.isfinite(aniso) and aniso < 0.05 else "")
        if res_ratio < 1.5:
            reflect_note = (f"ambiguous: proper {rp:.2f} m vs reflected {rr:.2f} m "
                            f"-- too close to call{caveat}")
        elif rr < rp:
            reflect_note = (f"reflected fit is {res_ratio:.1f}x better "
                            f"({rr:.2f} m vs {rp:.2f} m) -> suggests INVERTED "
                            f"handedness{caveat}")
        else:
            reflect_note = (f"proper rotation fits {res_ratio:.1f}x better "
                            f"({rp:.2f} m vs {rr:.2f} m) -> suggests CORRECT "
                            f"handedness{caveat}")

    return {
        "duration_s": duration,
        "gps_distance_travelled_m": total_dist,
        "reconstructed_path_length_m": recon_len,
        "path_length_ratio_recon_over_gps": len_ratio,
        "n_fixes_scored": int(len(err)),
        "n_fixes_outside_video_span": ev["n_fixes_outside_video"],
        "gps_accuracy_m": {
            "median": float(np.nanmedian(ev["gps_accuracy"])),
            "min": float(np.nanmin(ev["gps_accuracy"])),
            "max": float(np.nanmax(ev["gps_accuracy"])),
        },
        "position_error_m": {
            "final": float(err[-1]),
            "max": float(np.max(err)),
            "max_at_t_s": float(t_s[int(np.argmax(err))]),
            "median": float(np.median(err)),
            "p95": float(np.percentile(err, 95)),
            "at_25pct_of_drive": float(err[int(0.25 * (len(err) - 1))]),
            "at_50pct_of_drive": float(err[int(0.50 * (len(err) - 1))]),
            "at_75pct_of_drive": float(err[int(0.75 * (len(err) - 1))]),
        },
        "along_track_m": {
            "final": _f(ev["along_m"][-1]),
            "max_abs": _f(np.nanmax(np.abs(ev["along_m"]))),
        },
        "cross_track_m": {
            "final": _f(ev["cross_m"][-1]),
            "max_abs": _f(np.nanmax(np.abs(ev["cross_m"]))),
        },
        "heading_deg": {
            "bias_removed_over_align_window": ev["heading_bias_deg"],
            "final_drift": _f(hd[hd_ok][-1] if hd_ok.any() else np.nan),
            "max_abs_drift": _f(np.nanmax(np.abs(hd)) if hd_ok.any() else np.nan),
            "drift_rate_deg_per_s": hdg_rate,
            "drift_rate_r2": _f(hdg_rate_r2),
            "projected_drift_deg_at_10min": (
                hdg_rate * 600.0
                if (hand_confirmed and np.isfinite(hdg_rate)
                    and np.isfinite(hdg_rate_r2) and hdg_rate_r2 >= 0.5)
                else None),
            "projection_suppressed_reason": (
                None if (hand_confirmed and np.isfinite(hdg_rate_r2) and hdg_rate_r2 >= 0.5)
                else (no_projection if not hand_confirmed else
                      "linear drift model fits poorly (R2 < 0.5) -- the heading error is "
                      "not a steady ramp, so extrapolating it would be meaningless")),
            "unwrapped": True,
            "max_gap_between_usable_course_fixes_s": _f(ev["heading_unwrap_max_gap_s"]),
        },
        "growth": {
            "linear_in_time_m_per_s": k_lin, "linear_in_time_r2": r2_lin,
            "quadratic_in_time_m_per_s2": k_quad, "quadratic_in_time_r2": r2_quad,
            "better_time_model": "quadratic" if better_quad else "linear",
            "linear_in_distance_m_per_m": k_dist, "linear_in_distance_r2": r2_dist,
            "error_per_100m_travelled": k_dist * 100.0 if np.isfinite(k_dist) else None,
            "fits_through_origin": True,
            "extrapolated_error_at_10min_m": None if no_projection else (
                k_quad * 600.0 ** 2 if better_quad else
                (k_lin * 600.0 if np.isfinite(k_lin) else None)),
            "extrapolation_suppressed_reason": no_projection,
        },
        "handedness": {
            "yaw_increment_probe": ev["handedness_probe"],
            "kabsch_reflection_second_opinion": reflect_note,
        },
        "error_by_decile": deciles,
    }


def _f(v):
    v = float(v)
    return None if not np.isfinite(v) else v


# ============================================================================
# Writers
# ============================================================================

def write_csv(path: str, ev: dict) -> None:
    cols = ["t_s", "gps_lat", "gps_lon", "gps_speed_mps", "gps_accuracy_m",
            "dist_travelled_m", "recon_lat", "recon_lon",
            "gps_e_m", "gps_n_m", "recon_e_m", "recon_n_m",
            "error_m", "along_track_m", "cross_track_m",
            "recon_heading_deg", "gps_heading_deg", "heading_drift_deg"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in range(len(ev["t_s"])):
            w.writerow([
                f"{ev['t_s'][i]:.3f}",
                f"{ev['gps_lat'][i]:.7f}", f"{ev['gps_lon'][i]:.7f}",
                f"{ev['gps_speed'][i]:.3f}", f"{ev['gps_accuracy'][i]:.2f}",
                f"{ev['dist_m'][i]:.2f}",
                f"{ev['recon_lat'][i]:.7f}", f"{ev['recon_lon'][i]:.7f}",
                f"{ev['gps_e'][i]:.2f}", f"{ev['gps_n'][i]:.2f}",
                f"{ev['recon_e'][i]:.2f}", f"{ev['recon_n'][i]:.2f}",
                f"{ev['error_m'][i]:.2f}",
                _fmt(ev["along_m"][i]), _fmt(ev["cross_m"][i]),
                f"{ev['recon_heading_deg'][i]:.2f}",
                _fmt(ev["gps_heading_deg"][i]), _fmt(ev["heading_err_deg"][i]),
            ])


def _fmt(v, nd=2):
    return "" if not np.isfinite(v) else f"{float(v):.{nd}f}"


def write_plot(path: str, ev: dict, summary: dict, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, err, dist = ev["t_s"], ev["error_m"], ev["dist_m"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 12))

    # 1 -- error vs time, split into along/cross, against the GPS accuracy floor
    ax[0].fill_between(t, 0, ev["gps_accuracy"], color="0.85",
                       label="GPS reported accuracy (noise floor)")
    ax[0].plot(t, err, color="#d62728", lw=2, label="|position error|")
    ax[0].plot(t, np.abs(ev["along_m"]), color="#1f77b4", lw=1.2, ls="--",
               label="|along-track|")
    ax[0].plot(t, np.abs(ev["cross_m"]), color="#2ca02c", lw=1.2, ls="--",
               label="|cross-track|")
    ax[0].set_xlabel("time since first frame (s)")
    ax[0].set_ylabel("error (m)")
    ax[0].set_title(f"Ego-pose reconstruction error vs GPS  --  {label}")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # 2 -- error vs distance travelled, with the growth models
    ax[1].plot(dist, err, color="#d62728", lw=2, label="|position error|")
    g = summary["growth"]
    if g["linear_in_distance_m_per_m"] and np.isfinite(g["linear_in_distance_m_per_m"]):
        ax[1].plot(dist, g["linear_in_distance_m_per_m"] * dist, color="k", ls=":",
                   label=f"linear fit: {g['error_per_100m_travelled']:.1f} m per 100 m "
                         f"(R2={g['linear_in_distance_r2']:.2f})")
    ax[1].set_xlabel("distance travelled along the GPS track (m)")
    ax[1].set_ylabel("error (m)")
    ax[1].set_title("Error growth with distance -- an open-loop track has no mechanism to recover")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # 3 -- heading drift: the term that integrates into quadratic position error
    hd = ev["heading_err_deg"]
    ok = np.isfinite(hd)
    ax[2].axhline(0, color="0.6", lw=0.8)
    ax[2].plot(t[ok], hd[ok], color="#9467bd", lw=2, label="heading drift (bias removed)")
    r = summary["heading_deg"]["drift_rate_deg_per_s"]
    r2 = summary["heading_deg"]["drift_rate_r2"]
    if np.isfinite(r):
        lbl = f"linear fit: {r:+.3f} deg/s (R2={r2:.2f})" if r2 is not None else \
              f"linear fit: {r:+.3f} deg/s"
        if summary["heading_deg"]["projected_drift_deg_at_10min"] is not None:
            lbl += f"  -> {r * 600:+.0f} deg at 10 min"
        ax[2].plot(t, r * t, color="k", ls=":", label=lbl)
    ax[2].set_xlabel("time since first frame (s)")
    ax[2].set_ylabel("heading drift (deg)")
    ax[2].set_title("Heading drift -- integrates into cross-track position error as ~0.5*v*b*t^2")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML map -- generated from a template here, so no folium/branca dependency.
# The browser fetches Leaflet + the Esri tiles; nothing else leaves this machine.
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ego track vs GPS -- __TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;font:13px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
  #map{height:100%}
  .panel{position:absolute;top:10px;right:10px;z-index:1000;background:rgba(20,20,24,.88);
         color:#eee;padding:12px 14px;border-radius:8px;max-width:340px;
         box-shadow:0 2px 12px rgba(0,0,0,.5)}
  .panel h3{margin:0 0 6px;font-size:14px}
  .panel table{border-collapse:collapse;width:100%}
  .panel td{padding:1px 0;vertical-align:top}
  .panel td.k{color:#aaa;padding-right:10px;white-space:nowrap}
  .panel td.v{text-align:right;font-variant-numeric:tabular-nums}
  .sw{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:6px}
  .note{margin-top:8px;font-size:11px;color:#bbb;border-top:1px solid #444;padding-top:6px}
  .leaflet-tooltip.tl{background:rgba(0,0,0,.65);border:0;box-shadow:none;color:#fff;
                      font-size:10px;padding:1px 4px;white-space:nowrap}
  .leaflet-tooltip.tl:before{display:none}
</style></head><body>
<div id="map"></div>
<div class="panel">
  <h3>__TITLE__</h3>
  <table>
    <tr><td class="k"><span class="sw" style="background:#ff7043"></span>reconstruction</td>
        <td class="v">__RECON_LEN__ m</td></tr>
    <tr><td class="k"><span class="sw" style="background:#00e5ff"></span>GPS (truth)</td>
        <td class="v">__GPS_LEN__ m</td></tr>
    <tr><td class="k"><span class="sw" style="background:#76ff03"></span>GPS-anchored</td>
        <td class="v">__SEGERR__ m / fix</td></tr>
    <tr><td class="k">final drift</td><td class="v">__FINAL__ m</td></tr>
    <tr><td class="k">max error</td><td class="v">__MAX__ m (t=__MAXT__ s)</td></tr>
    <tr><td class="k">error / 100 m</td><td class="v">__PER100__ m</td></tr>
    <tr><td class="k">heading drift</td><td class="v">__HDG__ deg/s</td></tr>
    <tr><td class="k">GPS accuracy (med)</td><td class="v">__ACC__ m</td></tr>
  </table>
  <div class="note">__NOTE__</div>
</div>
<script>
const D = __DATA__;

// Basemaps. The default is a Google-Maps-style "hybrid": satellite imagery with the
// road network and street / place names drawn over it. Each basemap builds its own
// tile layers -- one Leaflet layer cannot belong to two base groups at once.
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
const ESRI_ATTR = 'Tiles &copy; Esri -- Source: Esri, Maxar, Earthstar Geographics, HERE, Garmin';
const esri = (service, opts) => L.tileLayer(`${ESRI}${service}/MapServer/tile/{z}/{y}/{x}`,
  Object.assign({maxZoom:21, maxNativeZoom:19}, opts));
const hybrid = L.layerGroup([
  esri('World_Imagery', {attribution:ESRI_ATTR}),
  esri('Reference/World_Transportation'),
  esri('Reference/World_Boundaries_and_Places'),
]);
const imagery = esri('World_Imagery', {attribution:ESRI_ATTR});
const osm = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19, attribution:'&copy; OpenStreetMap contributors'});

const map = L.map('map', {layers:[hybrid]});

const gpsLine   = L.polyline(D.gps,   {color:'#00e5ff', weight:4, opacity:0.95});
const reconLine = L.polyline(D.recon, {color:'#ff7043', weight:4, opacity:0.95});

const spokes = L.layerGroup(D.spokes.map(s =>
  L.polyline([s.a, s.b], {color:'#ff1744', weight:1.5, opacity:0.85, dashArray:'4,4'})
   .bindTooltip(`t=${s.t.toFixed(1)}s  frame ~${s.frame}<br>error ${s.err.toFixed(1)} m`)));

// Permanent time labels ride on the TRUTH track only -- doubling them on both
// tracks turns a long drive into unreadable soup. The spokes already pair them.
function markerLayer(pts, color, name, permanent){
  return L.layerGroup(pts.map(p => L.circleMarker([p.lat, p.lon], {
      radius:4, color:'#111', weight:1, fillColor:color, fillOpacity:1})
    .bindPopup(`<b>${name}</b><br>t = ${p.t.toFixed(1)} s<br>frame ~ ${p.frame}` +
               (p.err !== undefined ? `<br>error = ${p.err.toFixed(1)} m` : '') +
               (p.acc !== undefined ? `<br>GPS accuracy = ${p.acc.toFixed(1)} m` : ''))
    .bindTooltip(`${p.t.toFixed(0)}s`, {permanent:!!permanent, direction:'right',
                                        className:'tl', offset:[6,0]})));
}
const gpsMarks   = markerLayer(D.gps_marks,   '#00e5ff', 'GPS', true);
const reconMarks = markerLayer(D.recon_marks, '#ff7043', 'reconstruction', false);

const overlays = {
  '<span style="color:#ff7043">reconstruction</span>': reconLine,
  '<span style="color:#00e5ff">GPS (truth)</span>': gpsLine,
  'error spokes': spokes,
  'time markers (GPS)': gpsMarks,
  'time markers (recon)': reconMarks,
};
[reconLine, gpsLine, spokes, gpsMarks, reconMarks].forEach(l => l.addTo(map));

// Extra trails. `segs` is a LIST of polylines: the GPS-anchored trail is restarted
// at every fix, so drawing it as one continuous line would invent join strokes
// between the end of one segment and the anchor of the next.
(D.extras || []).forEach(x => {
  const opts = {color:x.color, weight:x.weight, opacity:0.95};
  if (x.dash) opts.dashArray = x.dash;
  const line = L.polyline(x.segs, opts);
  overlays[`<span style="color:${x.color}">${x.label}</span>`] = line;
  if (x.on) line.addTo(map);
});

// top-left, under the zoom buttons: top-right is taken by the stats panel, which
// would otherwise sit on top of the layer switcher and hide it
L.control.layers({'Satellite + roads (hybrid)':hybrid, 'Satellite only':imagery,
                  'Streets (OpenStreetMap)':osm},
                 overlays, {collapsed:false, position:'topleft'}).addTo(map);
L.control.scale({imperial:false}).addTo(map);

L.marker(D.gps[0]).addTo(map).bindPopup('start');
L.marker(D.gps[D.gps.length-1]).addTo(map).bindPopup('end (GPS)');

map.fitBounds(L.latLngBounds(D.gps.concat(D.recon)).pad(0.12));
</script></body></html>
"""


def _latlon_pairs(lat, lon):
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]


def _marker_points(t_s, lat, lon, frames, interval, err=None, acc=None):
    """One marker per --marker-interval-s, taken at the nearest available sample."""
    if len(t_s) == 0:
        return []
    out, next_t = [], float(t_s[0])
    for i in range(len(t_s)):
        if float(t_s[i]) + 1e-9 < next_t:
            continue
        p = {"t": float(t_s[i]), "lat": round(float(lat[i]), 7),
             "lon": round(float(lon[i]), 7), "frame": int(frames[i])}
        if err is not None:
            p["err"] = float(err[i])
        if acc is not None and np.isfinite(acc[i]):
            p["acc"] = float(acc[i])
        out.append(p)
        next_t = float(t_s[i]) + interval
    return out


def write_html(path: str, track: dict, ev: dict, summary: dict, args,
               title: str, note: str, extras: list | None = None,
               anchored: dict | None = None) -> None:
    enu = ev["_enu"]
    fit = ev["_fit"]

    # Full-resolution reconstructed polyline (every frame), in lat/lon.
    fe, fn = apply_alignment(fit, track["x"], track["y"])
    flat, flon = enu.to_latlon(fe, fn)
    t_frames = (track["t_ns"] - track["t_ns"][0]) / 1e9

    # Nearest frame index for each scored GPS fix (for the marker popups).
    fix_frames = np.interp(ev["t_s"], t_frames, track["frames"]).round().astype(int)

    data = {
        "gps": _latlon_pairs(ev["gps_lat"], ev["gps_lon"]),
        "recon": _latlon_pairs(flat, flon),
        "gps_marks": _marker_points(ev["t_s"], ev["gps_lat"], ev["gps_lon"], fix_frames,
                                    args.marker_interval_s, acc=ev["gps_accuracy"]),
        "recon_marks": _marker_points(ev["t_s"], ev["recon_lat"], ev["recon_lon"], fix_frames,
                                      args.marker_interval_s, err=ev["error_m"]),
        "spokes": [],
    }
    for p in _marker_points(ev["t_s"], ev["gps_lat"], ev["gps_lon"], fix_frames,
                            args.marker_interval_s, err=ev["error_m"]):
        i = int(np.argmin(np.abs(ev["t_s"] - p["t"])))
        data["spokes"].append({
            "a": [round(float(ev["gps_lat"][i]), 7), round(float(ev["gps_lon"][i]), 7)],
            "b": [round(float(ev["recon_lat"][i]), 7), round(float(ev["recon_lon"][i]), 7)],
            "t": float(ev["t_s"][i]), "err": float(ev["error_m"][i]),
            "frame": int(fix_frames[i]),
        })

    data["extras"] = extras or []

    seg_err = "n/a"
    if anchored is not None:
        seg_err = f"{anchored['stats']['endpoint_error_m']['median']:.1f}"

    g = summary["growth"]
    html = (_HTML
            .replace("__DATA__", json.dumps(data))
            .replace("__SEGERR__", seg_err)
            .replace("__TITLE__", title)
            .replace("__RECON_LEN__", f"{summary['reconstructed_path_length_m']:.0f}")
            .replace("__GPS_LEN__", f"{summary['gps_distance_travelled_m']:.0f}")
            .replace("__FINAL__", f"{summary['position_error_m']['final']:.1f}")
            .replace("__MAX__", f"{summary['position_error_m']['max']:.1f}")
            .replace("__MAXT__", f"{summary['position_error_m']['max_at_t_s']:.0f}")
            .replace("__PER100__", f"{g['error_per_100m_travelled']:.1f}"
                     if g["error_per_100m_travelled"] is not None else "n/a")
            .replace("__HDG__", f"{summary['heading_deg']['drift_rate_deg_per_s']:+.3f}"
                     if np.isfinite(summary["heading_deg"]["drift_rate_deg_per_s"]) else "n/a")
            .replace("__ACC__", f"{summary['gps_accuracy_m']['median']:.1f}")
            .replace("__NOTE__", note))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overlay the reconstructed world-frame ego track on satellite "
                    "imagery and score it against GPS.")
    ap.add_argument("folder", help="session folder (frames.csv, gps.csv, gyro.csv, gravity.csv)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: the session folder)")
    ap.add_argument("--heading-sign", type=int, default=Constants.HEADING_SIGN,
                    help=f"yaw-rate integration sign (default: Constants.HEADING_SIGN "
                         f"= {Constants.HEADING_SIGN})")
    ap.add_argument("--speed-source", choices=("auto", "fused", "gps"), default="auto",
                    help="auto reproduces main.py (fused accel+GPS when linacc.csv exists)")
    ap.add_argument("--origin-lat", type=float, default=None, help="override anchor latitude")
    ap.add_argument("--origin-lon", type=float, default=None, help="override anchor longitude")
    ap.add_argument("--heading-offset-deg", type=float, default=None,
                    help="override the solved rotation; CCW degrees added to the "
                         "reconstruction's heading to make it a true map heading")
    ap.add_argument("--anchor", choices=("first-fix", "kabsch"), default="first-fix",
                    help="translation: pin to the first usable GPS fix (default, shows "
                         "accumulated drift), or best-fit over the alignment window")
    ap.add_argument("--align-seconds", type=float, default=20.0,
                    help="length of the early motion window used to solve the rotation")
    ap.add_argument("--min-speed", type=float, default=2.0,
                    help="GPS speed (m/s) below which a fix is too slow for a heading")
    ap.add_argument("--min-turn-deg", type=float, default=3.0,
                    help="how much the GPS course must change between two fixes for that "
                         "interval to count toward the handedness probe")
    ap.add_argument("--marker-interval-s", type=float, default=10.0,
                    help="place a time/frame marker every N seconds")
    ap.add_argument("--gps-anchor-mode", choices=("position", "position+course"),
                    default="position",
                    help="GPS-anchored trail: restart each segment at the GPS fix using "
                         "the global rotation (position, default -- adds no GPS truth "
                         "beyond the anchors, but still carries accumulated heading "
                         "drift), or also spin it to depart along the GPS course at that "
                         "fix (position+course -- isolates the purely local error). Both "
                         "are always reported; this picks the one drawn on the map.")
    ap.add_argument("--compare-heading-sign", action="store_true",
                    help="also reconstruct with the OPPOSITE heading sign (a real "
                         "pipeline run, not a mirrored copy) and add it as a "
                         "diagnostic layer + a second metrics block")
    args = ap.parse_args()

    session = os.path.abspath(args.folder)
    out_dir = os.path.abspath(args.out) if args.out else session
    os.makedirs(out_dir, exist_ok=True)

    gps = load_gps(os.path.join(session, "gps.csv"))
    track = reconstruct(session, args.heading_sign, args.speed_source)

    # ── anchor ───────────────────────────────────────────────────────────────
    if (args.origin_lat is None) != (args.origin_lon is None):
        raise SystemExit("[ego-map] --origin-lat and --origin-lon must be given together")
    manual_origin = args.origin_lat is not None
    if manual_origin:
        enu = LocalENU(args.origin_lat, args.origin_lon)
        anchor_src = "manual (--origin-lat/--origin-lon)"
    else:
        enu = LocalENU(float(gps["lat"][0]), float(gps["lon"][0]))
        anchor_src = "first GPS fix"

    fit = solve_alignment(track, gps, enu, args, manual_origin=manual_origin)
    ev = evaluate(track, gps, enu, fit, args)
    ev["_enu"], ev["_fit"] = enu, fit
    summary = summarize(ev, track, fit, args)

    # ── console report ───────────────────────────────────────────────────────
    dur = summary["duration_s"]
    print(f"\n=== session: {session}")
    print(f"  frames {len(track['frames'])} over {(track['t_ns'][-1]-track['t_ns'][0])/1e9:.1f}s   "
          f"ego speed source: {track['speed_source']}   heading_sign: {track['heading_sign']}")
    print(f"  GPS: {summary['n_fixes_scored']} fixes scored "
          f"({summary['n_fixes_outside_video_span']} outside the video span), "
          f"accuracy median {summary['gps_accuracy_m']['median']:.1f} m "
          f"(max {summary['gps_accuracy_m']['max']:.1f} m)")

    print(f"\n=== alignment ({fit['source']}) ===")
    print(f"  tangent-plane origin : {anchor_src} "
          f"({enu.lat0:.7f}, {enu.lon0:.7f})")
    print(f"  translation          : {fit['anchor_note']}")
    print(f"  rotation solved      : {math.degrees(fit['theta_rad']):+.2f} deg "
          f"(CCW, applied to the reconstruction)")
    if np.isfinite(fit["rms_proper_m"]):
        print(f"  fit residual (RMS)   : proper {fit['rms_proper_m']:.2f} m   "
              f"reflected {fit['rms_reflected_m']:.2f} m   "
              f"over {fit['window_fixes']} fixes"
              + ("  [window widened: too few fixes in --align-seconds]"
                 if fit["window_widened"] else ""))
        print(f"  window anisotropy    : {fit['window_anisotropy']:.3f} "
              f"(near 0 = straight line, reflection undecidable)")

    hand = summary["handedness"]
    print(f"\n=== handedness ===")
    print(f"  yaw-increment probe  : {hand['yaw_increment_probe']['verdict']}")
    print(f"  kabsch reflection    : {hand['kabsch_reflection_second_opinion']}")

    pe = summary["position_error_m"]
    print(f"\n=== position error vs GPS ===")
    print(f"  final drift {pe['final']:.1f} m   max {pe['max']:.1f} m at t={pe['max_at_t_s']:.0f}s   "
          f"median {pe['median']:.1f} m   p95 {pe['p95']:.1f} m")
    print(f"  quartile points: 25% {pe['at_25pct_of_drive']:.1f} m   "
          f"50% {pe['at_50pct_of_drive']:.1f} m   75% {pe['at_75pct_of_drive']:.1f} m")
    at, ct = summary["along_track_m"], summary["cross_track_m"]
    print(f"  along-track final {_p(at['final'])} m (max |{_p(at['max_abs'])}|)   "
          f"cross-track final {_p(ct['final'])} m (max |{_p(ct['max_abs'])}|)")
    print(f"  path length: reconstruction {summary['reconstructed_path_length_m']:.0f} m vs "
          f"GPS {summary['gps_distance_travelled_m']:.0f} m "
          f"(ratio {summary['path_length_ratio_recon_over_gps']:.3f})")

    hd, g = summary["heading_deg"], summary["growth"]
    print(f"\n=== drift accumulation ===")
    print(f"  heading drift  : final {_p(hd['final_drift'])} deg (unwrapped), "
          f"max |{_p(hd['max_abs_drift'])}| deg, "
          f"rate {hd['drift_rate_deg_per_s']:+.4f} deg/s (R2 {_p(hd['drift_rate_r2'], 2)})")
    if hd["projected_drift_deg_at_10min"] is not None:
        print(f"                   -> {hd['projected_drift_deg_at_10min']:+.0f} deg at 10 min "
              f"if it keeps ramping at that rate")
    else:
        print(f"                   -> no projection: {hd['projection_suppressed_reason']}")
    print(f"  vs TIME        : linear {g['linear_in_time_m_per_s']:.3f} m/s "
          f"(R2 {g['linear_in_time_r2']:.2f})   "
          f"quadratic {g['quadratic_in_time_m_per_s2']:.5f} m/s^2 "
          f"(R2 {g['quadratic_in_time_r2']:.2f})   -> {g['better_time_model']} fits better")
    print(f"  vs DISTANCE    : {g['error_per_100m_travelled']:.2f} m per 100 m travelled "
          f"(R2 {g['linear_in_distance_r2']:.2f})")
    if g["extrapolated_error_at_10min_m"]:
        print(f"  EXTRAPOLATED   : ~{g['extrapolated_error_at_10min_m']:.0f} m error after "
              f"10 minutes at this rate (from a {dur:.0f}s sample -- treat as an order "
              f"of magnitude, not a promise)")
    elif g["extrapolation_suppressed_reason"]:
        print(f"  EXTRAPOLATED   : suppressed -- {g['extrapolation_suppressed_reason']}")

    print(f"\n=== error by decile of the drive (means hide accumulation; this does not) ===")
    print(f"  {'#':>2} {'t (s)':>14} {'dist (m)':>9} {'mean err':>9} {'max err':>8} "
          f"{'|cross|':>8} {'|along|':>8} {'|hdg| deg':>10}")
    for d in summary["error_by_decile"]:
        print(f"  {d['bucket']:>2} {d['t_start_s']:6.1f}-{d['t_end_s']:6.1f} "
              f"{d['dist_end_m']:9.0f} {d['mean_error_m']:9.1f} {d['max_error_m']:8.1f} "
              f"{_p(d['mean_abs_cross_m']):>8} {_p(d['mean_abs_along_m']):>8} "
              f"{_p(d['mean_abs_heading_err_deg']):>10}")

    # ── GPS-anchored trail: reconstruction restarted at every fix ────────────
    # Both modes are always computed: the gap between them separates ACCUMULATED
    # HEADING drift (which position-only anchoring still carries) from the purely
    # local per-interval shape error. Only the requested one is drawn.
    extras = []
    anchored_modes = {m: gps_anchored_track(track, gps, enu, fit, m)
                      for m in ("position", "position+course")}
    anchored = anchored_modes[args.gps_anchor_mode]
    if anchored is not None:
        print(f"\n=== GPS-anchored trail (dead reckoning restarted at every fix) ===")
        st = anchored["stats"]
        print(f"  {st['n_segments']} segments, mean interval {st['mean_interval_s']:.2f} s, "
              f"mean GPS chord {st['mean_gps_chord_m']:.1f} m   "
              f"(GPS's own median accuracy is {summary['gps_accuracy_m']['median']:.1f} m)")
        for m, a in anchored_modes.items():
            if a is None:
                continue
            s, ee = a["stats"], a["stats"]["endpoint_error_m"]
            drawn = " <- drawn" if m == args.gps_anchor_mode else ""
            print(f"  [{m}]{drawn}")
            print(f"    endpoint error: median {ee['median']:6.2f} m  mean {ee['mean']:6.2f} m  "
                  f"p95 {ee['p95']:6.2f} m  max {ee['max']:6.2f} m  "
                  f"({s['endpoint_error_pct_of_chord_median']:.1f}% of one interval's travel)")
            print(f"    split         : mean |along| {s['mean_abs_along_m']:.2f} m   "
                  f"mean |cross| {s['mean_abs_cross_m']:.2f} m   "
                  f"signed cross {s['mean_signed_cross_m']:+.2f} m")
        pos, crs = anchored_modes["position"], anchored_modes["position+course"]
        if pos and crs:
            pm = pos["stats"]["endpoint_error_m"]["median"]
            cm = crs["stats"]["endpoint_error_m"]["median"]
            print(f"  read it as   : {cm:.2f} m of that is LOCAL (one interval of dead "
                  f"reckoning); the remaining {max(pm - cm, 0.0):.2f} m is accumulated "
                  f"HEADING drift, which position-only anchoring cannot remove.")
        extras.append({
            "segs": anchored["segments"], "color": "#76ff03", "weight": 3,
            "dash": None, "on": True,
            "label": f"GPS-anchored ({args.gps_anchor_mode})",
        })

    # ── optional: what the OTHER heading sign actually produces ──────────────
    comparison = None
    if args.compare_heading_sign:
        alt_sign = -args.heading_sign
        print(f"\n=== comparison: real reconstruction at heading_sign={alt_sign} ===")
        alt = reconstruct(session, alt_sign, args.speed_source)
        alt_fit = solve_alignment(alt, gps, enu, args, manual_origin=manual_origin)
        alt_ev = evaluate(alt, gps, enu, alt_fit, args)
        alt_ev["_enu"], alt_ev["_fit"] = enu, alt_fit
        alt_sum = summarize(alt_ev, alt, alt_fit, args)
        ape = alt_sum["position_error_m"]
        print(f"  rotation {math.degrees(alt_fit['theta_rad']):+.2f} deg   "
              f"fit residual proper {alt_fit['rms_proper_m']:.2f} m / "
              f"reflected {alt_fit['rms_reflected_m']:.2f} m")
        print(f"  final drift {ape['final']:.1f} m   max {ape['max']:.1f} m   "
              f"median {ape['median']:.1f} m   "
              f"(primary run: final {pe['final']:.1f} m, max {pe['max']:.1f} m)")
        print(f"  heading drift rate {alt_sum['heading_deg']['drift_rate_deg_per_s']:+.4f} deg/s")
        alt_anchored = {m: gps_anchored_track(alt, gps, enu, alt_fit, m)
                        for m in ("position", "position+course")}
        for m, a in alt_anchored.items():
            if a is None or anchored_modes.get(m) is None:
                continue
            print(f"  GPS-anchored [{m}] endpoint error: median "
                  f"{a['stats']['endpoint_error_m']['median']:.2f} m (primary run: "
                  f"{anchored_modes[m]['stats']['endpoint_error_m']['median']:.2f} m)")

        ae, an = apply_alignment(alt_fit, alt["x"], alt["y"])
        alat, alon = enu.to_latlon(ae, an)
        extras.append({
            "segs": [_latlon_pairs(alat, alon)], "color": "#ffd54f", "weight": 3,
            "dash": "8,6", "on": False,
            "label": f"diagnostic: heading_sign={alt_sign} (NOT the pipeline default)",
        })
        comparison = {"heading_sign": alt_sign, "summary": alt_sum,
                      "alignment": _fit_json(alt_fit),
                      "gps_anchored": {m: (a["stats"] if a else None)
                                       for m, a in alt_anchored.items()}}

    # ── files ────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "ego_track_metrics.csv")
    json_path = os.path.join(out_dir, "ego_track_report.json")
    png_path = os.path.join(out_dir, "ego_track_error.png")
    html_path = os.path.join(out_dir, "ego_track_map.html")

    write_csv(csv_path, ev)
    label = (f"heading_sign={track['heading_sign']}, speed={track['speed_source']}, "
             f"anchor={args.anchor}")
    write_plot(png_path, ev, summary, label)

    note = ("Drawn track is the pipeline's own output "
            f"(Constants.HEADING_SIGN={Constants.HEADING_SIGN}"
            f"{', overridden to %d' % args.heading_sign if args.heading_sign != Constants.HEADING_SIGN else ''}"
            "). Alignment is a proper rotation + translation only -- no reflection, "
            "no scaling. GPS is truth but carries its own error; see the accuracy row.")
    write_html(html_path, track, ev, summary, args,
               title=os.path.basename(session.rstrip(os.sep)), note=note,
               extras=extras, anchored=anchored)

    report = {
        "session_dir": session,
        "config": {
            "heading_sign": args.heading_sign,
            "constants_heading_sign": Constants.HEADING_SIGN,
            "speed_source_requested": args.speed_source,
            "speed_source_used": track["speed_source"],
            "anchor_mode": args.anchor,
            "tangent_plane_origin": {"source": anchor_src, "lat": enu.lat0, "lon": enu.lon0},
            "align_seconds": args.align_seconds,
            "min_speed_mps": args.min_speed,
            "min_turn_deg": args.min_turn_deg,
            "marker_interval_s": args.marker_interval_s,
            "heading_offset_deg_override": args.heading_offset_deg,
            "compare_heading_sign": bool(args.compare_heading_sign),
        },
        "alignment": _fit_json(fit),
        "summary": summary,
        "gps_anchored": {
            "drawn_mode": args.gps_anchor_mode,
            "stats": {m: (a["stats"] if a else None) for m, a in anchored_modes.items()},
            "per_segment": None if anchored is None else anchored["per_segment"],
        },
        "comparison_opposite_heading_sign": comparison,
        "method_notes": {
            "gps_anchored": "dead reckoning restarted at every GPS fix: GPS gives the "
                            "anchor points, the reconstruction draws the shape between "
                            "them. POSITION is re-anchored, rotation is not re-fitted "
                            "per segment -- re-fitting would hide an inverted handedness. "
                            "This is the horizon speed_estimator actually integrates over.",
            "scale": "locked to 1 -- the track is already metric from GPS speed; a free "
                     "scale would absorb speed-bias drift and hide it",
            "reflection": "never applied to the drawn track; the reflected fit's residual "
                          "is reported as a handedness diagnostic only",
            "truth": "raw GPS fixes, never interpolated (map markers are interpolated for "
                     "display only)",
            "along_cross": "error decomposed in the GPS direction of travel; + along = "
                           "reconstruction ahead of truth, + cross = to its left",
        },
        "outputs": {"html": html_path, "csv": csv_path, "png": png_path},
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)

    print(f"\n[ego-map] map     -> {html_path}")
    print(f"[ego-map] metrics -> {csv_path}")
    print(f"[ego-map] report  -> {json_path}")
    print(f"[ego-map] plot    -> {png_path}")


def _fit_json(fit: dict) -> dict:
    return {
        "source": fit.get("source"),
        "rotation_deg_ccw": math.degrees(fit["theta_rad"]),
        "translation_enu_m": [float(fit["t"][0]), float(fit["t"][1])],
        "rms_proper_m": _f(fit["rms_proper_m"]),
        "rms_reflected_m": _f(fit["rms_reflected_m"]),
        "window_fixes": fit.get("window_fixes"),
        "window_widened": fit.get("window_widened"),
        "window_anisotropy": _f(fit.get("window_anisotropy", float("nan"))),
        "anchor_note": fit.get("anchor_note"),
        "anchor_fix_index": fit.get("anchor_fix_index"),
        "anchor_fix_accuracy_m": fit.get("anchor_fix_accuracy_m"),
    }


def _p(v, nd=1):
    if v is None:
        return "n/a"
    v = float(v)
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    main()
