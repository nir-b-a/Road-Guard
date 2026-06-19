"""
Ego-motion inputs for the world-frame speed estimator.

The world-frame core (speed_estimator.estimate_world_speeds / build_world_runs)
consumes TWO per-frame dicts, regardless of source:
    ego_pos      {frame -> (x, y)}   world metres
    ego_heading  {frame -> rad}      world heading

Two sources produce those dicts:
    1. CARLA telemetry.csv   (simulation) -- positions are exact, so heading and
       position come straight from the logged ego_x / ego_y.
    2. Android sensor logs   (real dashcam, see ANDROID_DATA_SPEC.md) -- there is
       no logged absolute pose, so we RECONSTRUCT it: integrate the vertical-axis
       gyro yaw RATE into a heading, then dead-reckon position from GPS speed along
       that heading. Both integrations use the REAL per-frame dt from frames.csv.

This module also resolves the per-clip pieces the rest of the pipeline needs from
the raw streams: the effective fps (fps_from_frame_timestamps) and the camera
intrinsics (load_intrinsics).

----------------------------------------------------------------------------
SIGNS (real footage only)
    Two handedness/mount-dependent signs must be re-validated on the first real
    clips (CARLA's values do NOT necessarily transfer):
      * Constants.HEADING_SIGN -- direction the vertical-axis yaw rate integrates
        into world heading (applied here, in ego_heading_from_android).
      * Constants.LAT_SIGN     -- handedness of the camera->world de-rotation
        (applied in speed_estimator.build_world_runs).
    The absolute heading offset and absolute position origin are arbitrary: speed
    magnitude is invariant to a global rotation, and a constant position offset
    cancels under differentiation -- so only heading CHANGES and speed need to be
    accurate, which is exactly what gyro + GPS give.
----------------------------------------------------------------------------
"""

import csv
import json
import math
import os
import numpy as np

import Constants


# ============================================================================
# Small numeric helpers
# ============================================================================

def _fill_nan(a: np.ndarray) -> np.ndarray:
    """Forward-fill then back-fill NaNs; all-NaN -> zeros."""
    a = a.astype(float).copy()
    n = len(a)
    last = np.nan
    for i in range(n):
        if np.isnan(a[i]):
            a[i] = last
        else:
            last = a[i]
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if np.isnan(a[i]):
            a[i] = nxt
        else:
            nxt = a[i]
    a[np.isnan(a)] = 0.0
    return a


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _maybe_savgol(values: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smooth if the series is long enough, else return as-is."""
    if len(values) < polyorder + 2:
        return values
    w = min(_odd(window), _odd(len(values) - 1))
    if w < polyorder + 2:
        return values
    from scipy.signal import savgol_filter
    return savgol_filter(values, w, polyorder)


# ============================================================================
# Source 1 -- CARLA telemetry
# ============================================================================

def ego_speed_from_telemetry(path: str) -> dict[int, float]:
    """Per-frame ego speed (m/s).  Prefers ego_speed_kmh, falls back to mps."""
    with open(path, "r") as f:
        header = (csv.DictReader(f).fieldnames or [])

    col, scale = None, 1.0
    if "ego_speed_kmh" in header:
        col, scale = "ego_speed_kmh", 1.0 / 3.6
    elif "ego_speed_mps" in header:
        col, scale = "ego_speed_mps", 1.0

    speeds: dict[int, float] = {}
    if col is not None:
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                try:
                    speeds[int(row["frame"])] = float(row[col]) * scale
                except (KeyError, ValueError):
                    continue
        return speeds

    # Fallback: derive from positions (central difference handled elsewhere is
    # overkill here; this branch is rare since sims log ego_speed_kmh).
    return speeds


def ego_yaw_rate_from_telemetry(path: str,
                                fps: float,
                                min_move_m: float = 0.05,
                                smooth_window: int = 15) -> dict[int, float]:
    """
    Per-frame ego yaw rate (rad/s), computed from the heading of the ego's
    motion between consecutive samples.

    At very low speed the heading from position deltas is meaningless, so steps
    shorter than `min_move_m` are marked NaN and filled from neighbours.
    """
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((int(row["frame"]),
                             float(row["ego_x"]),
                             float(row["ego_y"])))
            except (KeyError, ValueError):
                continue
    if len(rows) < 3:
        return {fr: 0.0 for fr, _, _ in rows}

    rows.sort(key=lambda r: r[0])
    frames = np.array([r[0] for r in rows])
    xs     = np.array([r[1] for r in rows])
    ys     = np.array([r[2] for r in rows])
    n = len(frames)

    headings = np.full(n, np.nan)
    for i in range(n - 1):
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        if math.hypot(dx, dy) >= min_move_m:
            headings[i] = math.atan2(dy, dx)
    headings[n - 1] = headings[n - 2]
    headings = _fill_nan(headings)

    unwrapped = np.unwrap(headings)
    dt = 1.0 / fps
    omega = np.gradient(unwrapped, dt)
    omega = _maybe_savgol(omega, smooth_window)

    return {int(frames[i]): float(omega[i]) for i in range(n)}


def ego_position_from_telemetry(path: str) -> dict[int, tuple[float, float]]:
    """frame -> (ego_x, ego_y) in world metres."""
    out: dict[int, tuple[float, float]] = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["frame"])] = (float(row["ego_x"]), float(row["ego_y"]))
            except (KeyError, ValueError):
                continue
    return out


def ego_heading_from_telemetry(path: str,
                               min_move_m: float = 0.05,
                               smooth_window: int = 0) -> dict[int, float]:
    """
    Per-frame ego heading (rad, world frame), from the direction of motion
    between consecutive samples.  Unlike the yaw RATE, heading is an integral
    quantity -- smooth and well-defined even through a sharp lane change, which
    is why the world-frame estimator uses it instead of omega.

    smooth_window=0 -> no smoothing (CARLA positions are exact).
    """
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((int(row["frame"]), float(row["ego_x"]), float(row["ego_y"])))
            except (KeyError, ValueError):
                continue
    if len(rows) < 2:
        return {fr: 0.0 for fr, _, _ in rows}

    rows.sort(key=lambda r: r[0])
    frames = np.array([r[0] for r in rows])
    xs     = np.array([r[1] for r in rows])
    ys     = np.array([r[2] for r in rows])
    n = len(frames)

    headings = np.full(n, np.nan)
    for i in range(n - 1):
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        if math.hypot(dx, dy) >= min_move_m:
            headings[i] = math.atan2(dy, dx)
    headings[n - 1] = headings[n - 2]
    headings = np.unwrap(_fill_nan(headings))
    if smooth_window and smooth_window > 2:
        headings = _maybe_savgol(headings, smooth_window)

    return {int(frames[i]): float(headings[i]) for i in range(n)}


# ============================================================================
# Source 2 -- Android sensor logs   (see ANDROID_DATA_SPEC.md)
#
# Streams (separate CSVs, all timestamps in nanoseconds on the SAME clock,
# i.e. SystemClock.elapsedRealtimeNanos):
#   frames.csv   : frame, timestamp_ns
#   gyro.csv     : timestamp_ns, gx, gy, gz            (rad/s, device frame)
#   gravity.csv  : timestamp_ns, grx, gry, grz         (m/s^2, device frame)
#   gps.csv      : timestamp_ns, lat, lon, speed_mps, bearing_deg, accuracy_m
# ============================================================================

def load_frame_timestamps(frames_csv: str) -> dict[int, int]:
    """frame -> timestamp_ns."""
    out: dict[int, int] = {}
    with open(frames_csv, "r") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["frame"])] = int(row["timestamp_ns"])
            except (KeyError, ValueError):
                continue
    return out


def _read_xyz(path: str, t_col: str, cols: tuple[str, str, str]):
    ts, vec = [], []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            try:
                ts.append(int(row[t_col]))
                vec.append([float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])])
            except (KeyError, ValueError):
                continue
    order = np.argsort(ts)
    return np.array(ts)[order].astype(np.int64), np.array(vec)[order]


def ego_yaw_rate_from_android(frames_csv: str,
                              gyro_csv: str,
                              gravity_csv: str) -> dict[int, float]:
    """
    Per-frame ego yaw rate (rad/s) about the VERTICAL (gravity) axis.

    Method: a gyroscope returns the angular-velocity vector in the device
    frame.  The component about the vertical axis is the yaw rate, regardless
    of how the phone is mounted.  We get "up" from the (negated) gravity
    vector, project the gyro vector onto it, then average the per-sample yaw
    rates that fall inside each video frame's time window.
    """
    frame_ts = load_frame_timestamps(frames_csv)
    if not frame_ts:
        return {}

    g_ts, g_vec = _read_xyz(gyro_csv, "timestamp_ns", ("gx", "gy", "gz"))
    grav_ts, grav_vec = _read_xyz(gravity_csv, "timestamp_ns", ("grx", "gry", "grz"))
    if len(g_ts) == 0 or len(grav_ts) == 0:
        return {fr: 0.0 for fr in frame_ts}

    # up unit vector per gyro sample (nearest gravity sample, negated, normalised)
    idx = np.clip(np.searchsorted(grav_ts, g_ts), 0, len(grav_ts) - 1)
    grav_at_gyro = grav_vec[idx]
    norms = np.linalg.norm(grav_at_gyro, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    up = -grav_at_gyro / norms
    yaw_rate_per_sample = np.sum(g_vec * up, axis=1)  # rad/s about vertical

    # frame time windows = midpoints between adjacent frame timestamps
    frames_sorted = sorted(frame_ts)
    t_arr = np.array([frame_ts[fr] for fr in frames_sorted], dtype=np.int64)
    mids = (t_arr[:-1] + t_arr[1:]) // 2
    lo = np.empty_like(t_arr); hi = np.empty_like(t_arr)
    lo[0] = t_arr[0] - (mids[0] - t_arr[0]); lo[1:] = mids
    hi[-1] = t_arr[-1] + (t_arr[-1] - mids[-1]); hi[:-1] = mids

    out: dict[int, float] = {}
    for k, fr in enumerate(frames_sorted):
        mask = (g_ts >= lo[k]) & (g_ts < hi[k])
        out[fr] = float(np.mean(yaw_rate_per_sample[mask])) if np.any(mask) else 0.0
    return out


def ego_speed_from_android(frames_csv: str, gps_csv: str) -> dict[int, float]:
    """Per-frame ego speed (m/s) by linear interpolation of GPS speed in time."""
    frame_ts = load_frame_timestamps(frames_csv)
    if not frame_ts:
        return {}

    ts, spd = [], []
    with open(gps_csv, "r") as f:
        for row in csv.DictReader(f):
            try:
                ts.append(int(row["timestamp_ns"]))
                spd.append(float(row["speed_mps"]))
            except (KeyError, ValueError):
                continue
    if not ts:
        return {fr: 0.0 for fr in frame_ts}

    order = np.argsort(ts)
    ts = np.array(ts)[order].astype(float)
    spd = np.array(spd)[order]

    out: dict[int, float] = {}
    for fr, t in frame_ts.items():
        out[fr] = float(np.interp(t, ts, spd))
    return out


# --- ego_speed_fused tuning -------------------------------------------------
# The complementary filter pulls the accel-integrated speed back toward GPS with a
# correction time constant tau. GPS is most trustworthy exactly when slow/stopped
# (Doppler reads ~0 cleanly) and laggiest when fast, so tau is ADAPTIVE: short at
# low speed (snap to GPS) and long at speed (let the accelerometer carry the
# dynamics between sparse fixes). tau ramps linearly over [0, TAU_FULL_SPEED_MPS].
TAU_FAST_S = 1.5            # correction time constant at/above TAU_FULL_SPEED_MPS
TAU_LOW_S = 0.4            # correction time constant near 0 speed (snappy)
TAU_FULL_SPEED_MPS = 8.0   # GPS speed (m/s) at which tau reaches TAU_FAST_S (~29 km/h)
# Zero-velocity update: when GPS says we're below this, the car is stopped, so the
# fused speed is forced to exactly 0 -- killing integrated-accel drift at a
# standstill (the "ends at 5.5 km/h instead of 0" tail).
ZUPT_GPS_MPS = 0.7         # ~2.5 km/h


def ego_speed_fused(frames_csv: str,
                    gps_csv: str,
                    linacc_csv: str,
                    gravity_csv: str,
                    *,
                    tau_s: float = TAU_FAST_S,
                    tau_low_s: float = TAU_LOW_S,
                    zupt_gps_mps: float = ZUPT_GPS_MPS) -> dict[int, float] | None:
    """
    Per-frame ego SPEED (m/s) at FRAME RATE, fusing the high-rate linear
    accelerometer (fast, drifts) with GPS speed (slow, absolute) via a 1st-order
    complementary filter.

    Why this exists: GPS alone is ~1 Hz (often far less), so it cannot match the
    camera's per-frame depth-closing rate. The world-frame reconstruction then fails
    to cancel ego motion and a STATIONARY target reads ~= ego speed. The accelerometer
    supplies the between-fix dynamics (the deceleration into a red light, etc.); GPS
    pins the absolute level and removes accelerometer drift/bias.

    Method (no camera/mount assumptions -- the forward axis is LEARNED from GPS):
      1. down = unit(mean gravity); build a horizontal 2D basis (e1, e2) perp to it.
      2. project linear accel into that plane -> a2d(t)  (vertical component dropped).
      3. learn the device->forward map w (2D) by least-squares fitting the integral of
         a2d over each GPS interval to the GPS speed change:  (integral a2d dt) . w ~= dv_gps.
         This resolves the forward axis, its sign, AND scale, anchored entirely to GPS.
      4. signed longitudinal accel  a_f(t) = a2d(t) . w.
      5. complementary filter over accel sample times, with two refinements that
         matter at a stop (where GPS is best and accel drift hurts most):
            * adaptive tau -- the correction time constant is short at low speed
              (snap to GPS) and long at speed (trust accel between sparse fixes),
              ramping over [0, TAU_FULL_SPEED_MPS]:
                  v += a_f*dt ;  v += min(1, dt/tau_eff) * (v_gps_interp(t) - v)
            * zero-velocity update (ZUPT) -- when v_gps_interp <= zupt_gps_mps the
              car is stopped, so v is forced to exactly 0 (no integrated-accel drift).
      6. sample v at frame timestamps; clamp >= 0.

    Tuning knobs (defaults = module constants):
        tau_s        correction time constant at speed (TAU_FAST_S).
        tau_low_s    correction time constant near 0 speed (TAU_LOW_S, snappier).
        zupt_gps_mps GPS speed below which a ZUPT pins the fused speed to 0.

    Returns None (so the caller falls back to GPS-only ego_speed_from_android) when
    linacc is missing/too short, GPS has < 3 fixes, or the forward-axis fit is degenerate.
    """
    frame_ts = load_frame_timestamps(frames_csv)
    if not frame_ts:
        return {}

    if not (linacc_csv and os.path.exists(linacc_csv)):
        return None
    a_ts, a_vec = _read_xyz(linacc_csv, "timestamp_ns", ("ax", "ay", "az"))
    if len(a_ts) < 10:
        return None

    # GPS speed
    g_ts_list, spd_list = [], []
    with open(gps_csv, "r") as f:
        for row in csv.DictReader(f):
            try:
                g_ts_list.append(int(row["timestamp_ns"]))
                spd_list.append(float(row["speed_mps"]))
            except (KeyError, ValueError):
                continue
    if len(g_ts_list) < 3:
        return None
    g_ts = np.array(g_ts_list, dtype=float)
    spd = np.array(spd_list, dtype=float)
    order = np.argsort(g_ts)
    g_ts, spd = g_ts[order], spd[order]

    # down direction from mean gravity (mount is ~fixed during a drive)
    if gravity_csv and os.path.exists(gravity_csv):
        _, gr_vec = _read_xyz(gravity_csv, "timestamp_ns", ("grx", "gry", "grz"))
        down = np.mean(gr_vec, axis=0)
    else:
        down = np.array([0.0, 0.0, 9.81])
    nd = np.linalg.norm(down)
    if nd == 0:
        return None
    down = down / nd
    seed = np.array([1.0, 0.0, 0.0]) if abs(down[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = seed - (seed @ down) * down
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(down, e1)

    a_t = a_ts.astype(float) / 1e9                      # s
    a2d = np.column_stack([a_vec @ e1, a_vec @ e2])     # N x 2 horizontal accel

    # learn forward map w from GPS speed deltas: (integral a2d) . w ~= dv
    H, dv = [], []
    for i in range(len(g_ts) - 1):
        t0, t1 = g_ts[i] / 1e9, g_ts[i + 1] / 1e9
        m = (a_t >= t0) & (a_t < t1)
        if np.count_nonzero(m) < 2:
            continue
        yy = a2d[m]                                     # (k, 2) horizontal accel
        step = np.diff(a_t[m])                          # (k-1,)
        integ = np.sum(0.5 * (yy[:-1] + yy[1:]) * step[:, None], axis=0)  # trapezoid -> 2-vector
        H.append(integ)
        dv.append(spd[i + 1] - spd[i])
    if len(H) < 2:
        return None
    w, *_ = np.linalg.lstsq(np.array(H), np.array(dv), rcond=None)
    a_f = a2d @ w                                       # signed longitudinal accel

    # complementary filter over accel sample times, with adaptive GPS authority
    # (faster pull when slow) and a zero-velocity update when GPS says stopped.
    v_gps = np.interp(a_t, g_ts / 1e9, spd)
    span = max(TAU_FULL_SPEED_MPS, 1e-6)
    v = np.zeros(len(a_t))
    v[0] = 0.0 if v_gps[0] <= zupt_gps_mps else v_gps[0]
    for k in range(1, len(a_t)):
        if v_gps[k] <= zupt_gps_mps:                    # ZUPT: GPS says stopped -> hard 0
            v[k] = 0.0
            continue
        dt = a_t[k] - a_t[k - 1]
        if dt <= 0 or dt > 1.0:                         # gap guard -> resync to GPS
            v[k] = v_gps[k]
            continue
        # tau short at low speed (trust GPS), long at speed (trust accel)
        tau_eff = tau_low_s + (tau_s - tau_low_s) * min(1.0, v_gps[k] / span)
        vp = v[k - 1] + a_f[k] * dt
        v[k] = vp + min(1.0, dt / tau_eff) * (v_gps[k] - vp)
    v = np.clip(v, 0.0, None)

    # sample at frame timestamps (accel-propagated past the last GPS fix, not flat-held)
    frames_sorted = sorted(frame_ts)
    ft = np.array([frame_ts[fr] for fr in frames_sorted], dtype=float) / 1e9
    v_frame = np.interp(ft, a_t, v)
    return {fr: float(v_frame[k]) for k, fr in enumerate(frames_sorted)}


def ego_heading_from_android(frames_csv: str,
                             gyro_csv: str,
                             gravity_csv: str,
                             *,
                             sign: int = Constants.HEADING_SIGN) -> dict[int, float]:
    """
    Per-frame ego heading (rad, arbitrary world frame) by integrating the
    vertical-axis yaw RATE over the REAL per-frame dt.

    ego_yaw_rate_from_android gives a per-frame yaw rate (rad/s) about the gravity
    axis; we trapezoid-integrate it across frame timestamps to a heading. Heading
    is anchored to 0 at the first frame -- the absolute offset is irrelevant
    (speed magnitude is invariant to a global rotation), so only the CHANGES (the
    gyro's strength) matter. `sign` flips the integration sense (Constants.HEADING_SIGN).
    """
    frame_ts = load_frame_timestamps(frames_csv)
    if len(frame_ts) < 2:
        return {fr: 0.0 for fr in frame_ts}

    yaw_rate = ego_yaw_rate_from_android(frames_csv, gyro_csv, gravity_csv)

    frames_sorted = sorted(frame_ts)
    t = np.array([frame_ts[fr] for fr in frames_sorted], dtype=np.float64) / 1e9  # s
    w = np.array([yaw_rate.get(fr, 0.0) for fr in frames_sorted], dtype=np.float64)

    heading = np.zeros(len(frames_sorted))
    for k in range(1, len(frames_sorted)):
        dt = t[k] - t[k - 1]
        heading[k] = heading[k - 1] + sign * 0.5 * (w[k] + w[k - 1]) * dt  # trapezoid
    return {frames_sorted[k]: float(heading[k]) for k in range(len(frames_sorted))}


def ego_position_from_android(frames_csv: str,
                              gps_csv: str,
                              ego_heading: dict[int, float],
                              *,
                              linacc_csv: str | None = None,
                              gravity_csv: str | None = None) -> dict[int, tuple[float, float]]:
    """
    Per-frame ego position (x, y in metres, arbitrary world frame) by
    dead-reckoning GPS speed along the integrated heading, over the REAL per-frame
    dt.

        x += v * dt * cos(heading);  y += v * dt * sin(heading)   (trapezoid in v)

    Origin anchored to (0, 0) at the first frame -- a constant position offset
    cancels when the speed estimator differentiates, so only relative motion
    matters. `ego_heading` is the output of ego_heading_from_android.
    """
    frame_ts = load_frame_timestamps(frames_csv)
    if not frame_ts:
        return {}

    # Prefer the accelerometer+GPS fused speed (frame-rate, cancels ego motion during
    # dynamic approaches); fall back to GPS-only interpolation when accel is unavailable.
    speed = None
    if linacc_csv:
        speed = ego_speed_fused(frames_csv, gps_csv, linacc_csv, gravity_csv or "")
    if speed is None:
        speed = ego_speed_from_android(frames_csv, gps_csv)
        print("[ego] ego speed: GPS-only interpolation (no linacc.csv or fusion unavailable)")
    else:
        print("[ego] ego speed: accelerometer+GPS fused (frame-rate)")

    frames_sorted = sorted(frame_ts)
    t = np.array([frame_ts[fr] for fr in frames_sorted], dtype=np.float64) / 1e9  # s

    out: dict[int, tuple[float, float]] = {frames_sorted[0]: (0.0, 0.0)}
    x = y = 0.0
    for k in range(1, len(frames_sorted)):
        dt = t[k] - t[k - 1]
        f_prev, f_cur = frames_sorted[k - 1], frames_sorted[k]
        h_prev, h_cur = ego_heading.get(f_prev, 0.0), ego_heading.get(f_cur, 0.0)
        v_prev, v_cur = speed.get(f_prev, 0.0), speed.get(f_cur, 0.0)
        vx = 0.5 * (v_prev * math.cos(h_prev) + v_cur * math.cos(h_cur))
        vy = 0.5 * (v_prev * math.sin(h_prev) + v_cur * math.sin(h_cur))
        x += vx * dt
        y += vy * dt
        out[f_cur] = (x, y)
    return out


# ============================================================================
# Per-clip pieces the rest of the pipeline needs from the raw streams
# ============================================================================

def fps_from_frame_timestamps(frames_csv: str) -> float | None:
    """
    Effective fps for a real clip = 1e9 / median(diff(timestamp_ns)).

    Phone capture is variable-frame-rate; the median frame interval is a robust
    single timebase for the differentiation step (which assumes uniform dt). The
    per-frame integrations (heading/position above) still use the exact dt.
    Returns None if there are fewer than 2 usable frames (caller keeps the video's
    own fps).
    """
    frame_ts = load_frame_timestamps(frames_csv)
    if len(frame_ts) < 2:
        return None
    ts = np.array([frame_ts[fr] for fr in sorted(frame_ts)], dtype=np.float64)
    dt_ns = np.diff(ts)
    dt_ns = dt_ns[dt_ns > 0]
    if len(dt_ns) == 0:
        return None
    return float(1e9 / np.median(dt_ns))


def load_intrinsics(intrinsics_json: str,
                    frame_width: int,
                    frame_height: int) -> tuple[float, float, float, float]:
    """
    Resolve camera intrinsics (fx, fy, cx, cy) in pixels for a real clip.

    Priority:
      1. explicit fx, fy, cx, cy in intrinsics.json  (most robust);
      2. image_width / image_height / fov_horizontal_deg in the json;
      3. fall back to the actual frame size + Constants.FOV_HORIZONTAL_DEG
         (reproduces the simulation's pinhole-from-FOV math).
    A missing or unreadable file simply falls through to (3) with a warning.
    """
    data: dict = {}
    if intrinsics_json and os.path.exists(intrinsics_json):
        try:
            with open(intrinsics_json, "r") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[ego_yaw] could not read intrinsics {intrinsics_json}: {e}")
    else:
        print(f"[ego_yaw] no intrinsics.json -> falling back to frame size + "
              f"FOV={Constants.FOV_HORIZONTAL_DEG} deg")

    if all(k in data for k in ("fx", "fy", "cx", "cy")):
        return float(data["fx"]), float(data["fy"]), float(data["cx"]), float(data["cy"])

    width = int(data.get("image_width", frame_width))
    height = int(data.get("image_height", frame_height))
    fov = float(data.get("fov_horizontal_deg", Constants.FOV_HORIZONTAL_DEG))
    fl = (width / 2) / math.tan(math.radians(fov / 2))
    return fl, fl, width / 2.0, height / 2.0
