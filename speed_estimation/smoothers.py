"""
Smoothers -- turn a series of world POSITIONS into VELOCITY (speed).

One file, four interchangeable tools that all operate on `Run` objects (a
contiguous, gap-free stretch of a single vehicle's world track):

    kalman_speed_runs     constant-acceleration Kalman + RTS smoother. The
                          DEFAULT and only one that also emits a per-frame
                          uncertainty (velocity std) for the alert gate.
    savgol_speed_runs     Savitzky-Golay reference (smooth then differentiate).
    theil_sen_speed_runs  robust median-of-slopes (outlier-tolerant), no std.

plus a measurement-level outlier filter used BEFORE differentiation:

    hampel_filter / hampel_clean_runs   median/MAD spike rejection.

`Run` (the shared position-series type) lives here too, since every smoother
consumes it. This module has no dependency on the estimator/geometry code -- it
only knows positions in, velocity out.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import savgol_filter

import Constants


def _odd(n: int) -> int:
    """Round up to the nearest odd integer (Savitzky-Golay windows must be odd)."""
    return n if n % 2 == 1 else n + 1


@dataclass
class Run:
    """One contiguous (gap-free) stretch of a single vehicle's track."""
    frames: list[int]
    x: np.ndarray              # measurement series (e.g. world Tx, or depth)
    y: np.ndarray | None = None  # optional second axis (e.g. world Ty)
    meta: dict = field(default_factory=dict)


# ============================================================================
# Hampel outlier rejection (measurement level, BEFORE differentiation)
#
# A sliding-window median/MAD spike detector. Median and MAD (not mean/std) are
# used because they are not corrupted by the very outliers we are looking for.
# Run before differentiation: one bad position corrupts TWO velocity samples
# (the finite difference on each side), so we fix it at the source.
# sigma = 1.4826 * MAD makes MAD a consistent std estimator for Gaussian data.
# ============================================================================

HAMPEL_HALF_WINDOW = 9       # frames each side (~0.3s @30fps): local baseline span
HAMPEL_N_SIGMAS = 3.0        # robust-sigma threshold; 3 keeps ~99.7% of clean data
MAD_TO_SIGMA = 1.4826        # scales MAD -> std for Gaussian data


def hampel_filter(values: np.ndarray,
                  half_window: int = HAMPEL_HALF_WINDOW,
                  n_sigmas: float = HAMPEL_N_SIGMAS,
                  replace: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Hampel filter to a 1-D series.

    Args:
        values:      1-D array (may contain NaN for missing samples).
        half_window: window radius in samples.
        n_sigmas:    rejection threshold in robust-sigma units.
        replace:     "median" -> substitute local median; "nan" -> mark NaN.

    Returns:
        (cleaned, mask) where mask[i] is True for rejected samples.

    Edge cases:
        - empty or all-NaN input -> returned unchanged, empty mask.
        - window with MAD == 0 (flat/stationary) -> cannot judge -> no rejection.
        - boundaries -> window is clipped (asymmetric) rather than padded.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    cleaned = x.copy()
    mask = np.zeros(n, dtype=bool)
    if n == 0 or np.all(np.isnan(x)):
        return cleaned, mask

    for i in range(n):
        if np.isnan(x[i]):
            continue
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        window = x[lo:hi]
        window = window[~np.isnan(window)]
        if len(window) == 0:
            continue
        med = np.median(window)
        sigma = MAD_TO_SIGMA * np.median(np.abs(window - med))
        if sigma == 0:
            continue  # degenerate flat window -- everything looks infinitely far
        if abs(x[i] - med) > n_sigmas * sigma:
            mask[i] = True
            cleaned[i] = med if replace == "median" else np.nan
    return cleaned, mask


def hampel_clean_runs(runs: list[Run],
                      half_window: int = HAMPEL_HALF_WINDOW,
                      n_sigmas: float = HAMPEL_N_SIGMAS,
                      replace: str = "median") -> list[Run]:
    """
    Hampel-clean the x (and y, if present) series of every Run in place,
    printing a single summary line.
    """
    rejected = 0
    total = 0
    for run in runs:
        run.x, mx = hampel_filter(run.x, half_window, n_sigmas, replace)
        rejected += int(mx.sum()); total += len(mx)
        if run.y is not None:
            run.y, my = hampel_filter(run.y, half_window, n_sigmas, replace)
            rejected += int(my.sum()); total += len(my)
    rate = (100.0 * rejected / total) if total else 0.0
    print(f"[hampel] rejected {rejected}/{total} samples ({rate:.2f}%) "
          f"across {len(runs)} runs")
    return runs


# ============================================================================
# Kalman smoother (constant-acceleration, forward filter + RTS backward smoother)
#
# State x = [position, velocity, acceleration]^T per 1-D axis.
#   F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]   constant-acceleration transition
#   H = [1, 0, 0]                                   we measure position only
#   Q = white-noise-JERK process noise (jerk PSD = jerk_psd); its dt^5/20..dt
#       structure is the exact integral of a continuous jerk process (Bar-Shalom).
#   R = meas_noise_std^2
# We have the whole track offline, so the symmetric RTS smoother removes the lag
# a causal filter would have. This is the ONLY smoother that emits a velocity std.
# ============================================================================

DEFAULT_MEAS_NOISE_STD_M = 1.0   # R. std of the position measurement (m). trust to about +-1m.
DEFAULT_JERK_PSD = 2.0           # q. grows P, bigger q means P grows more between frames but more noisy, smaller - less reactive
INIT_VAR_POS = 10.0 ** 2         # initial position variance (m^2): weak prior
INIT_VAR_VEL = 10.0 ** 2         # initial velocity variance ((m/s)^2)
INIT_VAR_ACC = 10.0 ** 2         # initial acceleration variance

# uncertanty of next state values
def _Q(dt: float, q: float) -> np.ndarray:
    return q * np.array([
        [dt**5 / 20, dt**4 / 8, dt**3 / 6],
        [dt**4 / 8,  dt**3 / 3, dt**2 / 2],
        [dt**3 / 6,  dt**2 / 2, dt],
    ])

# constant acceleration model
def _F(dt: float) -> np.ndarray:
    return np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])


def kalman_velocity_1d(values: np.ndarray,
                       fps: float,
                       meas_noise_std: float = DEFAULT_MEAS_NOISE_STD_M,
                       jerk_psd: float = DEFAULT_JERK_PSD,
                       dts: np.ndarray | None = None,
                       noise_scale: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth a 1-D position series and return (position, velocity, velocity_std).

    NaN samples are treated as missing (predict-only, no update).
    Series shorter than 2 valid samples are returned with zero velocity.

    `dts`: optional per-step time deltas (s), dts[k] = t[k]-t[k-1]; when given, the
    filter uses the REAL frame intervals (correct across variable fps / gaps) instead
    of a constant 1/fps. dts[0] is the propagation step for the first sample.

    `noise_scale`: optional per-sample multiplier (>= 1) on the measurement-noise
    std. A frame with scale s gets R = (meas_noise_std * s)^2, i.e. it is trusted
    LESS (the filter leans on its model there). Used to down-weight low-confidence
    frames such as wide-angle targets. None -> uniform meas_noise_std (unchanged).
    """
    z = np.asarray(values, dtype=float)                     # positions (m) along ONE axis
    n = len(z)
    if n == 0:
        return z, np.zeros(0), np.zeros(0)                  # nothing to smooth

    if dts is None:
        dt_arr = np.full(n, 1.0 / fps)
    else:
        dt_arr = np.asarray(dts, dtype=float).copy()
        # guard: non-finite / non-positive -> nominal step (avoids singular transitions)
        bad = ~np.isfinite(dt_arr) | (dt_arr <= 0)
        dt_arr[bad] = 1.0 / fps

    # Per-sample measurement-noise std (variance built inside the update step).
    if noise_scale is None:
        r_std = np.full(n, float(meas_noise_std))
    else:
        r_std = float(meas_noise_std) * np.asarray(noise_scale, dtype=float)

    H = np.array([[1.0, 0.0, 0.0]])                                          # "what we measure is the position"
    I = np.eye(3)

    first_valid = next((v for v in z if not np.isnan(v)), 0.0)
    x = np.array([[first_valid], [0.0], [0.0]])                             # start: there, speed 0, accel 0
    P = np.diag([INIT_VAR_POS, INIT_VAR_VEL, INIT_VAR_ACC]).astype(float)   # how unsure we are of that guess

    xp = np.zeros((n, 3, 1)); Pp = np.zeros((n, 3, 3))                      # every frame's prediction
    xu = np.zeros((n, 3, 1)); Pu = np.zeros((n, 3, 3))                      # every frame's result after the update
    Fs = np.zeros((n, 3, 3))                                                # the F used to reach each frame

    for k in range(n):
        dt = dt_arr[k]
        F = _F(dt)
        Fs[k] = F
        x = F @ x                                   # prediction step, move the state with physics, F - constant motion model, x - previouse state.
        P = F @ P @ F.T + _Q(dt, jerk_psd)          # uncertainty carried along, grows by Q
        xp[k], Pp[k] = x, P
        if not np.isnan(z[k]):
            y = np.array([[z[k]]]) - H @ x          # surprise = measured − predicted position
            R = np.array([[r_std[k] ** 2]])         # this frame's measurement variance
            S = H @ P @ H.T + R                     # how big a surprise is normal - trust in the surprise
            K = P @ H.T @ np.linalg.inv(S)          # gain: 3 numbers (position, speed, accel) - K=1  means we trust the measurment fully.
            x = x + K @ y                           # update, all three from one surprise
            P = (I - K @ H) @ P
        xu[k], Pu[k] = x, P

    # RTS backward smoother (uses the same per-step transition that produced k+1)
    xs = xu.copy(); Ps = Pu.copy()                  # start from the forward results
    for k in range(n - 2, -1, -1):                  # walk backwards, second-to-last frame
        F = Fs[k + 1]
        C = Pu[k] @ F.T @ np.linalg.inv(Pp[k + 1])  # how strongly frame k should follow frame k+1
        xs[k] = xu[k] + C @ (xs[k + 1] - xp[k + 1]) # correct frame k with what the future showed
        Ps[k] = Pu[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T

    pos = xs[:, 0, 0]                               # final position, every frame
    vel = xs[:, 1, 0]                               # final velocity on this axis (m/s)
    vel_std = np.sqrt(np.clip(Ps[:, 1, 1], 0, None))    # its std = √variance; clip guards rounding below 0
    return pos, vel, vel_std


def _run_dts(run: "Run", frame_ts: dict[int, int] | None, fps: float) -> np.ndarray | None:
    """Per-step time deltas (s) for a run's frames, from real frame timestamps.
    Returns None when timestamps are unavailable (caller uses constant 1/fps)."""
    if frame_ts is None:
        return None
    try:
        ts = np.array([frame_ts[f] for f in run.frames], dtype=float) / 1e9  # s
    except KeyError:
        return None
    dts = np.empty(len(ts))
    dts[0] = 1.0 / fps
    dts[1:] = np.diff(ts)
    return dts


def kalman_speed_runs(runs: list[Run],
                      fps: float,
                      meas_noise_std: float = DEFAULT_MEAS_NOISE_STD_M,
                      jerk_psd: float = DEFAULT_JERK_PSD,
                      frame_ts: dict[int, int] | None = None
                      ) -> dict[int, tuple[float, float]]:
    """
    Process Runs whose .x (and optional .y) are world-position series.
    Returns {frame: (speed_mps, speed_std_mps)}.

    1-D run  -> speed = |velocity|.
    2-D run  -> speed = hypot(vx, vy); std combined by first-order propagation:
                var(speed) ~ (vx^2 var_vx + vy^2 var_vy) / (vx^2 + vy^2).

    `frame_ts` (frame -> timestamp_ns): when given, differentiation uses the REAL
    per-frame interval instead of a constant 1/fps (correct across variable fps/gaps).
    """
    out: dict[int, tuple[float, float]] = {}
    frames_done = 0
    for run in runs:
        dts = _run_dts(run, frame_ts, fps)
        # Optional per-frame down-weighting (e.g. wide-angle), stashed by build_world_runs.
        noise_scale = run.meta.get("noise_scale")
        _, vx, vx_std = kalman_velocity_1d(run.x, fps, meas_noise_std, jerk_psd, dts, noise_scale)
        if run.y is None:
            for i, f in enumerate(run.frames):
                out[f] = (abs(float(vx[i])), float(vx_std[i]))
        else:
            _, vy, vy_std = kalman_velocity_1d(run.y, fps, meas_noise_std, jerk_psd, dts, noise_scale)
            for i, f in enumerate(run.frames):
                sp = float(np.hypot(vx[i], vy[i]))
                denom = vx[i]**2 + vy[i]**2
                if denom > 1e-9:
                    var = (vx[i]**2 * vx_std[i]**2 + vy[i]**2 * vy_std[i]**2) / denom
                    std = float(np.sqrt(max(var, 0.0)))
                else:
                    std = float(max(vx_std[i], vy_std[i]))
                out[f] = (sp, std)
        frames_done += len(run.frames)
    print(f"[kalman] processed {len(runs)} runs, {frames_done} frames")
    return out


# ============================================================================
# Theil-Sen robust local slope
#
# Velocity at frame i = MEDIAN of the pairwise slopes (x_b - x_a)/(t_b - t_a)
# over a local window. Being a median of slopes, a minority of bad samples
# cannot drag it: smoothing AND outlier rejection in one step, ~one parameter,
# ~29% breakdown point. Cost: O(w^2) slope pairs per frame (keep the window
# small). No uncertainty -- use the Kalman path if you need a std.
# ============================================================================

THEILSEN_HALF_WINDOW = 11   # frames each side (~0.7s @30fps): slope-estimation span


def theil_sen_slope_1d(values: np.ndarray,
                       fps: float,
                       half_window: int = THEILSEN_HALF_WINDOW) -> np.ndarray:
    """
    Robust local velocity (units of `values` per second) for a 1-D series.

    NaNs are ignored within each window.  Windows with fewer than 2 valid
    samples yield 0.0.  Boundaries use a clipped (asymmetric) window.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    vel = np.zeros(n)
    if n < 2:
        return vel
    dt = 1.0 / fps

    for i in range(n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        idx = np.arange(lo, hi)
        vals = x[idx]
        ok = ~np.isnan(vals)
        idx, vals = idx[ok], vals[ok]
        if len(vals) < 2:
            continue
        # all pairwise slopes (a < b)
        slopes = []
        for a in range(len(vals)):
            dtime = (idx[a + 1:] - idx[a]) * dt
            dval = vals[a + 1:] - vals[a]
            slopes.append(dval / dtime)
        vel[i] = float(np.median(np.concatenate(slopes)))
    return vel


def theil_sen_speed_runs(runs: list[Run],
                         fps: float,
                         half_window: int = THEILSEN_HALF_WINDOW
                         ) -> dict[int, float]:
    """
    Process Runs whose .x (and optional .y) are world-position series.
    Returns {frame: speed_mps}.  (No uncertainty -- Theil-Sen is point-robust
    but does not produce a variance; use the Kalman path if you need one.)
    """
    out: dict[int, float] = {}
    frames_done = 0
    for run in runs:
        vx = theil_sen_slope_1d(run.x, fps, half_window)
        if run.y is None:
            for i, f in enumerate(run.frames):
                out[f] = abs(float(vx[i]))
        else:
            vy = theil_sen_slope_1d(run.y, fps, half_window)
            for i, f in enumerate(run.frames):
                out[f] = float(np.hypot(vx[i], vy[i]))
        frames_done += len(run.frames)
    print(f"[theil-sen] processed {len(runs)} runs, {frames_done} frames")
    return out


# ============================================================================
# Savitzky-Golay reference smoother
#
# Smooth the (Tx, Ty) world-position series, then differentiate once. Windows
# come from Constants (SMOOTH_WINDOW / DERIV_WINDOW / POLYORDER) -- the same
# values smooth_distances uses. Produces no uncertainty (std=0).
# ============================================================================

def savgol_speed_runs(runs: list[Run], fps: float) -> dict[int, tuple[float, float]]:
    """Reference smoother: smooth (Tx,Ty) then differentiate; std unavailable (0)."""

    #test_window = 121
    #twst_derv_window = 61

    out: dict[int, tuple[float, float]] = {}
    frames_done = 0
    for run in runs:
        Tx, Ty = run.x, run.y
        sw = min(_odd(Constants.SMOOTH_WINDOW), _odd(len(Tx) - 1))
        if sw >= Constants.POLYORDER + 2:
            Tx = savgol_filter(Tx, sw, Constants.POLYORDER)
            Ty = savgol_filter(Ty, sw, Constants.POLYORDER)
        Vx = savgol_filter(Tx, Constants.DERIV_WINDOW, Constants.POLYORDER, deriv=1, delta=1.0 / fps)
        Vy = savgol_filter(Ty, Constants.DERIV_WINDOW, Constants.POLYORDER, deriv=1, delta=1.0 / fps)
        for i, f in enumerate(run.frames):
            out[f] = (float(np.hypot(Vx[i], Vy[i])), 0.0)
        frames_done += len(run.frames)
    print(f"[savgol] processed {len(runs)} runs, {frames_done} frames")
    return out
