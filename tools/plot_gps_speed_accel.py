#!/usr/bin/env python3
"""
plot_gps_speed_accel.py -- ego GPS speed and the accelerometer stream on one time axis.

WHAT IT DRAWS
    Three stacked panels sharing the same time axis (seconds from the first video frame,
    or from the first sensor sample when the session has no frames.csv):
      1. GPS speed (km/h) from gps.csv, one dot per fix
      2. linacc.csv ax / ay / az (m/s^2, device frame), raw samples faint + a centred
         moving average of --smooth-s on top
      3. |a| (smoothed accelerometer magnitude) next to the GPS-derived dv/dt, both m/s^2,
         so speed changes can be matched against what the accelerometer felt

    linacc.csv is Android's TYPE_LINEAR_ACCELERATION (gravity already removed), so a car
    at constant speed reads ~0 on every axis. Sample gaps longer than --gap-s are drawn
    as breaks instead of straight lines across the gap.

OUTPUTS
    <session>/gps_speed_accel.png   (or --out)

RUN (from the malshinon/ directory)
    python tools/plot_gps_speed_accel.py test_videos/chase_vid
    python tools/plot_gps_speed_accel.py SESSION --smooth-s 0.5 --start 20 --end 80
    python tools/plot_gps_speed_accel.py SESSION --show
"""

import argparse
import csv
import os

import numpy as np

# Chart colours: categorical slots 1-3 (blue, orange, aqua) + ink/grid chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"


# ============================================================================
# Loading
# ============================================================================

def read_columns(path: str, columns: tuple) -> dict:
    """CSV -> {column: float array}, rows sorted by timestamp_ns; unparsable rows skipped."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(tuple(float(r[c]) for c in columns))
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit(f"[plot] {path}: no usable rows with columns {columns}")
    arr = np.array(sorted(rows, key=lambda x: x[0]), dtype=float)
    return {c: arr[:, i] for i, c in enumerate(columns)}


def first_frame_ns(session: str):
    """Timestamp of the first video frame, or None when frames.csv is missing/empty."""
    path = os.path.join(session, "frames.csv")
    if not os.path.exists(path):
        return None
    ts = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                ts.append(int(r["timestamp_ns"]))
            except (KeyError, TypeError, ValueError):
                continue
    return min(ts) if ts else None


# ============================================================================
# Signal helpers
# ============================================================================

def moving_average(t_s: np.ndarray, y: np.ndarray, window_s: float) -> np.ndarray:
    """Centred moving average over ~window_s, sized from the median sample period."""
    if window_s <= 0 or len(y) < 3:
        return y.copy()
    dt = float(np.median(np.diff(t_s)))
    n = max(1, int(round(window_s / dt))) if dt > 0 else 1
    if n <= 1:
        return y.copy()
    kernel = np.ones(n) / n
    pad = n // 2
    padded = np.pad(y, (pad, n - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def break_gaps(t_s: np.ndarray, *series, gap_s: float):
    """Insert a NaN row after every dt > gap_s so matplotlib draws a break there."""
    if len(t_s) < 2:
        return (t_s, *series)
    idx = np.where(np.diff(t_s) > gap_s)[0] + 1
    if len(idx) == 0:
        return (t_s, *series)
    t_out = np.insert(t_s, idx, np.nan)
    return (t_out, *(np.insert(s, idx, np.nan) for s in series))


def gps_dvdt(t_s: np.ndarray, speed_mps: np.ndarray, gap_s: float):
    """Central-difference d(speed)/dt at each fix; NaN where a neighbour is beyond gap_s."""
    a = np.full(len(t_s), np.nan)
    for k in range(1, len(t_s) - 1):
        dt = t_s[k + 1] - t_s[k - 1]
        if 0 < dt <= 2 * gap_s:
            a[k] = (speed_mps[k + 1] - speed_mps[k - 1]) / dt
    return a


def crop(t_s: np.ndarray, start, end) -> np.ndarray:
    keep = np.ones(len(t_s), dtype=bool)
    if start is not None:
        keep &= t_s >= start
    if end is not None:
        keep &= t_s <= end
    return keep


# ============================================================================
# Plot
# ============================================================================

def style_axis(ax, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)


def plot(session: str, gps: dict, acc: dict, t0_label: str, args):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.2, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    ax_v, ax_xyz, ax_mag = axes

    # --- 1. GPS speed -------------------------------------------------------
    t_g, v_kmh = break_gaps(gps["t_s"], gps["speed_mps"] * 3.6, gap_s=args.gps_gap_s)
    ax_v.plot(t_g, v_kmh, color=BLUE, linewidth=2)
    ax_v.plot(gps["t_s"], gps["speed_mps"] * 3.6, "o", color=BLUE, markersize=4,
              markeredgecolor=SURFACE, markeredgewidth=1)
    style_axis(ax_v, "GPS speed (km/h)")
    ax_v.set_ylim(bottom=0)
    ax_v.set_title(f"Ego GPS speed  ({len(gps['t_s'])} fixes, "
                   f"max {np.nanmax(gps['speed_mps']) * 3.6:.1f} km/h)",
                   loc="left", color=INK, fontsize=11)

    # --- 2. accelerometer axes ----------------------------------------------
    smooth = {c: moving_average(acc["t_s"], acc[c], args.smooth_s) for c in ("ax", "ay", "az")}
    for c, color in (("ax", BLUE), ("ay", ORANGE), ("az", AQUA)):
        t_r, raw = break_gaps(acc["t_s"], acc[c], gap_s=args.gap_s)
        t_m, sm = break_gaps(acc["t_s"], smooth[c], gap_s=args.gap_s)
        ax_xyz.plot(t_r, raw, color=color, linewidth=0.6, alpha=0.25)
        ax_xyz.plot(t_m, sm, color=color, linewidth=2, label=c)
    ax_xyz.axhline(0, color=AXIS, linewidth=1)
    style_axis(ax_xyz, "linacc (m/s²)")
    ax_xyz.legend(loc="upper right", ncol=3, frameon=False, labelcolor=INK_2)
    ax_xyz.set_title(f"Linear acceleration, device frame  ({len(acc['t_s'])} samples, "
                     f"{args.smooth_s:g} s moving average)",
                     loc="left", color=INK, fontsize=11)
    if args.accel_ylim:
        ax_xyz.set_ylim(-args.accel_ylim, args.accel_ylim)
    else:
        # Scale to the smoothed traces; single raw spikes would otherwise flatten them.
        lim = 1.5 * max(np.nanpercentile(np.abs(smooth[c]), 99.5) for c in ("ax", "ay", "az"))
        ax_xyz.set_ylim(-max(lim, 0.5), max(lim, 0.5))

    # --- 3. |a| vs GPS dv/dt ------------------------------------------------
    mag = np.sqrt(smooth["ax"] ** 2 + smooth["ay"] ** 2 + smooth["az"] ** 2)
    t_m, mag = break_gaps(acc["t_s"], mag, gap_s=args.gap_s)
    ax_mag.plot(t_m, mag, color=AQUA, linewidth=2, label="|a| accelerometer (smoothed)")
    dvdt = gps_dvdt(gps["t_s"], gps["speed_mps"], args.gps_gap_s)
    ax_mag.plot(gps["t_s"], dvdt, color=BLUE, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=1, label="GPS dv/dt")
    ax_mag.axhline(0, color=AXIS, linewidth=1)
    style_axis(ax_mag, "m/s²")
    ax_mag.legend(loc="upper right", ncol=2, frameon=False, labelcolor=INK_2)
    ax_mag.set_title("Accelerometer magnitude vs GPS speed change", loc="left",
                     color=INK, fontsize=11)
    if args.accel_ylim:
        ax_mag.set_ylim(-args.accel_ylim, args.accel_ylim)
    ax_mag.set_xlabel(f"time (s) from {t0_label}", color=INK_2)

    fig.suptitle(os.path.basename(os.path.normpath(session)), x=0.01, ha="left",
                 color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = args.out or os.path.join(session, "gps_speed_accel.png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"[plot] wrote {out}")
    if args.show:
        plt.show()
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Plot ego GPS speed and accelerometer data for a session.")
    ap.add_argument("session", help="session folder (gps.csv, linacc.csv, optionally frames.csv)")
    ap.add_argument("--out", default=None, help="output PNG (default <session>/gps_speed_accel.png)")
    ap.add_argument("--accel-csv", default="linacc.csv",
                    help="accelerometer file name inside the session (default linacc.csv)")
    ap.add_argument("--smooth-s", type=float, default=0.25,
                    help="accelerometer moving-average window in seconds (0 = off)")
    ap.add_argument("--start", type=float, default=None, help="crop: first second to draw")
    ap.add_argument("--end", type=float, default=None, help="crop: last second to draw")
    ap.add_argument("--accel-ylim", type=float, default=None,
                    help="symmetric y-limit for the m/s^2 panels (default auto)")
    ap.add_argument("--gap-s", type=float, default=0.5,
                    help="accelerometer gaps longer than this are drawn as breaks")
    ap.add_argument("--gps-gap-s", type=float, default=2.5,
                    help="GPS gaps longer than this are drawn as breaks")
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    args = ap.parse_args()

    session = args.session
    gps_path = os.path.join(session, "gps.csv")
    acc_path = os.path.join(session, args.accel_csv)
    for p in (gps_path, acc_path):
        if not os.path.exists(p):
            raise SystemExit(f"[plot] missing file: {p}")

    gps = read_columns(gps_path, ("timestamp_ns", "speed_mps"))
    acc = read_columns(acc_path, ("timestamp_ns", "ax", "ay", "az"))

    t0 = first_frame_ns(session)
    t0_label = "first video frame"
    if t0 is None:
        t0 = min(gps["timestamp_ns"][0], acc["timestamp_ns"][0])
        t0_label = "first sensor sample"
    gps["t_s"] = (gps["timestamp_ns"] - t0) / 1e9
    acc["t_s"] = (acc["timestamp_ns"] - t0) / 1e9

    # Crop after smoothing would need the neighbours; cropping first is fine for a view.
    for d in (gps, acc):
        keep = crop(d["t_s"], args.start, args.end)
        for k in list(d):
            d[k] = d[k][keep]
    if len(gps["t_s"]) == 0 or len(acc["t_s"]) == 0:
        raise SystemExit("[plot] nothing left to draw after --start/--end cropping")

    print(f"[plot] gps: {len(gps['t_s'])} fixes over {gps['t_s'][-1] - gps['t_s'][0]:.1f} s, "
          f"speed {np.nanmin(gps['speed_mps']) * 3.6:.1f}-{np.nanmax(gps['speed_mps']) * 3.6:.1f} km/h")
    rate = (len(acc["t_s"]) - 1) / (acc["t_s"][-1] - acc["t_s"][0]) if len(acc["t_s"]) > 1 else 0.0
    print(f"[plot] accel: {len(acc['t_s'])} samples over {acc['t_s'][-1] - acc['t_s'][0]:.1f} s "
          f"(~{rate:.0f} Hz)")

    plot(session, gps, acc, t0_label, args)


if __name__ == "__main__":
    main()
