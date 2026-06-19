"""
Per-vehicle output plots.

After speed estimation, every vehicle that survived the short-track filter (i.e.
has a persisted speed series) gets ONE PNG with two stacked subplots:

    top    -- distance (m):  estimated vs. real
    bottom -- speed (km/h):  estimated vs. real  (+/-1 std band)

"Real" is the CARLA telemetry vehicle (car_1). A simulation has exactly ONE real
vehicle, so its real distance/speed is overlaid on EVERY vehicle's graph (in
--simulation) -- so you can compare each detected track against the single ground
truth. Each subplot title carries the vehicle ID and its resolved class.

File naming: {video_name}_vehicle_{id}.png, next to the other outputs.

Why one PNG per vehicle (not a multi-vehicle overlay or a CSV): it keeps a single
vehicle's distance+speed together and individually addressable, and avoids mixing
the one calibrated (telemetry) trace with many uncalibrated ones on shared axes.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")          # headless backend: save PNGs without a display
import matplotlib.pyplot as plt

from Constants import class_name

MPS_TO_KMH = 3.6


def _load_real_target(telemetry_csv: str) -> tuple[list[int], list[float], list[float]]:
    """
    Read the telemetry target's real (frame, speed_kmh, dist_m).

    Speed column: car_1_speed_kmh (current) or target_speed_mps (old, m/s->km/h).
    Distance column: car_1_dist_to_ego_m (forward distance to ego, metres).
    Missing values become NaN so they simply don't draw.
    """
    frames: list[int] = []
    speed_kmh: list[float] = []
    dist_m: list[float] = []
    with open(telemetry_csv, "r") as f:
        reader = csv.DictReader(f)
        spd_col, spd_scale, dist_col = None, 1.0, None
        for row in reader:
            if spd_col is None:
                if "car_1_speed_kmh" in row:
                    spd_col, spd_scale = "car_1_speed_kmh", 1.0
                elif "target_speed_mps" in row:
                    spd_col, spd_scale = "target_speed_mps", MPS_TO_KMH
                if "car_1_dist_to_ego_m" in row:
                    dist_col = "car_1_dist_to_ego_m"
            try:
                frames.append(int(row["frame"]))
            except (KeyError, ValueError):
                continue
            try:
                speed_kmh.append(float(row[spd_col]) * spd_scale if spd_col else float("nan"))
            except (KeyError, ValueError):
                speed_kmh.append(float("nan"))
            try:
                dist_m.append(float(row[dist_col]) if dist_col else float("nan"))
            except (KeyError, ValueError):
                dist_m.append(float("nan"))
    return frames, speed_kmh, dist_m


def plot_all_vehicles(world, video_dir: str, video_name: str,
                      telemetry_csv: str | None = None) -> None:
    """
    Write one distance+speed PNG per surviving vehicle (those with a persisted
    speed series). When telemetry_csv is given (simulation), the single real
    vehicle's distance/speed is overlaid on EVERY vehicle's graph; pass
    telemetry_csv=None for real-world video (no overlay).
    """
    real_frames = real_speed = real_dist = None
    if telemetry_csv and os.path.exists(telemetry_csv):
        real_frames, real_speed, real_dist = _load_real_target(telemetry_csv)

    survivors = [(vid, v) for vid, v in world.vehicles.items() if v.speed_per_frame]
    if not survivors:
        print("[vehicle_plots] no vehicles survived the speed filter; nothing to plot")
        return

    for vid, vehicle in survivors:
        # Estimated distance (m): only frames with a real (positive) value.
        d_frames = sorted(f for f, d in vehicle.dist_per_frame.items() if d > 0)
        d_vals = [vehicle.dist_per_frame[f] for f in d_frames]

        # Estimated speed (km/h) + uncertainty.
        s_frames = sorted(vehicle.speed_per_frame)
        s_vals = [vehicle.speed_per_frame[f] * MPS_TO_KMH for f in s_frames]
        s_std = [vehicle.speed_std_per_frame.get(f, 0.0) * MPS_TO_KMH for f in s_frames]

        cls = class_name(vehicle.vehicle_type)

        fig, (ax_d, ax_s) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # --- distance ---
        ax_d.plot(d_frames, d_vals, label="Estimated", color="tab:blue", alpha=0.9)
        if real_dist is not None:
            ax_d.plot(real_frames, real_dist, label="Real (telemetry)",
                      color="tab:orange", linewidth=2)
        ax_d.set_ylabel("Distance (m)")
        ax_d.set_title(f"Vehicle {vid} ({cls}) -- distance")
        ax_d.grid(True)
        ax_d.legend()

        # --- speed ---
        ax_s.plot(s_frames, s_vals, label="Estimated", color="tab:blue", alpha=0.9)
        if any(st > 0 for st in s_std):
            lo = [v - st for v, st in zip(s_vals, s_std)]
            hi = [v + st for v, st in zip(s_vals, s_std)]
            ax_s.fill_between(s_frames, lo, hi, alpha=0.2, color="tab:blue", label="+/-1 std")
        if real_speed is not None:
            ax_s.plot(real_frames, real_speed, label="Real (telemetry)",
                      color="tab:orange", linewidth=2)
        ax_s.set_ylabel("Speed (km/h)")
        ax_s.set_xlabel("Frame")
        ax_s.set_title(f"Vehicle {vid} ({cls}) -- speed")
        ax_s.grid(True)
        ax_s.legend()

        fig.tight_layout()
        out_path = os.path.join(video_dir, f"{video_name}_vehicle_{vid}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    print(f"[vehicle_plots] wrote {len(survivors)} per-vehicle plot(s) to {video_dir}")


def plot_ego_speed(ego_speed: dict[int, float], video_dir: str, video_name: str,
                   source: str = "",
                   *,
                   ego_speed_raw: dict[int, float] | None = None,
                   raw_source: str = "") -> None:
    """
    Write a single PNG of the EGO vehicle's own speed (km/h) over frames -- the
    counterpart to plot_all_vehicles, but for the camera car instead of the
    tracked vehicles.

    `ego_speed` is {frame -> speed (m/s)}: the SERIES THE ESTIMATOR ACTUALLY USES
    for ego motion -- CARLA telemetry in --simulation, and on an Android clip the
    accelerometer+GPS FUSED speed when fusion engaged (else GPS). `source` annotates
    it (e.g. "telemetry" / "accelerometer+GPS fused" / "GPS").

    `ego_speed_raw` (optional): the ORIGINAL, unfused series to overlay for
    comparison -- on Android this is the raw GPS-only interpolation, so the plot
    shows exactly how much the accelerometer fusion changed the ego speed the
    estimator consumes. Omitted (None) -> single trace, as before.

    File naming: {video_name}_ego_speed.png, next to the other outputs.
    """
    if not ego_speed:
        print("[ego_plot] no ego speed series available; nothing to plot")
        return

    frames = sorted(ego_speed)
    speed_kmh = [ego_speed[f] * MPS_TO_KMH for f in frames]

    fig, ax = plt.subplots(figsize=(10, 4))
    # Draw the raw/original series first (underneath), fainter, so the fused series
    # the estimator uses reads as the primary trace on top.
    if ego_speed_raw:
        raw_frames = sorted(ego_speed_raw)
        raw_kmh = [ego_speed_raw[f] * MPS_TO_KMH for f in raw_frames]
        raw_label = f"Ego speed ({raw_source})" if raw_source else "Ego speed (raw)"
        ax.plot(raw_frames, raw_kmh, color="tab:gray", linewidth=1.5,
                linestyle="--", alpha=0.8, label=raw_label)

    label = f"Ego speed ({source})" if source else "Ego speed"
    ax.plot(frames, speed_kmh, color="tab:green", linewidth=2, label=label)
    ax.set_ylabel("Speed (km/h)")
    ax.set_xlabel("Frame")
    ax.set_title("Ego vehicle -- speed")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(video_dir, f"{video_name}_ego_speed.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[ego_plot] wrote ego speed plot to {out_path}")
