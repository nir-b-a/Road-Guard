"""
Calibrates the distance estimation model using CARLA ground-truth data.

Matching strategy:
  Instead of relying on YOLO tracking IDs (which change on occlusion),
  this script projects each CARLA target's known 3D world position into
  image pixels per frame and picks whichever YOLO detection is closest
  to that projected point. ID-independent.

Usage:
  Edit the SIMULATIONS list at the bottom to include every simulation you
  have run. Each entry is a (telemetry_csv, bboxes_csv) pair:
    telemetry_csv -- from calc_distance_to_csv.py on the CARLA telemetry
    bboxes_csv    -- from main.py -> distanceLogger.export_all_bboxes

  Run once after all simulations are done. Data from all simulations is
  pooled into a single fit, so the model generalises across scenarios.
  Each run completely overwrites distance_model.py -- no need to reset it.

Output:
  distance_model.py  -- coefficients used by speed_estimator.py
  calibration_fit.png
"""

import csv
import math
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Camera parameters -- must match the CARLA camera used in all simulations.
IMAGE_W      = 1920
IMAGE_H      = 1080
FOV_H_DEG    = 90.0
CAM_HEIGHT_M = 1.2   # metres above vehicle reference point


def _focal_length():
    return (IMAGE_W / 2) / math.tan(math.radians(FOV_H_DEG / 2))


def _load_telemetry(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "frame":  int(row["frame"]),
                "ego":    (float(row["ego_x"]),    float(row["ego_y"]),    float(row["ego_z"])),
                "target": (float(row["target_x"]), float(row["target_y"]), float(row["target_z"])),
                "speed":  float(row["target_speed_mps"]),
            })

    telemetry = {}
    for i, r in enumerate(rows):
        ego, target = r["ego"], r["target"]
        dx = target[0] - ego[0]
        dy = target[1] - ego[1]
        dz = target[2] - ego[2]
        real_dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if i > 0:
            prev_ego = rows[i-1]["ego"]
            move_dx = ego[0] - prev_ego[0]
            move_dy = ego[1] - prev_ego[1]
            if math.sqrt(move_dx**2 + move_dy**2) > 0.001:
                yaw = math.atan2(move_dy, move_dx)
            else:
                yaw = math.atan2(dy, dx)
        else:
            yaw = math.atan2(dy, dx)

        telemetry[r["frame"]] = {
            "ego": ego, "target": target,
            "yaw": yaw, "real_dist": real_dist, "speed": r["speed"],
        }
    return telemetry


def _project_to_image(ego, target, yaw):
    """
    Projects target's 3D world position to image pixel (u, v).
    Returns None if the target is behind the camera.
    """
    fx = _focal_length()
    cx, cy = IMAGE_W / 2.0, IMAGE_H / 2.0

    dx = target[0] - ego[0]
    dy = target[1] - ego[1]
    dz = target[2] - (ego[2] + CAM_HEIGHT_M)

    body_fwd   =  dx * math.cos(yaw) + dy * math.sin(yaw)
    body_right = -dx * math.sin(yaw) + dy * math.cos(yaw)
    body_up    = dz

    if body_fwd <= 0.1:
        return None

    u = cx + fx * (body_right / body_fwd)
    v = cy - fx * (body_up   / body_fwd)
    return (u, v)


def _load_all_bboxes(path):
    bboxes: dict[int, list[tuple[int, int, int, int]]] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            box = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            bboxes.setdefault(frame, []).append(box)
    return bboxes


def _find_closest_bbox(uv, boxes, max_dist_px=300):
    if uv is None or not boxes:
        return None
    best, best_d = None, float("inf")
    for (x1, y1, x2, y2) in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        d = math.sqrt((cx - uv[0])**2 + (cy - uv[1])**2)
        if d < best_d:
            best_d, best = d, (x1, y1, x2, y2)
    return best if best_d <= max_dist_px else None


def _corrected_pinhole(ph, a, b):
    return a / (ph + b)


def _power_law(ph, a, b):
    return a * np.power(ph, b)


def _collect_from_simulation(telemetry_csv: str, bboxes_csv: str):
    """
    Match YOLO detections to the CARLA target for one simulation.
    Returns (pixel_heights, real_distances).
    """
    telemetry  = _load_telemetry(telemetry_csv)
    all_bboxes = _load_all_bboxes(bboxes_csv)

    matched_ph: list[float] = []
    matched_rd: list[float] = []
    unmatched = 0

    for frame, telem in sorted(telemetry.items()):
        boxes = all_bboxes.get(frame, [])
        uv    = _project_to_image(telem["ego"], telem["target"], telem["yaw"])
        bbox  = _find_closest_bbox(uv, boxes)

        if bbox is None:
            unmatched += 1
            continue

        ph = bbox[3] - bbox[1]
        if ph <= 0:
            continue

        matched_ph.append(float(ph))
        matched_rd.append(telem["real_dist"])

    print(f"  {telemetry_csv}: {len(matched_ph)} matched, {unmatched} skipped")
    return matched_ph, matched_rd


def fit_and_report(simulations: list[tuple[str, str]]):
    """
    Pool data from ALL simulations, then fit one model on the combined dataset.

    simulations: list of (telemetry_csv, bboxes_csv) pairs.

    Running this on all simulations together -- rather than once per simulation
    -- is what makes the model generalise across different scenarios and vehicle
    types. Every call completely overwrites distance_model.py; no manual reset
    is needed between runs.
    """
    all_ph: list[float] = []
    all_rd: list[float] = []

    print("Collecting data from simulations:")
    for telemetry_csv, bboxes_csv in simulations:
        ph, rd = _collect_from_simulation(telemetry_csv, bboxes_csv)
        all_ph.extend(ph)
        all_rd.extend(rd)

    print(f"\nTotal matched frames across all simulations: {len(all_ph)}")
    if len(all_ph) < 10:
        print("ERROR: too few matched frames -- check camera params or file paths.")
        return

    ph = np.array(all_ph, dtype=float)
    rd = np.array(all_rd, dtype=float)

    # corrected pinhole
    p0_cp = [_focal_length() * 1.5, 0.0]
    try:
        popt_cp, _ = curve_fit(_corrected_pinhole, ph, rd, p0=p0_cp, maxfev=20000)
    except RuntimeError:
        popt_cp = p0_cp
    a_cp, b_cp = popt_cp
    rmse_cp = math.sqrt(np.mean((_corrected_pinhole(ph, a_cp, b_cp) - rd) ** 2))
    print(f"\nCorrected pinhole:  d = {a_cp:.2f} / (px_h + {b_cp:.2f})   RMSE={rmse_cp:.3f} m")

    # power law
    p0_pl = [10000.0, -1.0]
    try:
        popt_pl, _ = curve_fit(_power_law, ph, rd, p0=p0_pl, maxfev=20000)
    except RuntimeError:
        popt_pl = p0_pl
    a_pl, b_pl = popt_pl
    rmse_pl = math.sqrt(np.mean((_power_law(ph, a_pl, b_pl) - rd) ** 2))
    print(f"Power law:          d = {a_pl:.2f} * px_h^{b_pl:.4f}          RMSE={rmse_pl:.3f} m")

    best = "corrected_pinhole" if rmse_cp <= rmse_pl else "power_law"
    print(f"\nBest model: {best}")

    _write_model(a_cp, b_cp, a_pl, b_pl, best)
    _plot(ph, rd, a_cp, b_cp, rmse_cp, a_pl, b_pl, rmse_pl)


def _write_model(a_cp, b_cp, a_pl, b_pl, best):
    fl = _focal_length()
    h_eff = a_cp / fl
    content = (
        "# Auto-generated by calibrate_distance.py -- do not edit manually.\n\n"
        f'BEST_MODEL = "{best}"\n\n'
        "# Effective vehicle height as seen by YOLOv8 bounding boxes (metres).\n"
        "# Camera-independent: multiply by the real focal length to get A.\n"
        f"H_EFF = {h_eff}\n\n"
        "# Focal length used during calibration (pixels). Reference only.\n"
        f"CALIBRATION_FOCAL_LENGTH_PX = {fl}\n\n"
        "# Bounding-box padding offset (pixels). Driven by YOLO behaviour,\n"
        "# not by the camera -- stays valid across different cameras.\n"
        f"B_CP = {b_cp}\n\n"
        "# Precomputed A for the calibration camera (H_EFF * CALIBRATION_FOCAL_LENGTH_PX).\n"
        f"A_CP = {a_cp}\n\n"
        f"A_PL = {a_pl}\n"
        f"B_PL = {b_pl}\n"
    )
    with open("distance_model.py", "w") as f:
        f.write(content)
    print("Coefficients written to distance_model.py")


def _plot(ph, rd, a_cp, b_cp, rmse_cp, a_pl, b_pl, rmse_pl):
    idx  = np.argsort(ph)
    ph_s = ph[idx]

    plt.figure(figsize=(10, 5))
    plt.scatter(ph_s, rd[idx], s=5, label="Ground truth (matched)", alpha=0.5)
    plt.plot(ph_s, _corrected_pinhole(ph_s, a_cp, b_cp),
             label=f"Corrected pinhole  RMSE={rmse_cp:.2f} m")
    plt.plot(ph_s, _power_law(ph_s, a_pl, b_pl), linestyle="--",
             label=f"Power law  RMSE={rmse_pl:.2f} m")
    plt.xlabel("Bounding box pixel height")
    plt.ylabel("Real distance (m)")
    plt.title("Calibration: pixel height vs real distance (all simulations pooled)")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig("calibration_fit.png", dpi=120)
    plt.show()
    print("Plot saved to calibration_fit.png")


if __name__ == "__main__":
    # Add one tuple per simulation you have run.
    # telemetry_csv -- output of calc_distance_to_csv.py
    # bboxes_csv    -- output of main.py -> distanceLogger.export_all_bboxes
    SIMULATIONS = [
        ("telemetry_0.csv", "dashcam_1_60sec_bboxes.csv"),
        # ("telemetry_1.csv", "sim_1_bboxes.csv"),
        # ("telemetry_2.csv", "sim_2_bboxes.csv"),
    ]
    fit_and_report(SIMULATIONS)
