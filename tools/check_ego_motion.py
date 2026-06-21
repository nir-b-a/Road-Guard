#!/usr/bin/env python3
"""
Diagnose ego-motion reconstruction quality for an Android clip (no YOLO needed).

It (1) summarises the GPS-derived ego speed/heading/position the pipeline will use,
(2) leave-one-out cross-validates the sparse GPS speed interpolation, and
(3) runs a self-consistency check: a SYNTHETIC stationary target reconstructed with
    the pipeline's OWN ego model must read ~0 speed (proves the formula/signs are fine,
    so any real stationary-car error comes from the ego MODEL, i.e. GPS, not the math).

Run from the malshinon/ directory:
    python tools/check_ego_motion.py ../real_life_video_test
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # malshinon/
sys.path.insert(0, ROOT)

import Constants
from speed_estimation import ego_yaw
from speed_estimation.smoothers import Run, kalman_speed_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="clip folder (frames.csv, gyro.csv, gravity.csv, gps.csv)")
    ap.add_argument("--lat-sign", type=int, default=Constants.LAT_SIGN)
    ap.add_argument("--heading-sign", type=int, default=Constants.HEADING_SIGN)
    ap.add_argument("--target-depth", type=float, default=20.0,
                    help="synthetic stationary target start depth (m)")
    args = ap.parse_args()

    frames_csv = os.path.join(args.folder, "frames.csv")
    gyro_csv = os.path.join(args.folder, "gyro.csv")
    gravity_csv = os.path.join(args.folder, "gravity.csv")
    gps_csv = os.path.join(args.folder, "gps.csv")

    frame_ts = ego_yaw.load_frame_timestamps(frames_csv)
    frames = sorted(frame_ts)
    fts = np.array([frame_ts[f] for f in frames], dtype=np.int64)
    t = (fts - fts[0]) / 1e9
    fps = ego_yaw.fps_from_frame_timestamps(frames_csv) or 30.0
    print(f"frames: {len(frames)}   duration: {t[-1]:.2f}s   fps(median): {fps:.2f}")

    # --- GPS fixes ---
    gt, gv = [], []
    with open(gps_csv) as f:
        for row in csv.DictReader(f):
            gt.append(int(row["timestamp_ns"]))
            gv.append(float(row["speed_mps"]))
    gt = np.array(gt, float)
    gv = np.array(gv, float)
    order = np.argsort(gt)
    gt, gv = gt[order], gv[order]
    gt_rel = (gt - fts[0]) / 1e9
    print(f"\nGPS: {len(gt)} fixes over {gt_rel[-1] - gt_rel[0]:.1f}s "
          f"(median dt {np.median(np.diff(gt_rel)):.2f}s, rate {1.0/np.median(np.diff(gt_rel)):.2f} Hz)")
    for i in range(len(gt)):
        print(f"   t={gt_rel[i]:6.2f}s   v={gv[i]:.3f} m/s")

    # --- ego model the pipeline builds ---
    ego_speed = ego_yaw.ego_speed_from_android(frames_csv, gps_csv)
    ego_heading = ego_yaw.ego_heading_from_android(frames_csv, gyro_csv, gravity_csv,
                                                   sign=args.heading_sign)
    ego_pos = ego_yaw.ego_position_from_android(frames_csv, gps_csv, ego_heading)

    v = np.array([ego_speed[f] for f in frames])
    h = np.array([ego_heading[f] for f in frames])
    pos = np.array([ego_pos[f] for f in frames])
    path_len = float(np.sum(np.hypot(np.diff(pos[:, 0]), np.diff(pos[:, 1]))))
    print(f"\nego speed (interp/frame): min {v.min():.3f}  max {v.max():.3f}  mean {v.mean():.3f} m/s")
    print(f"ego heading: total change {np.degrees(h[-1]-h[0]):+.2f} deg, "
          f"range [{np.degrees(h.min()):.1f}, {np.degrees(h.max()):.1f}] deg")
    print(f"ego dead-reckoned path length: {path_len:.2f} m  "
          f"(net displacement {np.hypot(*(pos[-1]-pos[0])):.2f} m)")
    n_before = int(np.sum(fts < gt[0]))
    n_after = int(np.sum(fts > gt[-1]))
    print(f"frames using flat extrapolation (outside GPS range): "
          f"{n_before} before first fix, {n_after} after last fix")

    # --- (1) leave-one-out: can sparse GPS predict its own dropped fix? ---
    print("\n[leave-one-out] drop one GPS fix, interpolate it from the rest:")
    errs = []
    for i in range(1, len(gt) - 1):
        tt = np.delete(gt, i)
        vv = np.delete(gv, i)
        pred = float(np.interp(gt[i], tt, vv))
        errs.append(abs(pred - gv[i]))
        print(f"   drop t={gt_rel[i]:6.2f}s: actual {gv[i]:.3f}  interp {pred:.3f}  "
              f"err {pred - gv[i]:+.3f} m/s")
    if errs:
        print(f"   mean |error| = {np.mean(errs):.3f} m/s "
              f"({100*np.mean(errs)/max(gv.mean(),1e-6):.0f}% of mean speed)")

    # --- (2) self-consistency: synthetic STATIONARY target ---
    lat_sign = args.lat_sign
    D0 = args.target_depth
    ex0, ey0 = ego_pos[frames[0]]
    h0 = ego_heading[frames[0]]
    Wx, Wy = ex0 + D0 * np.cos(h0), ey0 + D0 * np.sin(h0)  # fixed world point ahead

    Tx = np.zeros(len(frames))
    Ty = np.zeros(len(frames))
    depth = np.zeros(len(frames))
    for k, f in enumerate(frames):
        ex, ey = ego_pos[f]
        th = ego_heading[f]
        dx, dy = Wx - ex, Wy - ey
        d = dx * np.cos(th) + dy * np.sin(th)              # camera-frame depth of the fixed point
        lat = lat_sign * (dx * np.sin(th) - dy * np.cos(th))
        depth[k] = d
        Tx[k] = ex + d * np.cos(th) + lat_sign * lat * np.sin(th)   # forward reconstruct
        Ty[k] = ey + d * np.sin(th) - lat_sign * lat * np.cos(th)

    sp = kalman_speed_runs([Run(frames=list(frames), x=Tx, y=Ty)], fps)
    spv = np.array([sp[f][0] for f in frames])
    print(f"\n[self-consistency] stationary target reconstructed with the pipeline's OWN ego model:")
    print(f"   reconstructed speed: max {spv.max()*3.6:.4f} km/h, mean {spv.mean()*3.6:.4f} km/h "
          f"(~0 => formula & signs are correct)")
    print(f"   its camera depth closes {depth[0]:.2f} -> {depth[-1]:.2f} m "
          f"(this is the ego forward motion the GPS model must match)")


if __name__ == "__main__":
    main()
