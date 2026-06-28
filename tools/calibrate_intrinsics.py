#!/usr/bin/env python3
"""
Checkerboard camera calibration for the Road Guard Android capture.

Measures the TRUE pinhole intrinsics (fx, fy, cx, cy) + lens distortion for the
phone's ACTUAL 1080p video stream, replacing the app's metadata estimate (which
assumes the 16:9 video uses the full sensor width — wrong if CameraX crops/zooms).
This is the gold-standard fix: the result is measured, not assumed.

--------------------------------------------------------------------------------
HOW TO USE
--------------------------------------------------------------------------------
1. Print a checkerboard (e.g. the classic 10x7-squares board => 9x6 INNER corners)
   and tape it FLAT to a rigid surface (a wall or stiff board — no waves).

2. Record a calibration clip *with this same app* (so it is the exact same camera
   config: 1080p, same FOV/crop as your real drives). Slowly move the phone so the
   board appears:
     - near and far,
     - tilted left/right/up/down (~30-45 deg),
     - in the corners and centre of the frame.
   Aim for 15-30 seconds. Hold steady for a moment at each pose (avoid motion blur).

3. Pull the clip to your PC and run (from the malshinon/ directory):

     python tools/calibrate_intrinsics.py path/to/calib.mp4 --cols 9 --rows 6

   --cols / --rows are the number of INNER corners (squares - 1), not squares.

4. Paste the printed values into RecordingActivity.kt's CALIB_FX/FY/CX/CY constants,
   OR drop the written intrinsics.json next to a video. The pipeline reads fx/fy/cx/cy.

--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------
- The calibration clip MUST match the real recordings' resolution AND FOV. If you
  change the app's camera config (resolution, ViewPort, use cases), recalibrate.
- A good calibration has RMS reprojection error < ~0.5 px and >= ~10-15 views.
- Square size does NOT affect fx/fy/cx/cy (only extrinsics scale), so it is left at 1.
- The Road Guard pipeline uses a pinhole model WITHOUT distortion. The reported
  distortion tells you whether that is safe: small k1 -> fine; large k1 -> consider
  undistorting frames before the pipeline (see the printed hint).
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def iter_frames(source: str, sample_every: int):
    """Yield (name, BGR image) from a video file or a folder of images."""
    if os.path.isdir(source):
        files = sorted(
            f for f in glob.glob(os.path.join(source, "*"))
            if f.lower().endswith(IMAGE_EXTS)
        )
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                yield os.path.basename(f), img
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[error] could not open video: {source}")
        sys.exit(1)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % sample_every == 0:
            yield f"frame_{i:06d}", frame
        i += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser(
        description="Checkerboard intrinsics calibration for the Android 1080p stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("source", help="calibration video file OR folder of images")
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row (squares_per_row - 1)")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column (squares_per_col - 1)")
    ap.add_argument("--square", type=float, default=1.0, help="square size (any unit; does not affect fx/fy/cx/cy)")
    ap.add_argument("--sample-every", type=int, default=10, help="use every Nth video frame")
    ap.add_argument("--max-views", type=int, default=40, help="stop after this many good detections")
    ap.add_argument("--out", default=None, help="path to write the drop-in intrinsics.json (default: next to source)")
    ap.add_argument("--save-detections", default=None, help="dir to dump corner-overlay images for visual QC")
    args = ap.parse_args()

    pattern = (args.cols, args.rows)  # (cols, rows) of inner corners

    # 3D object points for one board view: (0,0,0),(1,0,0),...,(cols-1,rows-1,0)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square

    objpoints: list = []
    imgpoints: list = []
    img_size = None  # (w, h)
    scanned = 0
    found = 0

    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    find_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                  + cv2.CALIB_CB_NORMALIZE_IMAGE
                  + cv2.CALIB_CB_FAST_CHECK)

    if args.save_detections:
        os.makedirs(args.save_detections, exist_ok=True)

    for name, img in iter_frames(args.source, args.sample_every):
        scanned += 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if img_size is None:
            img_size = (w, h)
        elif (w, h) != img_size:
            print(f"[warn] {name}: size {w}x{h} != {img_size[0]}x{img_size[1]}; skipping")
            continue

        ok, corners = cv2.findChessboardCorners(gray, pattern, find_flags)
        if not ok:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        found += 1

        if args.save_detections:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern, corners, ok)
            cv2.imwrite(os.path.join(args.save_detections, f"{name}_corners.jpg"), vis)

        if found >= args.max_views:
            break

    print(f"[calib] scanned {scanned} frames, detected the {args.cols}x{args.rows} "
          f"board in {found} of them")
    if found < 8:
        print(f"[error] only {found} usable views (need >= ~10-15). Record more poses: "
              f"vary distance, tilt, and position in the frame; avoid motion blur. "
              f"Also double-check --cols/--rows are INNER corners.")
        sys.exit(1)

    print(f"[calib] calibrating at {img_size[0]}x{img_size[1]} ...")
    rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    dist = dist.ravel().tolist()  # [k1, k2, p1, p2, k3]

    print("\n================ CALIBRATION RESULT ================")
    print(f"resolution        : {img_size[0]} x {img_size[1]}")
    print(f"views used        : {found}")
    print(f"RMS reproj error  : {rms:.4f} px   ({'GOOD' if rms < 0.6 else 'high — recapture for better accuracy' if rms > 1.0 else 'ok'})")
    print(f"fx, fy            : {fx:.4f}, {fy:.4f}")
    print(f"cx, cy            : {cx:.4f}, {cy:.4f}")
    hfov = 2.0 * np.degrees(np.arctan2(img_size[0] / 2.0, fx))
    print(f"implied HFOV      : {hfov:.2f} deg")
    print(f"distortion k1,k2,p1,p2,k3 : {[round(d, 5) for d in dist]}")
    if abs(dist[0]) > 0.15:
        print("  [hint] |k1| is sizeable -> the lens has noticeable distortion. The pipeline\n"
              "         assumes a pinhole; for best accuracy undistort frames with this K+dist\n"
              "         before running main.py (cv2.undistort), or accept a small residual error.")
    print("====================================================")

    if img_size != (1920, 1080):
        print(f"[warn] calibrated at {img_size[0]}x{img_size[1]}, NOT 1920x1080. The app records\n"
              f"       1080p, so calibrate a 1080p clip or the intrinsics will not match.")

    # 1) drop-in intrinsics.json (exactly the 4 keys the pipeline reads)
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.source)) if not os.path.isdir(args.source)
        else os.path.abspath(args.source),
        "intrinsics.json",
    )
    with open(out_path, "w") as f:
        json.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy}, f)
    print(f"\n[write] drop-in intrinsics.json -> {out_path}")

    # 2) full report sidecar (distortion + metadata, for the record)
    report_path = os.path.splitext(out_path)[0] + "_calibration_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "resolution": {"width": img_size[0], "height": img_size[1]},
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "distortion": {"k1": dist[0], "k2": dist[1], "p1": dist[2], "p2": dist[3], "k3": dist[4]},
            "rms_reprojection_px": rms,
            "views_used": found,
            "implied_hfov_deg": hfov,
        }, f, indent=2)
    print(f"[write] full report      -> {report_path}")

    # 3) Kotlin snippet for the app's gold-standard path
    print("\nPaste into RecordingActivity.kt companion object (then rebuild):")
    print(f"        val CALIB_FX: Float? = {fx:.4f}f")
    print(f"        val CALIB_FY: Float? = {fy:.4f}f")
    print(f"        val CALIB_CX: Float? = {cx:.4f}f")
    print(f"        val CALIB_CY: Float? = {cy:.4f}f")


if __name__ == "__main__":
    main()
