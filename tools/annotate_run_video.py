#!/usr/bin/env python3
"""
annotate_run_video.py -- draw a finished run's per-vehicle ID / speed / distance onto
a copy of the source video, from the run's CSV outputs alone (no detection re-run).

INPUTS
    run_dir   the main.py output folder:
                <name>_tracks.csv          frame, track_id, x1..y2   (every detection)
                <name>_vehicles.csv        vehicle_id, class_id, ...  (resolved class)
                <name>_vehicle_speeds.csv  vehicle_id, frame, speed_mps, x1..y2
                                           (speed series; its bboxes include the
                                           interpolated gap frames tracks.csv lacks)
    video     the ORIGINAL video the run processed (frame N here = frame N in the CSVs)
    --session folder holding intrinsics.json (default: the video's folder)

DISTANCE
    The run CSVs do not store distance, so it is recomputed with the pipeline's own
    functions, exactly as main.py step 6 does it: ego_yaw.load_intrinsics ->
    speed_estimator.estimateDistance(world, fx, fy, cx, cy, camera_height) ->
    smooth_distances(world, SMOOTH_WINDOW, POLYORDER). Pass --camera-height if the run
    used main.py --camera-height. Runs made with --undistort are NOT supported: their
    boxes are in rectified coordinates and would not line up with the raw video.

OUTPUT
    <run_dir>/<name>_overlay.mp4 (or --out): every vehicle gets its box and a label
    above it -- "ID <id> <class>", "<speed> km/h" ("-- km/h" when the vehicle has no
    speed estimate on that frame), "<distance> m".

RUN (from the malshinon/ directory)
    python tools/annotate_run_video.py test_videos/chase_run_gpsonly test_videos/chase_vid/vid.mp4
    python tools/annotate_run_video.py RUN VIDEO --start-frame 5400 --end-frame 6300
"""

import argparse
import colorsys
import csv
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # malshinon/
sys.path.insert(0, ROOT)

import Constants                                                   # noqa: E402
from Constants import class_name                                   # noqa: E402
from Objects.World import World                                    # noqa: E402
from speed_estimation import ego_yaw                               # noqa: E402
from speed_estimation.speed_estimator import estimateDistance, smooth_distances  # noqa: E402

MPS_TO_KMH = 3.6
DEFAULT_CLASS_ID = 2          # "car", for a track id missing from vehicles.csv

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_PAD = 6


# ============================================================================
# Rebuilding the World from the run CSVs
# ============================================================================

def _rows(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def _bbox(r: dict):
    """(x1, y1, x2, y2) ints, or None for a blank / zero bbox."""
    try:
        b = tuple(int(float(r[k])) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    return None if b == (0, 0, 0, 0) else b


def build_world(run_dir: str, name: str, frame_count: int) -> World:
    tracks_csv = os.path.join(run_dir, f"{name}_tracks.csv")
    vehicles_csv = os.path.join(run_dir, f"{name}_vehicles.csv")
    speeds_csv = os.path.join(run_dir, f"{name}_vehicle_speeds.csv")
    if not os.path.exists(tracks_csv):
        raise SystemExit(f"[overlay] missing {tracks_csv} (wrong run folder or --name?)")

    world = World(frame_count)

    if os.path.exists(vehicles_csv):
        for r in _rows(vehicles_csv):
            try:
                vid, cls, first = int(r["vehicle_id"]), int(r["class_id"]), int(r["first_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            world.addVehicle(vid, cls, first)
    else:
        print(f"[overlay] no {vehicles_csv}; every vehicle is labelled as {class_name(DEFAULT_CLASS_ID)}")

    def vehicle(vid: int, frame: int):
        v = world.getVehicle(vid)
        if v is None:
            v = world.addVehicle(vid, DEFAULT_CLASS_ID, frame)
        return v

    n_boxes = 0
    for r in _rows(tracks_csv):
        b = _bbox(r)
        try:
            frame, vid = int(r["frame"]), int(r["track_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if b is not None:
            vehicle(vid, frame).bounding_box[frame] = b
            n_boxes += 1

    n_speed = n_filled = 0
    if os.path.exists(speeds_csv):
        for r in _rows(speeds_csv):
            try:
                frame, vid, mps = int(r["frame"]), int(r["vehicle_id"]), float(r["speed_mps"])
            except (KeyError, TypeError, ValueError):
                continue
            v = vehicle(vid, frame)
            v.speed_per_frame[frame] = mps
            n_speed += 1
            b = _bbox(r)
            if b is not None and frame not in v.bounding_box:   # interpolated gap frame
                v.bounding_box[frame] = b
                n_filled += 1
    else:
        print(f"[overlay] no {speeds_csv}; speeds will show as '--'")

    for v in world.vehicles.values():
        if v.bounding_box:
            v.start_frame, v.end_frame = min(v.bounding_box), max(v.bounding_box)

    print(f"[overlay] {len(world.vehicles)} vehicles, {n_boxes} detection boxes "
          f"(+{n_filled} interpolated), {n_speed} speed samples")
    return world


# ============================================================================
# Drawing
# ============================================================================

def color_for_id(vid: int) -> tuple:
    """Same golden-ratio hue stepping as speed_estimation/annotated_video.py (BGR)."""
    h = (vid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _overlaps(a: tuple, b: tuple, gap: int = 2) -> bool:
    """Axis-aligned (left, top, right, bottom) rectangles closer than `gap` px."""
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def place_label(size: tuple, bbox: tuple, placed: list, img_w: int, img_h: int) -> tuple:
    """Label rectangle for `bbox` that avoids every rectangle in `placed`.

    Preferred spot is directly above the box. On a collision it is pushed further up
    past the blocking label; if that runs off the top, it is tried below the box and
    pushed down instead. If neither fits, the preferred spot is used (overlap accepted).
    """
    box_w, box_h = size
    x1, y1, x2, y2 = bbox
    left = max(0, min(x1, img_w - box_w))

    def search(top: int, step_up: bool):
        for _ in range(len(placed) + 1):
            if top < 0 or top + box_h > img_h:
                return None
            rect = (left, top, left + box_w, top + box_h)
            hits = [p for p in placed if _overlaps(rect, p)]
            if not hits:
                return rect
            top = (min(p[1] for p in hits) - box_h - 2) if step_up else (max(p[3] for p in hits) + 2)
        return None

    rect = search(y1 - box_h, step_up=True) or search(y2 + 2, step_up=False)
    if rect is None:
        top = max(0, min(y1 - box_h, img_h - box_h))
        rect = (left, top, left + box_w, top + box_h)
    return rect


def draw_frame(img, by_frame: dict, world: World, frame: int, scale: float) -> int:
    thickness = max(1, round(scale * 2))
    h_img, w_img = img.shape[:2]
    ids = by_frame.get(frame, ())

    items = []
    for vid in ids:
        v = world.vehicles[vid]
        bbox = v.bounding_box[frame]
        speed = v.speed_per_frame.get(frame)
        dist = v.dist_per_frame.get(frame, 0.0)
        lines = [f"ID {vid} {class_name(v.vehicle_type)}",
                 f"{speed * MPS_TO_KMH:.1f} km/h" if speed is not None else "-- km/h",
                 f"{dist:.1f} m" if dist > 0 else "-- m"]
        items.append((dist if dist > 0 else float("inf"), vid, bbox, lines))

    # Nearest vehicles claim their preferred label spot first; farther ones move aside.
    items.sort(key=lambda it: it[0])
    placed = []
    for _, vid, bbox, lines in items:
        color = color_for_id(vid)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, thickness))

        sizes = [cv2.getTextSize(t, _FONT, scale, thickness)[0] for t in lines]
        line_h = max(h for _, h in sizes) + int(10 * scale)
        size = (max(w for w, _ in sizes) + 2 * _PAD, line_h * len(lines) + 2 * _PAD - int(6 * scale))
        left, top, right, bottom = place_label(size, bbox, placed, w_img, h_img)
        placed.append((left, top, right, bottom))

        if bottom < y1 - 2 or top > y2 + 2:           # moved away from the box -> leader line
            anchor_y = y1 if bottom < y1 else y2
            cv2.line(img, ((left + right) // 2, bottom if bottom < y1 else top),
                     ((x1 + x2) // 2, anchor_y), color, max(1, thickness - 1), cv2.LINE_AA)
        cv2.rectangle(img, (left, top), (right, bottom), color, -1)
        ty = top + _PAD + sizes[0][1]
        for t in lines:
            cv2.putText(img, t, (left + _PAD, ty), _FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
            ty += line_h
    return len(items)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Overlay a run's vehicle ID/speed/distance onto a video copy.")
    ap.add_argument("run_dir", help="main.py output folder (<name>_tracks.csv, _vehicles.csv, _vehicle_speeds.csv)")
    ap.add_argument("video", help="the original video the run processed")
    ap.add_argument("--session", default=None,
                    help="folder with intrinsics.json (default: the video's folder)")
    ap.add_argument("--name", default=None, help="CSV prefix (default: video file name without extension)")
    ap.add_argument("--out", default=None, help="output .mp4 (default <run_dir>/<name>_overlay.mp4)")
    ap.add_argument("--camera-height", type=float, default=Constants.DEFAULT_CAMERA_HEIGHT_M,
                    help="camera height in metres, as passed to main.py (default %(default)s)")
    ap.add_argument("--font-scale", type=float, default=0.8, help="label text size (default 0.8)")
    ap.add_argument("--start-frame", type=int, default=0, help="first frame to write")
    ap.add_argument("--end-frame", type=int, default=None, help="last frame to write (inclusive)")
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.video))[0]
    session = args.session or os.path.dirname(os.path.abspath(args.video))
    out = args.out or os.path.join(args.run_dir, f"{name}_overlay.mp4")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[overlay] cannot open video {args.video}")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[overlay] video {width}x{height} @ {fps:.2f} fps, {n_video} frames")

    world = build_world(args.run_dir, name, n_video)
    last_csv_frame = max((max(v.bounding_box) for v in world.vehicles.values() if v.bounding_box),
                         default=-1)
    if last_csv_frame >= n_video > 0:
        print(f"[overlay] WARNING: CSV boxes reach frame {last_csv_frame} but the video has "
              f"{n_video} frames -- is this the right video?")

    fx, fy, cx, cy = ego_yaw.load_intrinsics(os.path.join(session, "intrinsics.json"), width, height)
    print(f"[overlay] intrinsics fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}, "
          f"camera height {args.camera_height} m, method {Constants.DISTANCE_CALCULATION_METHOD}")
    estimateDistance(world, fx, fy, cx, cy, args.camera_height)
    smooth_distances(world, Constants.SMOOTH_WINDOW, Constants.POLYORDER)

    by_frame: dict = {}
    for vid, v in world.vehicles.items():
        for f in v.bounding_box:
            by_frame.setdefault(f, []).append(vid)

    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"[overlay] cannot open writer for {out}")

    end = args.end_frame if args.end_frame is not None else (n_video - 1 if n_video > 0 else 10 ** 9)
    frame = written = labels = 0
    t0 = time.time()
    while frame <= end:
        if frame < args.start_frame:
            if not cap.grab():
                break
            frame += 1
            continue
        ok, img = cap.read()
        if not ok:
            break
        labels += draw_frame(img, by_frame, world, frame, args.font_scale)
        writer.write(img)
        written += 1
        frame += 1
        if written % 1000 == 0:
            print(f"[overlay] frame {frame - 1} ({written / (time.time() - t0):.0f} fps)")

    cap.release()
    writer.release()
    print(f"[overlay] wrote {written} frames (frames {args.start_frame}-{frame - 1}), "
          f"{labels} vehicle labels -> {out}")


if __name__ == "__main__":
    main()
