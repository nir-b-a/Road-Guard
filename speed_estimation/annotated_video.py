"""
Annotated output video.

After estimation, re-read the source footage and write a SECOND video in which
every tracked vehicle is drawn with its bounding box, track id, resolved class,
and the estimated speed (km/h) and distance (m) for that frame. The ego (camera
vehicle) speed is drawn as a banner centred at the top of every frame, so the
own-speed reference rides along next to the per-vehicle estimates.

This is the qualitative companion to the per-vehicle PNG plots: instead of one
graph per vehicle you watch the numbers ride along with each box in the video.
Speed/distance are only drawn on frames where an estimate exists -- a vehicle too
short-lived for the speed filter still gets its box + id + class, and a frame
whose distance could not be resolved (0.0 placeholder) simply omits the distance.

Frame indexing matches main.processFrame: the source is read sequentially from
frame 0, so frame N here is the same frame the bboxes were recorded against.
Interpolated bboxes (bbox_interpolator) are real tuples, so they are drawn too --
the boxes stay smooth through the short occlusion gaps that were filled.

File naming: {video_name}_annotated.mp4, next to the other outputs.
"""

import colorsys

import cv2

from Objects.World import World
from Constants import class_name

MPS_TO_KMH = 3.6

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2
_PAD = 4

# Ego banner: bigger than the per-vehicle labels so the own-speed reference reads
# at a glance, with bright text on a solid dark plate for contrast over any scene.
_EGO_FONT_SCALE = 0.8
_EGO_FONT_THICKNESS = 2
_EGO_BG_COLOR = (0, 0, 0)        # black plate (BGR)
_EGO_TEXT_COLOR = (0, 255, 255)  # yellow text (BGR)


def _color_for_id(vid: int) -> tuple[int, int, int]:
    """Deterministic, visually-distinct BGR colour for a track id.

    Golden-ratio hue stepping spreads consecutive ids far apart on the colour
    wheel, so neighbouring tracks rarely share a similar colour. High value +
    saturation keep the boxes bright, so black label text stays readable on them.
    """
    h = (vid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def _draw_label(frame, lines: list[str], x: int, y: int,
                color: tuple[int, int, int]) -> None:
    """Draw stacked text lines in a filled box anchored to a bbox top-left."""
    sizes = [cv2.getTextSize(t, _FONT, _FONT_SCALE, _FONT_THICKNESS)[0] for t in lines]
    text_w = max(w for w, _ in sizes)
    line_h = max(h for _, h in sizes) + 4
    box_w = text_w + 2 * _PAD
    box_h = line_h * len(lines) + 2 * _PAD

    # Prefer above the bbox; if it would clip the top, drop it just below the edge.
    top = y - box_h
    if top < 0:
        top = y
    left = x

    # Keep the label fully inside the frame.
    h_img, w_img = frame.shape[:2]
    left = max(0, min(left, w_img - box_w))
    top = max(0, min(top, h_img - box_h))

    cv2.rectangle(frame, (left, top), (left + box_w, top + box_h), color, -1)
    ty = top + _PAD + sizes[0][1]
    for t, (_, h) in zip(lines, sizes):
        cv2.putText(frame, t, (left + _PAD, ty), _FONT, _FONT_SCALE,
                    (0, 0, 0), _FONT_THICKNESS, cv2.LINE_AA)
        ty += line_h


def _draw_frame(frame, world: World, frame_id: int) -> None:
    """Draw every vehicle present in `frame_id` onto `frame` (in place)."""
    for vid, vehicle in world.vehicles.items():
        bbox = vehicle.bounding_box.get(frame_id)
        if bbox is None or bbox == (0, 0, 0, 0):
            continue

        x1, y1, x2, y2 = bbox
        color = _color_for_id(vid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

        lines = [f"ID {vid} {class_name(vehicle.vehicle_type)}"]
        speed = vehicle.speed_per_frame.get(frame_id)
        if speed is not None:
            lines.append(f"{speed * MPS_TO_KMH:.1f} km/h")
        dist = vehicle.dist_per_frame.get(frame_id, 0.0)
        if dist > 0:
            lines.append(f"{dist:.1f} m")

        _draw_label(frame, lines, x1, y1, color)


def _draw_ego_speed(frame, ego_speed_mps: float) -> None:
    """Draw the ego (camera vehicle) speed as a banner centred at the top edge."""
    text = f"EGO {ego_speed_mps * MPS_TO_KMH:.1f} km/h"
    (text_w, text_h), base = cv2.getTextSize(
        text, _FONT, _EGO_FONT_SCALE, _EGO_FONT_THICKNESS)
    w_img = frame.shape[1]
    box_w = text_w + 2 * _PAD
    box_h = text_h + base + 2 * _PAD
    left = max(0, (w_img - box_w) // 2)

    cv2.rectangle(frame, (left, 0), (left + box_w, box_h), _EGO_BG_COLOR, -1)
    cv2.putText(frame, text, (left + _PAD, _PAD + text_h), _FONT, _EGO_FONT_SCALE,
                _EGO_TEXT_COLOR, _EGO_FONT_THICKNESS, cv2.LINE_AA)


def render_annotated_video(world: World, video_path: str, output_path: str,
                           fps: float | None = None,
                           ego_speed: dict[int, float] | None = None) -> None:
    """
    Write {output_path}: the source footage with every tracked vehicle's bbox,
    id, class, and estimated speed/distance overlaid per frame.

    Args:
        world:        the populated World (its .vehicles hold the per-frame bboxes,
                      speeds and distances).
        video_path:   the ORIGINAL source video (re-read from frame 0).
        output_path:  destination .mp4.
        fps:          output frame rate; falls back to the source's CAP_PROP_FPS
                      (then 30) when None/0. Pass the same fps the pipeline used so
                      the annotated clip plays at real time.
        ego_speed:    optional {frame -> ego speed (m/s)} drawn as a banner at the
                      top of each frame. The last known value is held across frames
                      the dict skips, so the banner stays steady. None -> no banner.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[annotated_video] cannot open source {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        print(f"[annotated_video] cannot open writer for {output_path}")
        cap.release()
        return

    frame_id = 0
    last_ego: float | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _draw_frame(frame, world, frame_id)
        if ego_speed is not None:
            # Hold the last known speed across frames the dict skips so the banner
            # doesn't flicker out on a gap.
            last_ego = ego_speed.get(frame_id, last_ego)
            if last_ego is not None:
                _draw_ego_speed(frame, last_ego)
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"[annotated_video] wrote {frame_id} frame(s) -> {output_path}")
