"""
lane_render.py -- put the lane model's output INTO the videos, and offer a lanes-only run.

Two problems this solves, without touching main.py or any pipeline module:

1. THE LANES WERE INVISIBLE. `phase3_v3_yellowprotect.pt` segments every lane marking on every
   frame, and the solid-line / yellow-line rules are decided from those polygons -- but the
   annotated video only ever drew vehicle boxes, so you could not see WHAT the rule saw. Here the
   polygons are drawn into both the full annotated video and each per-violation evidence clip:
   the solid white line you may not cross, the solid yellow (shoulder) line, dashed lines, and
   traffic islands, each in its own colour with a legend.

2. SPEED WAS ALWAYS COMPUTED. `--lanes-only` skips the world-frame speed stage entirely, so a
   plain video with no GPS/gyro CSVs runs the lane rules at full strength and nothing tries to
   reconstruct a speed it cannot know.

HOW IT ATTACHES (why no existing file changed)
----------------------------------------------
`install(main, ...)` swaps three functions on the already-imported `main` module:

  * `main.evaluate_yellow_line` -> a wrapper that CAPTURES the per-frame lane cache (`seg_frames`,
    which the pipeline builds in its frame loop and then throws away) before delegating to the
    original. Nothing about the yellow-line rule changes.
  * `main.annotated_video.render_annotated_video` -> the same renderer plus a lane layer.
  * `main.annotate_clip` -> the same evidence-clip renderer plus a lane layer.
  * with `lanes_only=True`, `main.run_speed_estimation` -> a no-op, and `--no-overspeed` is added
    to argv so the Overpass speed-limit lookup (the pipeline's only network call) is skipped too.

Everything is undone by `restore()`. The drawing helpers of the two original renderers are reused
rather than copied, so the vehicle boxes, labels and violation flashes stay pixel-identical --
this module only adds a layer underneath them.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from speed_estimation import annotated_video as _av
from violations.annotate_clip import _banner, _put_text_bg
from violations.clip_encoder import open_clip_writer
from violations.clip_extract import ClipAsset

# --------------------------------------------------------------------------- #
# Lane classes -> colour (BGR) + what the legend says about them.
# The class names are the ones weights/phase3_v3_yellowprotect.pt emits.
# --------------------------------------------------------------------------- #
LANE_STYLE = {
    "solid_white_lane":  ((255, 255, 255), "SOLID white - no crossing"),
    "yellow_solid_lane": ((0, 215, 255),   "SOLID yellow - no crossing"),
    "dashed_lane":       ((0, 255, 0),     "dashed - crossing allowed"),
    "traffic_island":    ((0, 140, 255),   "traffic island"),
}
DEFAULT_STYLE = ((200, 200, 200), "lane marking")

FILL_ALPHA = 0.35        # translucency of the filled polygon over the road
OUTLINE_PX = 2
MIN_LABEL_AREA = 1200    # px^2: below this a polygon gets no text (keeps the frame readable)


def _style(cls: str):
    return LANE_STYLE.get(cls, DEFAULT_STYLE)


def _contour(lane) -> np.ndarray | None:
    pts = lane.get("contour")
    if not pts or len(pts) < 3:
        return None
    return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)


def draw_lanes(frame, lanes, *, alpha: float = FILL_ALPHA, labels: bool = True) -> None:
    """Draw one frame's lane polygons onto `frame`, in place.

    Filled translucent first (so the road texture still shows through), then a crisp outline and
    a small `class conf` tag on the polygons big enough to carry one.
    """
    if not lanes:
        return
    contours = []
    for lane in lanes:
        cnt = _contour(lane)
        if cnt is not None:
            contours.append((lane, cnt))
    if not contours:
        return

    overlay = frame.copy()
    for lane, cnt in contours:
        cv2.fillPoly(overlay, [cnt], _style(lane.get("cls"))[0])
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    for lane, cnt in contours:
        color = _style(lane.get("cls"))[0]
        cv2.polylines(frame, [cnt], True, color, OUTLINE_PX, cv2.LINE_AA)
        if not labels or cv2.contourArea(cnt) < MIN_LABEL_AREA:
            continue
        top = cnt.reshape(-1, 2)[cnt.reshape(-1, 2)[:, 1].argmin()]
        tag = str(lane.get("cls", "lane"))
        conf = lane.get("conf")
        if conf is not None:
            tag += f" {float(conf):.2f}"
        _put_text_bg(frame, tag, (int(top[0]), max(18, int(top[1]))),
                     color=(0, 0, 0), bg=color, scale=0.45)


def draw_legend(frame, lanes, *, x_margin: int = 12, y_top: int = 12) -> None:
    """Colour key for the classes present in THIS frame, top-right, over a dark plate."""
    present = []
    for lane in lanes or []:
        cls = lane.get("cls")
        if cls not in [c for c, _ in present]:
            present.append((cls, _style(cls)))
    if not present:
        return

    rows = [f"{desc}" for _, (_, desc) in present]
    scale, thick = 0.5, 1
    sizes = [cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0] for r in rows]
    text_w = max(w for w, _ in sizes)
    row_h = max(h for _, h in sizes) + 10
    box_w = text_w + 46
    box_h = row_h * len(rows) + 12
    h_img, w_img = frame.shape[:2]
    x0 = max(0, w_img - box_w - x_margin)
    y0 = y_top

    plate = frame.copy()
    cv2.rectangle(plate, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(plate, 0.55, frame, 0.45, 0, frame)

    y = y0 + 10
    for (cls, (color, desc)), (_, th) in zip(present, sizes):
        cv2.rectangle(frame, (x0 + 8, y + 2), (x0 + 30, y + th + 2), color, -1)
        cv2.rectangle(frame, (x0 + 8, y + 2), (x0 + 30, y + th + 2), (255, 255, 255), 1)
        cv2.putText(frame, desc, (x0 + 38, y + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
        y += row_h


def index_lane_frames(seg_frames) -> dict:
    """The pipeline's per-frame cache -> {frame_id: [lane, ...]} for O(1) lookup while rendering."""
    out = {}
    for rec in seg_frames or []:
        lanes = rec.get("lanes")
        if lanes:
            out[int(rec.get("frame", 0))] = lanes
    return out


# --------------------------------------------------------------------------- #
# The two renderers, each = the original + a lane layer
# --------------------------------------------------------------------------- #
def render_annotated_video_with_lanes(world, video_path: str, output_path: str,
                                      fps: float | None = None, violation_events=None,
                                      lane_frames: dict | None = None) -> None:
    """Drop-in replacement for annotated_video.render_annotated_video that also draws the lanes.

    Lanes go on FIRST so vehicle boxes, labels and the red violation flash stay on top and
    unchanged; those are drawn by the original module's own helpers."""
    lane_frames = lane_frames or {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[annotated_video] cannot open source {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not fps or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print(f"[annotated_video] cannot open writer for {output_path}")
        cap.release()
        return

    alerts = _av._build_alerts(violation_events)
    frame_id = 0
    drawn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lanes = lane_frames.get(frame_id)
        if lanes:
            draw_lanes(frame, lanes)
            draw_legend(frame, lanes)
            drawn += 1
        _av._draw_frame(frame, world, frame_id)
        if frame_id in alerts:
            _av._draw_alert(frame, world, alerts[frame_id], frame_id)
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    print(f"[annotated_video] wrote {frame_id} frame(s) ({drawn} with lane overlay) -> {output_path}")


def annotate_clip_with_lanes(src: str, dest: str, window, box_for_frame, caption: str, *,
                             fps: float | None = None, color=(0, 0, 255), tag: str = "",
                             fourcc: str = "mp4v", lane_frames: dict | None = None) -> ClipAsset:
    """Drop-in replacement for violations.annotate_clip.annotate_clip that also draws the lanes,
    so a solid-line-crossing clip actually shows the line that was crossed."""
    lane_frames = lane_frames or {}
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {src}")
    fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Same single-pass H.264 encode as violations.annotate_clip -- this function is the one that
    # actually runs whenever the lane renderer is installed, so it has to match.
    writer = open_clip_writer(dest, fps, (w, h), fourcc=fourcc)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open a clip writer for {dest} (fourcc {fourcc})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, window.start_frame)
    try:
        for fid in range(window.start_frame, window.end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            lanes = lane_frames.get(fid)
            if lanes:
                draw_lanes(frame, lanes)
            box = box_for_frame(fid)
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                if tag:
                    _put_text_bg(frame, tag, (x1, max(20, y1)), color=(255, 255, 255), bg=color)
            is_incident = abs(fid - window.key_frame) <= 1
            _banner(frame, [caption], color)
            footer = f"frame {fid}" + ("   <<< VIOLATION FRAME" if is_incident else "")
            _put_text_bg(frame, footer, (12, h - 10), color=(255, 255, 255),
                         bg=(color if is_incident else (0, 0, 0)), scale=0.55)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return ClipAsset(path=dest, window=window,
                     container=os.path.splitext(dest)[1].lstrip(".").lower() or "mp4",
                     recompressed=True)


# --------------------------------------------------------------------------- #
# Installing / removing the patches
# --------------------------------------------------------------------------- #
class LaneRenderer:
    """Holds the captured lane cache and the originals it replaced."""

    def __init__(self, main, *, overlay: bool, lanes_only: bool):
        self._main = main
        self.overlay = overlay
        self.lanes_only = lanes_only
        self.lane_frames: dict = {}
        self._orig = {}

    # -- capture ---------------------------------------------------------- #
    def reset(self) -> None:
        """Forget the previous clip's lanes (call once per video in a batch run)."""
        self.lane_frames = {}

    def _wrap_evaluate_yellow_line(self):
        original = self._main.evaluate_yellow_line

        def wrapper(world, seg_frames, *args, **kwargs):
            self.lane_frames = index_lane_frames(seg_frames)
            if self.lane_frames:
                print(f"[lanes] captured lane polygons for {len(self.lane_frames)} frame(s)")
            return original(world, seg_frames, *args, **kwargs)

        self._orig["evaluate_yellow_line"] = original
        self._main.evaluate_yellow_line = wrapper

    # -- render ----------------------------------------------------------- #
    def _wrap_renderers(self):
        original_video = self._main.annotated_video.render_annotated_video

        def video_wrapper(world, video_path, output_path, fps=None, violation_events=None):
            return render_annotated_video_with_lanes(
                world, video_path, output_path, fps=fps, violation_events=violation_events,
                lane_frames=self.lane_frames)

        self._orig["render_annotated_video"] = original_video
        self._main.annotated_video.render_annotated_video = video_wrapper

        if hasattr(self._main, "annotate_clip"):
            original_clip = self._main.annotate_clip

            def clip_wrapper(src, dest, window, box_for_frame, caption, **kwargs):
                kwargs.pop("lane_frames", None)
                return annotate_clip_with_lanes(src, dest, window, box_for_frame, caption,
                                                lane_frames=self.lane_frames, **kwargs)

            self._orig["annotate_clip"] = original_clip
            self._main.annotate_clip = clip_wrapper

    # -- lanes-only ------------------------------------------------------- #
    def _disable_speed(self):
        original = self._main.run_speed_estimation

        def noop(*_args, **_kwargs):
            print("[lanes-only] world-frame speed estimation skipped "
                  "(lane rules + plates only; no speeding, no speed plots)")

        self._orig["run_speed_estimation"] = original
        self._main.run_speed_estimation = noop
        # main.py gates its speeding stage on this flag, so the Overpass lookup never runs.
        if "--no-overspeed" not in sys.argv:
            sys.argv.append("--no-overspeed")

    def restore(self) -> None:
        """Put every patched function back (tests, or a caller that wants the stock behaviour)."""
        if "evaluate_yellow_line" in self._orig:
            self._main.evaluate_yellow_line = self._orig["evaluate_yellow_line"]
        if "render_annotated_video" in self._orig:
            self._main.annotated_video.render_annotated_video = self._orig["render_annotated_video"]
        if "annotate_clip" in self._orig:
            self._main.annotate_clip = self._orig["annotate_clip"]
        if "run_speed_estimation" in self._orig:
            self._main.run_speed_estimation = self._orig["run_speed_estimation"]
        self._orig.clear()


def install(main, *, overlay: bool = True, lanes_only: bool = False) -> LaneRenderer:
    """Attach the lane overlay and/or the lanes-only behaviour to an imported `main` module."""
    r = LaneRenderer(main, overlay=overlay, lanes_only=lanes_only)
    if overlay:
        r._wrap_evaluate_yellow_line()
        r._wrap_renderers()
        print("[lanes] overlay ON: detected lane markings are drawn into the annotated video "
              "and every violation clip")
    if lanes_only:
        if not overlay:                      # still need the capture hook for nothing else, but
            r._wrap_evaluate_yellow_line()   # keep the log line about how many frames had lanes
        r._disable_speed()
    return r
