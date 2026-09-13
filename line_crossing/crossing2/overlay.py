"""Debug overlay for crossing2.

With no labelled ground truth, the rendered video IS the evaluation instrument, so
it has to show WHY something fired, not just that it did. Per frame this draws:

  * every tracked marking as a polyline, coloured by its ONE track-level type and
    captioned with its track id and tags (double / gore);
  * for each vehicle: the box, the ground anchor, the corrected footprint interval
    drawn on the contact row, and a tick where the nearest solid marking crosses
    that same row - the two things the offset is the distance between;
  * the numeric offset u in lane widths, so a disagreement can be read off the
    frame instead of guessed at;
  * state colour: green = clear, amber = strafing, red = confirmed crossing.

`legacy_flag` lets the A/B runner stamp the old detector's verdict on the same
frame, so both detectors can be judged from one video.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import CrossingResult

TYPE_COLOR = {
    "solid_yellow": (0, 220, 255),
    "solid_white": (245, 245, 245),
    "solid": (200, 200, 255),
    "dashed": (150, 150, 150),
    "unknown": (110, 110, 110),
}
# Raw segmentation classes, drawn UNDER the tracked lines so the two layers can be
# compared: thin outline = what the lane model emitted this frame, thick polyline =
# what Stage 0 made of it. A thick line with no outline under it is a coasted /
# interpolated track - which is exactly the case that used to lose crossings.
SEG_COLOR = {
    "solid_white_lane": (255, 255, 255),
    "yellow_solid_lane": (0, 220, 255),
    "dashed_lane": (140, 140, 140),
    "traffic_island": (220, 120, 255),
}
CLEAR = (0, 200, 0)
STRAFE = (0, 190, 255)
CROSSING = (0, 0, 255)
ANCHOR = (0, 165, 255)
LINE_TICK = (255, 0, 255)


class CrossingOverlay:
    """Pre-indexes a `CrossingResult` so drawing a frame is a dict lookup."""

    def __init__(self, result: CrossingResult, cfg=None) -> None:
        self.r = result
        self.step = float(result.row_grid[1] - result.row_grid[0]) if len(result.row_grid) > 1 else 10.0
        self.index = {int(f): i for i, f in enumerate(result.frame_ids)}

        # vehicle anchors, addressable by (vehicle, frame)
        self.anchor: dict[tuple[int, int], tuple] = {}
        for va in result.anchors.values():
            for k in range(len(va.idx)):
                if va.valid[k]:
                    self.anchor[(va.vehicle_id, int(va.frame_ids[k]))] = (
                        va.x_c[k], va.y_c[k], va.x_left[k], va.x_right[k])

        # the closest solid marking per (vehicle, frame), and its offset
        self.offset: dict[tuple[int, int], tuple] = {}
        for s in result.offsets:
            for k in range(len(s.idx)):
                u = s.u_c[k]
                if not np.isfinite(u):
                    continue
                key = (s.vehicle_id, int(s.frame_ids[k]))
                prev = self.offset.get(key)
                if prev is None or abs(u) < abs(prev[1]):
                    self.offset[key] = (s.track_id, float(u), float(s.u_l[k]),
                                        float(s.u_r[k]), bool(s.extrapolated[k]))

        # event spans per (vehicle, frame) -> the strongest state active there
        self.state: dict[tuple[int, int], str] = {}
        for e in result.events:
            for f in range(e.start_frame, e.end_frame + 1):
                key = (e.vehicle_id, f)
                if e.kind == "crossing" or self.state.get(key) != "crossing":
                    self.state[key] = e.kind

        self.tracks = {t.track_id: t for t in result.tracks}
        self.probes = result.probes or {}

    def draw_probe(self, canvas: np.ndarray, frame_id: int, vid: int) -> None:
        """v3 only: the buffered box that was tested, with the two probed half-edges
        picked out - the left half of the bottom edge and the bottom half of the left
        edge for a right-side line, mirrored for a left-side one. Yellow while they
        are merely being tested, red on the frame they actually touch the line."""
        p = self.probes.get((vid, int(frame_id)))
        if not p:
            return
        (bx1, by1, bx2, by2), side, hit = p
        bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
        xm, ym = (bx1 + bx2) // 2, (by1 + by2) // 2
        col = CROSSING if hit else (0, 210, 210)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 190, 190), 1, cv2.LINE_AA)
        if side == "right":
            cv2.line(canvas, (bx1, by2), (xm, by2), col, 4, cv2.LINE_AA)   # bottom, left half
            cv2.line(canvas, (bx1, ym), (bx1, by2), col, 4, cv2.LINE_AA)   # left edge, bottom half
        else:
            cv2.line(canvas, (xm, by2), (bx2, by2), col, 4, cv2.LINE_AA)
            cv2.line(canvas, (bx2, ym), (bx2, by2), col, 4, cv2.LINE_AA)

    # -- drawing ------------------------------------------------------------
    @staticmethod
    def draw_raw_lanes(canvas: np.ndarray, records, fill: bool = True) -> None:
        """The RAW lane-model output for this frame: `[{"cls": name, "contour": Nx2}]`,
        exactly what the segmentation head produced before any tracking. Drawn as a
        translucent fill plus a thin outline, so the lane model can be debugged
        independently of the crossing logic."""
        if not records:
            return
        polys, colors = [], []
        for r in records:
            c = np.asarray(r.get("contour", ()), dtype=np.int32).reshape(-1, 2)
            if len(c) >= 3:
                polys.append(c)
                colors.append(SEG_COLOR.get(r.get("cls"), (120, 120, 120)))
        if not polys:
            return
        if fill:
            layer = canvas.copy()
            for c, col in zip(polys, colors):
                cv2.fillPoly(layer, [c], col)
            cv2.addWeighted(layer, 0.30, canvas, 0.70, 0, dst=canvas)
        for c, col in zip(polys, colors):
            cv2.polylines(canvas, [c], True, col, 1, cv2.LINE_AA)

    def draw_event_log(self, canvas: np.ndarray, frame_id: int, max_rows: int = 10) -> None:
        """A scrollable-by-time log in the corner: every confirmed event with the
        vehicle id and the frame range, the ones active NOW highlighted. Lets you
        scrub the output video and see what fired where without a console."""
        rows = [e for e in self.r.events if e.start_frame <= frame_id + 1]
        if not rows:
            return
        rows = rows[-max_rows:]
        x0, y0 = canvas.shape[1] - 470, 150
        cv2.rectangle(canvas, (x0 - 12, y0 - 30), (canvas.shape[1] - 10,
                      y0 + 26 * len(rows)), (0, 0, 0), -1)
        cv2.putText(canvas, "EVENTS", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (220, 220, 220), 1, cv2.LINE_AA)
        for k, e in enumerate(rows):
            active = e.start_frame <= frame_id <= e.end_frame
            col = (CROSSING if e.kind == "crossing" else STRAFE) if active else (170, 170, 170)
            txt = (f"v{e.vehicle_id} {e.kind[:5]} f{e.start_frame}-{e.end_frame} "
                   f"{e.lane_type.replace('solid_', '')} c={e.confidence:.2f}")
            cv2.putText(canvas, txt, (x0, y0 + 20 + 26 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, col, 2 if active else 1, cv2.LINE_AA)

    def draw_lanes(self, canvas: np.ndarray, frame_id: int) -> None:
        i = self.index.get(int(frame_id))
        if i is None:
            return
        for t in self.r.tracks:
            if t.suppressed:
                continue
            xs = t.X[i]
            ok = np.isfinite(xs)
            if ok.sum() < 2:
                continue
            pts = np.stack([xs[ok], self.r.row_grid[ok]], axis=1).astype(np.int32)
            color = TYPE_COLOR.get(t.lane_type, TYPE_COLOR["unknown"])
            cv2.polylines(canvas, [pts], False, color, 3 if t.is_solid else 1, cv2.LINE_AA)
            tag = f"L{t.track_id} {t.lane_type}"
            if t.is_double:
                tag += " DBL"
            if t.is_gore:
                tag += " GORE"
            cv2.putText(canvas, tag, (int(pts[-1][0]) + 6, int(pts[-1][1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def draw_vehicle(self, canvas: np.ndarray, frame_id: int, vid: int, bbox) -> str:
        """Draw one vehicle and return its state ("clear"/"strafe"/"crossing")."""
        f = int(frame_id)
        state = self.state.get((vid, f), "clear")
        color = {"crossing": CROSSING, "strafe": STRAFE}.get(state, CLEAR)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 5 if state == "crossing" else 2)

        label = f"ID {vid}"
        if state == "crossing":
            label = f"CROSSING {vid}"
        elif state == "strafe":
            label = f"strafe {vid}"

        a = self.anchor.get((vid, f))
        off = self.offset.get((vid, f))
        if a is not None:
            x_c, y_c, x_l, x_r = a
            yi = int(round(y_c))
            cv2.line(canvas, (int(x_l), yi), (int(x_r), yi), color, 2, cv2.LINE_AA)
            for xe in (x_l, x_r):                        # footprint end ticks
                cv2.line(canvas, (int(xe), yi - 7), (int(xe), yi + 7), color, 2, cv2.LINE_AA)
            cv2.circle(canvas, (int(x_c), yi), 5, ANCHOR, -1)
            if off is not None:
                # where the nearest solid marking crosses the SAME row: the offset
                # being measured is the gap between this tick and the anchor dot.
                tid, u, u_l, u_r, ex = off
                t = self.tracks.get(tid)
                i = self.index.get(f)
                if t is not None and i is not None:
                    r = int(np.clip(round(y_c / self.step), 0, t.X.shape[1] - 1))
                    xl = t.X[i, r]
                    if np.isfinite(xl):
                        cv2.line(canvas, (int(xl), yi - 14), (int(xl), yi + 14),
                                 LINE_TICK, 2, cv2.LINE_AA)
                label += f"  u={u:+.2f}" + ("*" if ex else "")
        cv2.putText(canvas, label, (x1, max(y1 - 8, 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,
                    3 if state == "crossing" else 2, cv2.LINE_AA)
        return state

    def draw_banner(self, canvas: np.ndarray, frame_id: int, states: list[str],
                    extra: str = "") -> None:
        # Kept clear of the top-left corner: dashcam clips usually burn a date/time
        # stamp there, and covering it makes the output harder to cross-reference.
        txt = f"frame {frame_id}  veh={len(states)}  {extra}"
        cv2.rectangle(canvas, (16, 66), (26 + 15 * len(txt), 104), (0, 0, 0), -1)
        cv2.putText(canvas, txt, (24, 94), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if "crossing" in states:
            cv2.putText(canvas, "!!! SOLID LINE CROSSING !!!", (40, 168),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, CROSSING, 5, cv2.LINE_AA)
        elif "strafe" in states:
            cv2.putText(canvas, "strafing solid line", (40, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, STRAFE, 3, cv2.LINE_AA)

    def draw_legacy_badge(self, canvas: np.ndarray, bbox, flagged: bool) -> None:
        """The legacy detector's verdict for the same vehicle, so one video shows
        both. A box with a red OLD badge and no red border is a legacy-only hit."""
        if not flagged:
            return
        x1, _y1, _x2, y2 = bbox
        cv2.putText(canvas, "OLD", (x1, min(y2 + 26, canvas.shape[0] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2, cv2.LINE_AA)
