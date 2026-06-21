"""
Dynamic aspect-ratio rendering for violation overlays.

Scales text size, box thickness and banner height to the frame so the UI stays readable and does
not clip on BOTH 9:16 vertical Shorts and 16:9 widescreen. The trick: scale off the SHORTER side
(min(w, h)) -- scaling off height alone makes text gigantic on tall vertical videos.

`scale_params` is pure (unit-tested); `draw_violation` lazy-imports cv2.
"""
from __future__ import annotations

from typing import Any


def scale_params(w: int, h: int) -> dict:
    """UI dimensions derived from the shorter side so vertical and wide frames both look right."""
    s = min(w, h)
    return {
        "thick": max(2, round(s / 240)),
        "font_scale": max(0.5, s / 900.0),
        "banner_h": max(26, round(h * 0.06)),
        "margin": max(6, round(s / 100)),
    }


# distinct-ish colours so neighbouring track ids don't collide visually
_TRACK_PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255), (0, 255, 255),
    (128, 255, 0), (80, 80, 255), (200, 200, 0), (255, 80, 80), (160, 0, 255),
]


def track_color(track_id: int) -> tuple:
    """Stable BGR colour for a track id (so the SAME id keeps its colour across frames)."""
    return _TRACK_PALETTE[int(track_id) % len(_TRACK_PALETTE)]


def draw_track_box(frame: Any, *, bbox, track_id: int) -> Any:
    """Thin id-coloured box + 'ID N' tag for EVERY tracked vehicle (diagnostic overlay). Lighter
    than draw_violation: no banner/border, just enough to trace the raw tracker output per frame."""
    import cv2  # lazy

    h, w = frame.shape[:2]
    p = scale_params(w, h)
    color = track_color(track_id)
    th = max(1, p["thick"] - 1)
    fs = max(0.4, p["font_scale"] * 0.7)
    x1, y1, x2, y2 = (int(round(c)) for c in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, th)

    label = f"ID {track_id}"
    (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th))
    ly = max(tht + 4, y1)                                   # keep the tag on-screen near the top edge
    cv2.rectangle(frame, (x1, ly - tht - 4), (x1 + tw + 4, ly), color, -1)
    cv2.putText(frame, label, (x1 + 2, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 0, 0), max(1, th), cv2.LINE_AA)
    return frame


def draw_violation(frame: Any, *, bbox, track_id: int, plate: str | None,
                   violation_type: str, confidence: float, in_event: bool = True) -> Any:
    """Draw the red vehicle box, a top violation banner, and a bottom id/plate strip -- all scaled
    to the frame. `plate` None -> 'UNKNOWN'. Returns the frame (drawn in place)."""
    import cv2  # lazy: keeps this module importable without OpenCV

    h, w = frame.shape[:2]
    p = scale_params(w, h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    red, white, black = (0, 0, 255), (255, 255, 255), (0, 0, 0)
    plate_txt = plate or "UNKNOWN"

    # vehicle box (+ full-frame border while the offense is active)
    x1, y1, x2, y2 = (int(round(c)) for c in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), red, p["thick"])
    if in_event:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), red, p["thick"] + 1)

    # top banner: violation type (ASCII -- cv2 cannot draw Hebrew)
    banner = f"VIOLATION: {violation_type}"
    cv2.rectangle(frame, (0, 0), (w, p["banner_h"]), black, -1)
    cv2.putText(frame, banner, (p["margin"], int(p["banner_h"] * 0.72)),
                font, p["font_scale"], white, max(1, p["thick"] - 1), cv2.LINE_AA)

    # bottom strip: unified track id + voted plate + confidence
    label = f"ID {track_id}   PLATE {plate_txt}   conf {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, font, p["font_scale"], max(1, p["thick"] - 1))
    by2 = h - p["margin"]
    cv2.rectangle(frame, (0, by2 - th - p["margin"]), (tw + 2 * p["margin"], h), black, -1)
    cv2.putText(frame, label, (p["margin"], by2 - p["margin"] // 2),
                font, p["font_scale"], white, max(1, p["thick"] - 1), cv2.LINE_AA)
    return frame
