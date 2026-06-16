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
