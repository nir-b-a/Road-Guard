"""
Evidence cards for authorities.

After the annotated clip, we append a held "evidence page" per violating car: the 2-3 SHARPEST
plate crops, zoomed in (tight to the plate via the OCR detector box, upscaled + sharpened), with
the recognized text, the plate score and the violation type. The point is human verifiability --
an officer can read the plate off the zoom, including when OCR returned UNKNOWN (a side-on or
distant car still gets a "manual review" card so a human can try).

Layout math (`card_layout`) is pure and unit-tested; the cv2 drawing/zoom helpers lazy-import cv2.
"""
from __future__ import annotations

from typing import Any

HOLD_SEC = 2.5          # how long each evidence card is held on screen
MAX_ZOOM = 8.0          # never upscale a tiny crop more than this (keeps it from turning to mush)


def card_layout(canvas_w: int, canvas_h: int, n_imgs: int) -> dict:
    """Pure geometry for one evidence page: a title band, a row of n_imgs zoom slots, a caption
    band. Returns pixel boxes that always fit inside the canvas. n_imgs is clamped to >=1."""
    n = max(1, n_imgs)
    title_h = max(28, round(canvas_h * 0.12))
    caption_h = max(28, round(canvas_h * 0.16))
    margin = max(6, round(min(canvas_w, canvas_h) / 100))

    row_top = title_h + margin
    row_bottom = canvas_h - caption_h - margin
    row_h = max(1, row_bottom - row_top)
    slot_w = max(1, (canvas_w - margin * (n + 1)) // n)

    slots = []
    for i in range(n):
        x = margin + i * (slot_w + margin)
        slots.append({"x": x, "y": row_top, "w": slot_w, "h": row_h})

    return {
        "title": {"x": margin, "y": 0, "w": canvas_w, "h": title_h},
        "caption": {"x": margin, "y": canvas_h - caption_h, "w": canvas_w, "h": caption_h},
        "slots": slots,
        "margin": margin,
        "font_scale": max(0.5, min(canvas_w, canvas_h) / 900.0),
    }


def zoom_crop(vehicle_crop: Any, reader, *, target_h: int = 160) -> "tuple[Any, bool]":
    """Return (zoomed_image, is_plate). Tries reader.crop_plate to crop tight to the plate; falls
    back to the whole vehicle crop. Upscales toward target_h (capped at MAX_ZOOM) and sharpens."""
    import cv2
    import numpy as np

    plate = None
    if reader is not None and hasattr(reader, "crop_plate"):
        try:
            plate = reader.crop_plate(vehicle_crop)
        except Exception:
            plate = None
    src = plate if (plate is not None and getattr(plate, "size", 0)) else vehicle_crop
    is_plate = plate is not None and getattr(plate, "size", 0) > 0

    h, w = src.shape[:2]
    if h == 0 or w == 0:
        return src, is_plate
    scale = min(MAX_ZOOM, max(1.0, target_h / float(h)))
    big = cv2.resize(src, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(big, -1, kernel), is_plate


def _fit_into(img: Any, box: dict) -> "tuple[Any, int, int]":
    """Scale img to fit inside box (keep aspect), return (resized, x_offset, y_offset) for centring."""
    import cv2

    h, w = img.shape[:2]
    s = min(box["w"] / float(w), box["h"] / float(h))
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return resized, box["x"] + (box["w"] - nw) // 2, box["y"] + (box["h"] - nh) // 2


def compose_card(canvas_w: int, canvas_h: int, *, track_id: int, plate: str | None,
                 score: float, violation_type: str, zoom_imgs: list, has_plate: bool,
                 source: str | None = None) -> Any:
    """Build one full-frame (canvas_h x canvas_w) evidence page. Black background, title with the
    violation type + car id, a row of zoomed plate crops, and a big caption with the recognized
    plate (or a manual-review note when UNKNOWN). A Re-ID-inherited plate (source=
    'reid_inherited:<tid>') is captioned in yellow as 'inherited from ID <tid> - verify' so the
    human editor confirms the two crops are the same vehicle."""
    import cv2
    import numpy as np

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    lay = card_layout(canvas_w, canvas_h, len(zoom_imgs))
    font = cv2.FONT_HERSHEY_SIMPLEX
    white, yellow, red, green = (255, 255, 255), (0, 255, 255), (0, 0, 255), (0, 220, 0)
    fs = lay["font_scale"]
    th = max(1, round(fs * 2))

    # title band
    t = lay["title"]
    cv2.rectangle(canvas, (0, 0), (canvas_w, t["h"]), (40, 40, 40), -1)
    cv2.putText(canvas, f"EVIDENCE  ID {track_id}  |  {violation_type}",
                (t["x"], int(t["h"] * 0.66)), font, fs, white, th, cv2.LINE_AA)

    # zoom slots (frame each crop, red border if it's a real plate localisation)
    for img, slot in zip(zoom_imgs, lay["slots"]):
        if img is None or getattr(img, "size", 0) == 0:
            continue
        resized, ox, oy = _fit_into(img, slot)
        rh, rw = resized.shape[:2]
        canvas[oy:oy + rh, ox:ox + rw] = resized
        cv2.rectangle(canvas, (ox - 2, oy - 2), (ox + rw + 1, oy + rh + 1),
                      green if has_plate else (120, 120, 120), max(2, th))

    # caption band: recognized plate + score, or manual-review note
    c = lay["caption"]
    cv2.rectangle(canvas, (0, c["y"]), (canvas_w, canvas_h), (40, 40, 40), -1)
    if plate and source and source.startswith("reid_inherited"):
        src_id = source.split(":", 1)[1] if ":" in source else "?"
        cap = f"PLATE  {plate}   (inherited from ID {src_id} - verify)"
        colour = yellow                                  # flag for the reviewer: confirm same vehicle
    elif plate:
        cap = f"PLATE  {plate}   (score {score:.2f})"
        colour = green if score >= 0.6 else yellow
    else:
        cap = "PLATE UNKNOWN - manual review (zoom above)"
        colour = red
    cap_fs = fs * 1.6
    cv2.putText(canvas, cap, (c["x"], c["y"] + int(c["h"] * 0.62)),
                font, cap_fs, colour, max(2, th + 1), cv2.LINE_AA)
    return canvas
