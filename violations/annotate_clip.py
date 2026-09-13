"""
annotate_clip -- burn a RED BOX on the violating vehicle + a violation caption into the evidence clip.

The lossless ``clip_extract`` cut (``-c copy``) is the untouched evidence; this is the human-facing
ANNOTATED version: for every frame in the window it draws a red rectangle on the offending vehicle
(from a per-frame bbox lookup) and a translucent caption banner saying what the violation is. Because
it draws pixels it must RE-ENCODE, so the returned ``ClipAsset`` is flagged ``recompressed=True``
(validate that hop with ``allow_recompress=True`` -- same resolution, different bytes by design).

The bbox source is INJECTED as ``box_for_frame(frame_id) -> (x1,y1,x2,y2) | None`` so this module
knows nothing about where boxes come from (a tracker, a cached analysis JSON, ...) and unit-tests
with a trivial lambda. ``cv2`` is lazy-imported so importing this module never requires OpenCV.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

from violations.clip_encoder import open_clip_writer
from violations.clip_extract import ClipAsset, ClipWindow

# frame_id -> bounding box (x1, y1, x2, y2) of the violating vehicle, or None if absent that frame.
BoxForFrame = Callable[[int], Optional[Sequence[float]]]

RED = (0, 0, 255)          # BGR
WHITE = (255, 255, 255)


def _put_text_bg(img, text, org, *, color=WHITE, bg=(0, 0, 0), scale=0.55, thick=1, pad=4):
    """Draw ``text`` at ``org`` (bottom-left baseline) on a filled background box for legibility."""
    import cv2
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y), bg, -1)
    cv2.putText(img, text, (x + pad, y - pad), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _banner(img, lines, accent):
    """Translucent dark strip across the top with the caption (line 0 in the accent colour)."""
    import cv2
    h, w = img.shape[:2]
    bh = 14 + 26 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = 26
    for i, ln in enumerate(lines):
        col = accent if i == 0 else WHITE
        sc = 0.72 if i == 0 else 0.55
        tk = 2 if i == 0 else 1
        cv2.putText(img, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, tk, cv2.LINE_AA)
        y += 26


def annotate_clip(src: str, dest: str, window: ClipWindow, box_for_frame: BoxForFrame,
                  caption: str, *, fps: Optional[float] = None, color=RED, tag: str = "",
                  fourcc: str = "mp4v") -> ClipAsset:
    """Render an annotated evidence clip for ``window`` of ``src`` to ``dest``.

    For each frame in ``[window.start_frame, window.end_frame]``: draw the red box from
    ``box_for_frame`` (skipped on frames where it returns None), a ``tag`` label above the box, the
    ``caption`` banner, and a footer with the frame index (the incident frame is marked). Returns a
    ``ClipAsset`` (recompressed=True) ready to fold into the export bundle.
    """
    import cv2
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {src}")
    fps = float(fps or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ONE H.264 encode straight from these frames -- see violations/clip_encoder.py. The frames
    # handed over below are byte-identical to what cv2.VideoWriter used to get; `fourcc` is now
    # only the fallback container for when ffmpeg is missing.
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
            box = box_for_frame(fid)
            if box is not None:
                x1, y1, x2, y2 = (int(v) for v in box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                if tag:
                    _put_text_bg(frame, tag, (x1, max(20, y1)), color=WHITE, bg=color)
            is_incident = abs(fid - window.key_frame) <= 1
            _banner(frame, [caption], color)
            footer = f"frame {fid}" + ("   <<< VIOLATION FRAME" if is_incident else "")
            _put_text_bg(frame, footer, (12, h - 10),
                         color=WHITE, bg=(color if is_incident else (0, 0, 0)), scale=0.55)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return ClipAsset(path=dest, window=window,
                     container=os.path.splitext(dest)[1].lstrip(".").lower() or "mp4",
                     recompressed=True)
