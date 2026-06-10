"""
Temporal stabilizer for solid-lane segmentation masks (raster output).

Fixes per-frame classification FLICKER - a 1-2 frame mask dropout or confidence
dip that breaks the crossing test - with three layers, all host-side (no GPU/VRAM):

  * fast-attack / slow-decay : a newly-confident solid pixel turns ON instantly;
                               a vanished pixel COASTS for `max_age` (K) frames
                               before turning OFF. Asymmetric, so latency is paid
                               only on turn-OFF (the safe direction for a violation).
  * confidence hysteresis    : a pixel ADMITS to ON only at conf >= t_high, but
                               STAYS ON while conf >= t_low (a Schmitt band), so a
                               0.85 -> 0.49 dip never flickers off.
  * morphology               : CLOSE (fill pinholes) + light DILATE (fatten thin
                               lines) on the OUTPUT each frame.

Output is a single binary uint8 HxW mask (0 / 255) so the downstream crossing test
is an O(1) `mask[y, x]` lookup (CrossingMonitor.update_from_mask_image).

Per-PIXEL (not per-polygon) on purpose: with a raster output there's no need to
track lane identity across frames, which removes the fragile IoU / contour-matching
the polygon approach would have required.
"""
from __future__ import annotations

import cv2
import numpy as np


class SolidMaskStabilizer:
    """Stateful per-stream temporal filter. Create one per video; feed it the
    SOLID masks of each frame via `update()`, get back a stabilized binary mask."""

    def __init__(
        self,
        max_age: int = 4,            # K: frames a vanished pixel is kept before turning off
        t_high: float = 0.40,        # admit a NEW solid pixel only at conf >= t_high
        t_low: float = 0.25,         # keep an existing ON pixel until conf < t_low
        close_kernel: int = 5,       # morphological CLOSE size (fill pinholes); 0 = off
        dilate_kernel: int = 5,      # morphological DILATE size (fatten lines); 0 = off
        dilate_iter: int = 1,
        frame_size: tuple[int, int] | None = None,  # (w, h); else inferred on first update
    ) -> None:
        if not (0.0 <= t_low <= t_high <= 1.0):
            raise ValueError(f"need 0 <= t_low ({t_low}) <= t_high ({t_high}) <= 1")
        self.max_age = int(max_age)
        self.t_high = float(t_high)
        self.t_low = float(t_low)
        self.dilate_iter = int(dilate_iter)
        self._close = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
                       if close_kernel else None)
        self._dilate = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
                        if dilate_kernel else None)

        self._on: np.ndarray | None = None    # uint8 HxW, 1 = stabilized solid pixel
        self._age: np.ndarray | None = None   # int16 HxW, frames since last confident keep
        if frame_size is not None:
            w, h = frame_size
            self._ensure(h, w)

    def reset(self) -> None:
        """Clear all temporal state (call between independent video streams)."""
        self._on = None
        self._age = None

    def _ensure(self, h: int, w: int) -> None:
        if self._on is None or self._on.shape != (h, w):
            self._on = np.zeros((h, w), dtype=np.uint8)
            self._age = np.zeros((h, w), dtype=np.int16)

    def update(
        self,
        current_masks,                          # list[(lane_type, polygon int32 Nx2, conf float)]
        frame_shape: tuple[int, int] | None = None,   # (h, w); required if frame_size unset
    ) -> np.ndarray:
        """
        Ingest THIS frame's SOLID masks (e.g. DLLaneDetector.solid_lane_masks()) and
        return the stabilized binary mask (uint8 HxW, 0 or 255), morphed. The
        morphed result is fresh each call and never fed back into the state.
        """
        if frame_shape is not None:
            self._ensure(int(frame_shape[0]), int(frame_shape[1]))
        if self._on is None:
            raise RuntimeError(
                "SolidMaskStabilizer needs a size: pass frame_size to __init__ or "
                "frame_shape to update()."
            )
        h, w = self._on.shape

        # Rasterize this frame's masks into two confidence bands.
        admit = np.zeros((h, w), dtype=np.uint8)   # conf >= t_high  (can turn a pixel ON)
        keep = np.zeros((h, w), dtype=np.uint8)     # conf >= t_low   (can hold an ON pixel)
        for _lane_type, poly, conf in current_masks:
            if poly is None or len(poly) < 3 or conf < self.t_low:
                continue
            pts = poly.astype(np.int32)
            cv2.fillPoly(keep, [pts], 1)
            if conf >= self.t_high:
                cv2.fillPoly(admit, [pts], 1)

        on = self._on.astype(bool)
        admit_b = admit.astype(bool)
        keep_b = keep.astype(bool)

        # Asymmetric update:
        #   refreshed = confident this frame (admitted, or already-on & still kept) -> age 0
        #   stale     = was on but not refreshed -> age++, expire past max_age (slow decay)
        #   new ON    = (was on OR admitted this frame) minus expired
        refreshed = admit_b | (on & keep_b)
        stale = on & ~refreshed
        self._age[refreshed] = 0
        self._age[stale] += 1
        expired = stale & (self._age > self.max_age)
        self._on = ((on | admit_b) & ~expired).astype(np.uint8)

        # OUTPUT: morphology on a copy, never compounded back into the state.
        out = (self._on * 255).astype(np.uint8)
        if self._close is not None:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, self._close)
        if self._dilate is not None and self.dilate_iter > 0:
            out = cv2.dilate(out, self._dilate, iterations=self.dilate_iter)
        return out
