"""
Deep-learning lane detector wrapper (Phase 1).

`DLLaneDetector` adapts an UnLanedet model (CLRerNet in Phase 2) to Road Guard's
`LaneSource` protocol: `detect(frame) -> list[Lane]` in full-resolution image
coordinates, with ego-relative `position` assigned geometrically.

Design
------
The class is split so that everything EXCEPT the UnLanedet-specific bits is pure
and unit-testable today, before the framework is even installed:

  * `FrameTransform`        - pure crop/resize coordinate math (test the round-trip)
  * `_preprocess`           - frame -> model-input tensor
  * `_parse_raw`            - SEAM: UnLanedet's native output -> generic lanes   (model coords)
  * `_postprocess`          - generic lanes -> full-res `Lane` objects            (pure)
  * `_assign_positions`     - geometric ego-relative indexing                     (pure)
  * `_ensure_model`/`_run`  - SEAM: UnLanedet load + forward                      (needs the lib)

Only the two SEAM methods touch UnLanedet. Fill those in during Phase 2 against
the actual repo API; the rest needs no changes.

Coordinate convention (CLR-family default; verify against the chosen config):
  full frame --(top-crop `crop_top` rows)--> --(resize to model_w x model_h)--> model input.
The inverse is applied to every output point before a `Lane` is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lane_types import Lane, LaneType, Point

# A raw lane as produced by `_parse_raw`, still in MODEL-input coordinates:
# (polyline points [(x, y), ...], confidence score, raw type label).
RawLane = tuple[list[tuple[float, float]], float, LaneType]


@dataclass(frozen=True)
class FrameTransform:
    """
    Invertible map between full-resolution frame coords and model-input coords.

    Forward:  crop `crop_top` rows off the top, then resize the remaining
              (src_w x (src_h - crop_top)) region to (model_w x model_h).
    """

    src_w: int
    src_h: int
    model_w: int
    model_h: int
    crop_top: int = 0

    @property
    def _cropped_h(self) -> int:
        return self.src_h - self.crop_top

    @property
    def scale_x(self) -> float:
        return self.model_w / self.src_w

    @property
    def scale_y(self) -> float:
        return self.model_h / self._cropped_h

    def forward_point(self, x: float, y: float) -> tuple[float, float]:
        """Full-res (x, y) -> model-input (x, y). Mainly for tests."""
        return x * self.scale_x, (y - self.crop_top) * self.scale_y

    def invert_point(self, mx: float, my: float) -> tuple[float, float]:
        """Model-input (mx, my) -> full-res (x, y)."""
        return mx / self.scale_x, my / self.scale_y + self.crop_top


class DLLaneDetector:
    """UnLanedet model behind the `LaneSource` protocol. Satisfies LaneSource."""

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        *,
        device: str = "cuda",
        conf_threshold: float = 0.4,
        src_size: tuple[int, int] = (1920, 1080),   # (w, h)
        model_input: tuple[int, int] = (800, 320),  # (w, h) — CLRerNet default-ish
        crop_top: int = 270,                          # sky rows cut before resize
        use_fp16: bool = True,
        min_points: int = 2,
    ) -> None:
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.use_fp16 = use_fp16
        self.min_points = min_points

        src_w, src_h = src_size
        model_w, model_h = model_input
        self.transform = FrameTransform(
            src_w=src_w, src_h=src_h,
            model_w=model_w, model_h=model_h,
            crop_top=crop_top,
        )
        self._model = None  # lazily loaded on first detect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Lane]:
        self._ensure_model()
        tensor = self._preprocess(frame)
        raw = self._run_model(tensor)
        raw_lanes = self._parse_raw(raw)
        lanes = self._postprocess(raw_lanes)
        lanes = self._assign_positions(lanes, frame_width=frame.shape[1])
        return lanes

    # ------------------------------------------------------------------
    # Pure / framework-agnostic (testable now)
    # ------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> "np.ndarray":
        """
        Full-res BGR frame -> normalized model-input tensor on `device`.

        Returns a torch tensor at call time; typed loosely to keep this module
        importable without torch installed (for transform/compat unit tests).
        """
        import cv2  # local import: keep module import cheap and torch-free
        import torch

        t = self.transform
        cropped = frame[t.crop_top:, :, :]                       # (cropped_h, src_w, 3)
        resized = cv2.resize(cropped, (t.model_w, t.model_h))    # cv2 takes (w, h)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))                       # (3, H, W)
        tensor = torch.from_numpy(chw).unsqueeze(0)              # (1, 3, H, W)
        tensor = tensor.to(self.device)
        if self.use_fp16:
            tensor = tensor.half()
        return tensor

    def _postprocess(self, raw_lanes: list[RawLane]) -> list[Lane]:
        """
        Generic: confidence-gate, invert the coordinate transform per point,
        drop degenerate lanes, build full-res `Lane` objects (bottom->top).
        """
        out: list[Lane] = []
        for model_points, score, lane_type in raw_lanes:
            if score < self.conf_threshold:
                continue
            pts: list[Point] = [
                tuple(round(c) for c in self.transform.invert_point(mx, my))  # type: ignore[misc]
                for mx, my in model_points
            ]
            if len(pts) < self.min_points:
                continue
            # Enforce bottom (max y) -> top (min y) ordering.
            pts.sort(key=lambda p: p[1], reverse=True)
            out.append(Lane(points=pts, lane_type=lane_type, score=float(score)))
        return out

    @staticmethod
    def _assign_positions(lanes: list[Lane], frame_width: int) -> list[Lane]:
        """
        Assign ego-relative `position` geometrically from each lane's x at its
        bottom-most point, relative to image center. Nearest lane on each side
        gets +-1, next +-2, etc. (Refinement #2: position is derived, not
        learned — so this works even with pretrained weights and no type head.)
        """
        center = frame_width / 2.0
        lefts = sorted(
            (ln for ln in lanes if ln.bottom[0] < center),
            key=lambda ln: ln.bottom[0], reverse=True,   # closest-to-center first
        )
        rights = sorted(
            (ln for ln in lanes if ln.bottom[0] >= center),
            key=lambda ln: ln.bottom[0],                 # closest-to-center first
        )
        for i, ln in enumerate(lefts, start=1):
            ln.position = -i
        for i, ln in enumerate(rights, start=1):
            ln.position = i
        return lanes

    # ------------------------------------------------------------------
    # UnLanedet SEAMS — fill these against the actual repo API in Phase 2.
    # Nothing else in this file should need to change.
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        # SEAM: load UnLanedet config + checkpoint, move to device, eval, half().
        # Pseudostructure (adapt names to the installed API):
        #
        #   from unlanedet.config import Config
        #   from unlanedet.model import build_model
        #   cfg = Config.fromfile(self.config_path)
        #   model = build_model(cfg)
        #   load_checkpoint(model, self.ckpt_path, map_location=self.device)
        #   model.to(self.device).eval()
        #   if self.use_fp16: model.half()
        #   self._model = model
        raise NotImplementedError(
            "Wire UnLanedet model loading here (Phase 2). See pseudostructure "
            "in _ensure_model(). Until then, DLLaneDetector.detect() will raise."
        )

    def _run_model(self, tensor: "np.ndarray"):
        """SEAM: forward pass. Return the repo's native lane output object."""
        import torch
        with torch.no_grad():
            return self._model(tensor)  # type: ignore[misc]

    def _parse_raw(self, raw) -> list[RawLane]:
        """
        SEAM: convert UnLanedet's native output into `RawLane` tuples in
        MODEL-input coordinates.

        CLR-family output is typically per-lane: a confidence score plus x
        values at a fixed set of y-anchors, with invalid samples flagged
        (negative / out-of-range x). Implement, per lane:
          * read the score,
          * keep only valid (x, y) samples (drop flagged points PER POINT),
          * lane_type = "unknown" until the Phase 3 head exists.
        """
        raise NotImplementedError(
            "Implement UnLanedet output parsing (Phase 2): per-lane score + "
            "valid (x, y) samples in model coords. Return list[RawLane]."
        )
