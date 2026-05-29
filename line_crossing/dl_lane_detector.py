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

Phase 2 reality (CLRerNet via UnLanedet)
----------------------------------------
The installed UnLanedet API differs from the original seam sketch, so the
framework-touching methods (`_ensure_model`, `_preprocess`, `_run_model`,
`_parse_raw`) all delegate to the library:

  * the model is called with a `dict` (`{"img": tensor}`), not a bare tensor;
  * preprocessing (BGR mean-subtraction, resize) is done by the config's own
    `dataloader.test.dataset.processes`, so it matches training exactly;
  * decoded lanes come from `model.get_lanes(out)[0]` and `Lane.to_array(cfg)`
    already returns points in the model's ORIGINAL image space (CULane:
    `ori_img_w x ori_img_h`, e.g. 1640x590), NOT the network input space.

Coordinate convention
  Because `to_array` emits CULane-original coords, `FrameTransform` is repurposed
  (in `_ensure_model`) as a pure resize map between CULane-original space and the
  full-resolution source frame: `model_w/h = ori_img_w/h`, `crop_top = 0`. The
  existing `_postprocess` inversion then maps every point straight back onto the
  source frame, so `_postprocess`/`_assign_positions` need no changes.
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
        src_size: tuple[int, int] = (1920, 1080),   # (w, h) of the source frames
        model_input: tuple[int, int] = (800, 320),  # advisory; real geometry comes from cfg
        crop_top: int = 270,                          # advisory; real geometry comes from cfg
        use_fp16: bool = False,                       # GTX 1060 (Pascal) prefers fp32; no fp16 speedup
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
        self._param_config = None  # cfg.param_config, set in _ensure_model
        self._processes = None     # UnLanedet Preprocess pipeline, set in _ensure_model

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
    def _preprocess(self, frame: np.ndarray) -> dict:
        """
        Full-res BGR frame -> UnLanedet input `dict` ({"img": (1,3,H,W) tensor}).

        Delegates normalization/resize to the config's own test `processes`
        (BGR mean-subtraction, resize to network input) so the input matches how
        CLRerNet was trained. The frame is first resized to CULane-original
        geometry (`ori_img_w x ori_img_h`) so the config's `cut_height` and the
        `to_array` output dimensions stay consistent regardless of source size.
        """
        import cv2  # local import: keep module import cheap

        pc = self._param_config
        # Map our source frame onto CULane-original geometry, then crop sky rows
        # exactly as UnLanedet's detect.py does before its processes run.
        resized = cv2.resize(frame, (int(pc.ori_img_w), int(pc.ori_img_h)))
        img = resized[int(pc.cut_height):, :, :].astype(np.float32)
        data = {"img": img, "lanes": []}
        data = self._processes(data)
        data["img"] = data["img"].unsqueeze(0)
        return data

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
    # UnLanedet-specific methods (Phase 2, implemented against the real API).
    # The pure helpers above (_postprocess, _assign_positions) are unchanged.
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import os

        from ._np_compat import install as _install_np_compat
        _install_np_compat()  # UnLanedet uses removed np.bool/np.int/... aliases

        from unlanedet.checkpoint import Checkpointer
        from unlanedet.config import LazyConfig, instantiate
        from unlanedet.data.transform import Preprocess

        cfg_path = os.path.abspath(self.config_path)
        # The CLRerNet config calls get_config("config/common/train.py"), a
        # CWD-relative path, so LazyConfig.load must run from the UnLanedet root.
        root = self._find_unlanedet_root(cfg_path)
        cwd = os.getcwd()
        try:
            if root:
                os.chdir(root)
            cfg = LazyConfig.load(cfg_path)
            cfg = LazyConfig.apply_overrides(cfg, [])
            model = instantiate(cfg.model)
            processes = Preprocess(instantiate(cfg.dataloader.test.dataset.processes))
        finally:
            os.chdir(cwd)

        pc = cfg.param_config
        # Gate inside the library (proper softmax + CUDA NMS) at our threshold,
        # so parsed lanes carry score=1.0 and pass _postprocess unchanged.
        pc.test_parameters.conf_threshold = self.conf_threshold

        model.to(self.device)
        model.eval()
        Checkpointer(model).load(self.ckpt_path)

        self._model = model
        self._param_config = pc
        self._processes = processes
        # Repurpose FrameTransform as a pure CULane-original <-> source-frame
        # resize map (to_array already returns CULane-original coords).
        self.transform = FrameTransform(
            src_w=self.transform.src_w,
            src_h=self.transform.src_h,
            model_w=int(pc.ori_img_w),
            model_h=int(pc.ori_img_h),
            crop_top=0,
        )

    @staticmethod
    def _find_unlanedet_root(cfg_path: str) -> str | None:
        """Walk up from a config file to the dir holding config/common/train.py."""
        import os
        d = os.path.dirname(cfg_path)
        for _ in range(6):
            if os.path.exists(os.path.join(d, "config", "common", "train.py")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return None

    def _run_model(self, data: dict):
        """SEAM: forward pass + native decode. Returns list[unlanedet Lane]."""
        import torch
        with torch.no_grad():
            out = self._model(data)
        return self._model.get_lanes(out)[0]  # batch size 1

    def _parse_raw(self, raw) -> list[RawLane]:
        """
        SEAM: convert UnLanedet's decoded lanes into `RawLane` tuples.

        `get_lanes` already confidence-gated (at our threshold) and NMS'd, and
        `Lane.to_array(cfg)` returns valid (x, y) samples in CULane-ORIGINAL
        image coordinates (the space FrameTransform now inverts from). Score is
        1.0 because gating happened in the library; type is "unknown" until the
        Phase 3 type head exists.
        """
        raw_lanes: list[RawLane] = []
        for lane in raw:
            arr = lane.to_array(self._param_config)
            if arr is None or len(arr) < self.min_points:
                continue
            pts = [(float(x), float(y)) for x, y in arr]
            raw_lanes.append((pts, 1.0, "unknown"))
        return raw_lanes
