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

# Phase 3 seg-class (canonical name from the 3-class V1 model) -> LaneType.
# Mapped by NAME (via the model's own `names`), so it's robust to class ordering.
_SEG_NAME_TO_LANETYPE: dict[str, LaneType] = {
    "solid_white_lane": "solid_white",
    "yellow_solid_lane": "solid_yellow",
    "dashed_lane": "dashed",
}


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
        type_weights: str = "models/phase3_israeli_head.pth",  # Phase 3 seg head (stub path for now)
        type_conf: float = 0.20,                      # LOW seg gate: let dipped masks through
                                                       # so SolidMaskStabilizer hysteresis can act
        type_samples: int = 20,                       # points sampled per lane for voting
        type_every_n: int = 5,                        # run the seg head every N frames (1 = every frame)
    ) -> None:
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.use_fp16 = use_fp16
        self.min_points = min_points

        # --- Phase 3 lane-type head (point-in-mask voting over YOLOv8-seg) ---
        self.type_weights = type_weights
        self.type_conf = type_conf
        self.type_samples = max(2, type_samples)
        self.type_every_n = max(1, type_every_n)
        self._type_model = None         # lazily loaded on first classify
        self._type_unavailable = False  # True once we confirm the weights are absent
        self._type_masks: list[tuple[LaneType, np.ndarray, float]] | None = None  # (type, poly, conf)
        self._frame_count = 0           # drives the every-N-frames cadence

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
        lanes = self._classify_types(frame, lanes)
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

    def _classify_types(self, frame: np.ndarray, lanes: list[Lane]) -> list[Lane]:
        """
        Assign each geometric lane a semantic `lane_type` via point-in-mask voting
        against the Phase 3 YOLOv8-seg lane-type head.

        Strategy
          * Every-N-frames: the seg model is the expensive stage, so we run it
            only once every `type_every_n` frames and cache its masks; the cheap
            voting runs EVERY frame against the most recent masks. Lane *geometry*
            is always fresh from CLRerNet; lane *type* is temporally stable, so
            voting fresh polylines against slightly-older masks is sound.
          * Voting: for each lane, sample `type_samples` points down its polyline
            (via `Lane.x_at_y`) and tally which seg mask each point falls inside
            (`cv2.pointPolygonTest`). The class with the most votes wins; a lane
            that overlaps no mask keeps its current type.

        No-op until the model exists: if `type_weights` is absent (e.g. still
        training) the wiring stays live but lanes keep lane_type="unknown", so the
        geometric pipeline behaves exactly as in Phase 2.
        """
        self._ensure_type_model()
        if self._type_model is None:
            return lanes  # weights absent: documented no-op

        # Refresh masks on the every-N-frames cadence (or when the cache is empty).
        # The seg head runs regardless of how many CLRerNet lanes there are, so
        # solid_lane_masks() reflects the current frame even when CLRerNet found
        # nothing - the mask-based crossing test depends on those masks.
        if self._type_masks is None or self._frame_count % self.type_every_n == 0:
            self._type_masks = self._run_type_model(frame)
        self._frame_count += 1

        for lane in lanes:
            voted = self._vote_lane_type(lane, self._type_masks)
            if voted is not None:
                lane.lane_type = voted
        return lanes

    def solid_lane_masks(self) -> list[tuple[LaneType, np.ndarray, float]]:
        """
        SOLID lane masks (solid_white / solid_yellow) from the most recent
        `detect()`: a list of (lane_type, polygon, conf) where each polygon is an
        int32 Nx2 array of full-resolution frame coordinates (cv2 contour form) and
        `conf` is the seg head's confidence for that instance.

        Empty if the Phase 3 type head is unavailable or no solid lane was seen.
        Feed this straight into SolidMaskStabilizer.update(): the confidence drives
        the hysteresis (t_high/t_low), and the stabilized binary mask it returns is
        what the crossing test checks the vehicle's bottom-center anchor against.
        """
        if not self._type_masks:
            return []
        return [(lt, poly, conf) for lt, poly, conf in self._type_masks
                if lt in ("solid_white", "solid_yellow")]

    def _ensure_type_model(self) -> None:
        """Lazily load the YOLOv8-seg lane-type head. If the weights file is
        absent (model still training), mark it unavailable and stay a no-op."""
        if self._type_model is not None or self._type_unavailable:
            return
        import os
        if not os.path.isfile(self.type_weights):
            self._type_unavailable = True
            print(f"[DLLaneDetector] lane-type weights not found: {self.type_weights} "
                  f"- lane_type stays 'unknown' until the Phase 3 head is trained.")
            return
        from ultralytics import YOLO
        self._type_model = YOLO(self._as_pt(self.type_weights), task="segment")

    @staticmethod
    def _as_pt(weights_path: str) -> str:
        """Ultralytics' loader only accepts a `.pt` suffix; our Phase 3 weights are
        saved `.pth`, so materialize a `.pt` copy alongside and load that."""
        if weights_path.lower().endswith(".pt"):
            return weights_path
        import os
        import shutil
        pt_path = os.path.splitext(weights_path)[0] + ".pt"
        if not os.path.isfile(pt_path) or os.path.getmtime(pt_path) < os.path.getmtime(weights_path):
            shutil.copy2(weights_path, pt_path)
        return pt_path

    def _run_type_model(self, frame: np.ndarray) -> list[tuple[LaneType, np.ndarray, float]]:
        """Run the seg head on `frame`; return [(LaneType, polygon int32 array, conf)]
        for every instance whose class maps onto our contract. Polygons come back
        in full-resolution frame pixel coords - the same space as `Lane.points`.
        Confidence is surfaced so SolidMaskStabilizer can apply hysteresis."""
        device = 0 if self.device == "cuda" else "cpu"
        res = self._type_model.predict(frame, conf=self.type_conf,
                                       verbose=False, device=device)[0]
        masks: list[tuple[LaneType, np.ndarray, float]] = []
        if res.masks is None:
            return masks
        names = res.names  # {idx: class_name}
        for poly, cls_id, conf in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            if poly is None or len(poly) < 3:
                continue
            lane_type = _SEG_NAME_TO_LANETYPE.get(names[int(cls_id)])
            if lane_type is None:
                continue  # class outside our contract: ignore
            masks.append((lane_type, np.asarray(poly, dtype=np.int32), float(conf)))
        return masks

    def _vote_lane_type(
        self,
        lane: Lane,
        masks: list[tuple[LaneType, np.ndarray]],
    ) -> LaneType | None:
        """Sample points down `lane` and return the seg class most of them fall
        inside, or None if the lane overlaps no mask."""
        import cv2

        y_top, y_bottom = lane.top[1], lane.bottom[1]
        if y_bottom <= y_top:
            return None
        votes: dict[LaneType, int] = {}
        n = self.type_samples
        for i in range(n):
            y = int(round(y_top + (y_bottom - y_top) * i / (n - 1)))
            x = lane.x_at_y(y)
            if x is None:
                continue
            pt = (float(x), float(y))
            for lane_type, poly, _conf in masks:
                if cv2.pointPolygonTest(poly, pt, False) >= 0:
                    votes[lane_type] = votes.get(lane_type, 0) + 1
        if not votes:
            return None
        return max(votes, key=votes.get)

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
