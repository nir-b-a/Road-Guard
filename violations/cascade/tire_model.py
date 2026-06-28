"""
Thin wrapper around the custom YOLOv11 tire detector for Stage-2 smart-crop inference.

Kept separate from the geometry so the math stays GPU-free and testable. This module is the ONLY
place that touches ultralytics; everything downstream consumes plain tire boxes in FRAME
coordinates.

Usage:
    tm = TireModel("models/tire_yolo11n.pt")
    tires = tm.detect_in_crop(frame, vehicle_bbox)   # -> list[(x1,y1,x2,y2,conf)] in frame coords
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from .geometry import smart_crop_box

BBoxConf = Tuple[float, float, float, float, float]


class TireModel:
    """Lazy-loaded YOLOv11 tire detector. Runs on a smart crop, maps boxes back to frame coords."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 320,
                 expand_bottom_frac: float = 0.15, device: Optional[str] = None):
        self.weights = weights
        self.conf = conf
        self.imgsz = imgsz
        self.expand_bottom_frac = expand_bottom_frac
        self.device = device
        self._model = None

    @property
    def available(self) -> bool:
        return os.path.exists(self.weights)

    def _ensure(self):
        if self._model is None:
            if not self.available:
                raise FileNotFoundError(
                    f"tire weights not found: {self.weights} -- train with train_tire.py first")
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def detect_in_crop(self, frame, vehicle_bbox: Sequence[float]) -> List[BBoxConf]:
        """
        Smart-crop the vehicle's lower body, run the tire model, and return tire boxes in FRAME
        coordinates. Returns [] if nothing detected. `frame` is an HxWx3 numpy/cv2 image.
        """
        h, w = frame.shape[:2]
        cx1, cy1, cx2, cy2 = smart_crop_box(
            tuple(vehicle_bbox), self.expand_bottom_frac, frame_w=w, frame_h=h)
        cx1i, cy1i, cx2i, cy2i = int(cx1), int(cy1), int(cx2), int(cy2)
        if cx2i - cx1i < 4 or cy2i - cy1i < 4:
            return []
        crop = frame[cy1i:cy2i, cx1i:cx2i]
        if crop.size == 0:
            return []
        model = self._ensure()
        res = model.predict(crop, imgsz=self.imgsz, conf=self.conf,
                            device=self.device, verbose=False)[0]
        out: List[BBoxConf] = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            conf = float(b.conf[0])
            # map crop-local -> frame coords
            out.append((x1 + cx1i, y1 + cy1i, x2 + cx1i, y2 + cy1i, conf))
        return out
