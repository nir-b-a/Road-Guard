import re
import cv2
import numpy as np
from abc import ABC, abstractmethod


def _preprocess_plate(crop: np.ndarray) -> np.ndarray:
    # upscale so small plates have enough pixels for OCR to work with
    h, w = crop.shape[:2]
    crop = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    # sharpen
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    crop = cv2.filter2D(crop, -1, kernel)
    # boost contrast via CLAHE on the L channel
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    crop = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return crop


def _validate_israeli_plate(text: str) -> str | None:
    digits = re.sub(r'\D', '', text)
    if len(digits) == 7:   # pre-2017: XX-XXX-XX
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
    if len(digits) == 8:
        if digits[0] == '0':  # likely OCR hallucinated a leading zero on a 7-digit plate
            return f"{digits[1:3]}-{digits[3:6]}-{digits[6:]}"
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"   # post-2017: XXX-XX-XXX
    return None


class LPRReader(ABC):
    @abstractmethod
    def read_plate(self, vehicle_crop: np.ndarray) -> str | None:
        """Returns a validated Israeli plate string (XX-XXX-XX or XXX-XX-XXX), or None."""
        pass

    def read_plate_with_conf(self, vehicle_crop: np.ndarray) -> tuple[str | None, float]:
        """Return (validated plate | None, OCR confidence in [0,1]).

        Default delegates to read_plate with a neutral confidence of 1.0. FastALPRReader
        overrides this with the real OCR confidence, which feeds the Module B pipeline score
        (vote_fraction * mean_OCR_conf). Other readers can override as their OCR exposes it."""
        return self.read_plate(vehicle_crop), 1.0

    def crop_plate(self, vehicle_crop: np.ndarray):
        """Return the tight plate-region sub-crop inside `vehicle_crop`, or None if no plate is
        localised. Used to build zoomed evidence images for human verification. Default: no
        localiser available -> None (callers fall back to the whole vehicle crop)."""
        return None

    def locate_plate(self, vehicle_crop: np.ndarray):
        """Return the plate bounding box as ((cx, cy), w, h, conf) in `vehicle_crop` pixel
        coords, or None if no plate is localised. Used by the Stage-2 cascade's LPR fallback to
        build a ground-projected axle-proxy (the geometry only needs the box, not the OCR text).
        Default: no localiser available -> None."""
        return None


def _ocr_confidence(conf: float | list[float]) -> float:
    return float(np.mean(conf)) if isinstance(conf, list) else conf


class FastALPRReader(LPRReader):
    def __init__(self):
        from fast_alpr import ALPR  # lazy import: only required when this reader is constructed
        # detector_model default: yolo-v9-t-384-license-plate-end2end
        # ocr_model: global-plates-mobile-vit-v2-model gives broader plate coverage
        self._alpr = ALPR(ocr_model="global-plates-mobile-vit-v2-model")

    def read_plate(self, vehicle_crop: np.ndarray) -> str | None:
        return self.read_plate_with_conf(vehicle_crop)[0]

    def read_plate_with_conf(self, vehicle_crop: np.ndarray) -> tuple[str | None, float]:
        results = self._alpr.predict(vehicle_crop)
        if not results:
            return None, 0.0
        best = max(results, key=lambda r: _ocr_confidence(r.ocr.confidence) if r.ocr else 0.0)
        if best.ocr is None:
            return None, 0.0
        return _validate_israeli_plate(best.ocr.text), float(_ocr_confidence(best.ocr.confidence))

    def crop_plate(self, vehicle_crop: np.ndarray):
        """Tight plate-region sub-crop via the detector box (highest-confidence plate), or None."""
        results = self._alpr.predict(vehicle_crop)
        if not results:
            return None
        best = max(results, key=lambda r: r.detection.confidence if r.detection else 0.0)
        if best.detection is None:
            return None
        bb = best.detection.bounding_box
        h, w = vehicle_crop.shape[:2]
        x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
        x2, y2 = min(w, int(bb.x2)), min(h, int(bb.y2))
        if x2 <= x1 or y2 <= y1:
            return None
        plate_crop = vehicle_crop[y1:y2, x1:x2]
        return plate_crop if plate_crop.size else None

    def locate_plate(self, vehicle_crop: np.ndarray):
        """Highest-confidence plate detection box as ((cx, cy), w, h, conf) in crop coords."""
        results = self._alpr.predict(vehicle_crop)
        if not results:
            return None
        best = max(results, key=lambda r: r.detection.confidence if r.detection else 0.0)
        if best.detection is None:
            return None
        bb = best.detection.bounding_box
        x1, y1, x2, y2 = float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None
        return (((x1 + x2) / 2.0, (y1 + y2) / 2.0), w, h, float(best.detection.confidence))


class PaddleOCRDetectorReader(LPRReader):
    def __init__(self, detector_path: str = "israeli_plates.pt", min_ocr_confidence: float = 0.6):
        from ultralytics import YOLO
        from paddleocr import PaddleOCR
        self._detector = YOLO(detector_path)
        self._ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)
        self._min_ocr_confidence = min_ocr_confidence

    def read_plate(self, vehicle_crop: np.ndarray) -> str | None:
        results = self._detector.predict(vehicle_crop, verbose=False, conf=0.3)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        best_idx = int(boxes.conf.argmax().item())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
        plate_crop = vehicle_crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None
        plate_crop = _preprocess_plate(plate_crop)
        result = self._ocr.ocr(plate_crop, det=False, cls=False)
        if not result or not result[0]:
            return None
        items = [item for item in result[0] if item]
        if not items:
            return None
        if _ocr_confidence([item[1] for item in items]) < self._min_ocr_confidence:
            return None
        text = ''.join(item[0] for item in items)
        return _validate_israeli_plate(text)


