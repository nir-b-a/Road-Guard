"""
Quick standalone test: run fast-alpr on sampled frames of a video.
Usage: python lpr/test_lpr.py <video_path> [sample_every_n_frames]

Runs the full ALPR on whole frames (no YOLO pre-crop) so we can see
whether fast-alpr detects Israeli plates at all before full integration.
"""

import re
import sys
import cv2
import numpy as np
from fast_alpr import ALPR


def _ocr_confidence(conf):
    return float(np.mean(conf)) if isinstance(conf, list) else float(conf)


def _validate_israeli_plate(text: str):
    digits = re.sub(r'\D', '', text)
    if len(digits) == 7:
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
    if len(digits) == 8:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    return None


def test_video(video_path: str, sample_every: int = 15):
    alpr = ALPR(ocr_model="global-plates-mobile-vit-v2-model")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path}  |  {total} frames @ {fps:.1f} fps")
    print(f"Sampling every {sample_every} frames\n")

    found: list[tuple[int, str, float]] = []   # (frame_idx, plate, confidence)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            results = alpr.predict(frame)
            for r in results:
                if r.ocr is None:
                    continue
                conf = _ocr_confidence(r.ocr.confidence)
                validated = _validate_israeli_plate(r.ocr.text)
                status = f"VALID → {validated}" if validated else f"invalid ({r.ocr.text})"
                print(f"  frame {frame_idx:>5}  conf={conf:.2f}  {status}")
                if validated:
                    found.append((frame_idx, validated, conf))

        frame_idx += 1

    cap.release()

    print(f"\n{'='*50}")
    print(f"Valid Israeli plates found: {len(found)}")
    for frame_idx, plate, conf in found:
        print(f"  frame {frame_idx}  {plate}  (conf={conf:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lpr/test_lpr.py <video_path> [sample_every]")
        sys.exit(1)
    sample_every = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    test_video(sys.argv[1], sample_every)
