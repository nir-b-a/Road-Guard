"""
Appearance embedders for Module F (reid_linker). An embedder is a callable: vehicle_crop -> 1-D
vector; reid_linker compares vectors by cosine similarity. Embedders are injected so the linking
core stays model-free and CPU-testable.

  HistogramEmbedder -- cheap HSV colour histogram (cv2 only, no model). Good baseline / fallback,
                       runs anywhere. Weak under big lighting/pose changes.
  OSNetEmbedder     -- production vehicle Re-ID (torchreid OSNet). Best accuracy; needs the
                       torchreid dependency + weights. Lazy-imported so this module loads without it.

build_track_embeddings() picks each track's SHARPEST/largest crop (area x Laplacian, same readiness
score used for plates) and embeds it -> {track_id: vector} for reid_linker.
"""
from __future__ import annotations

import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import lpr_consumer  # noqa: E402  -- bbox_area / laplacian_variance / build_track_index


class HistogramEmbedder:
    """Normalised HSV colour histogram. Cheap, model-free, CPU-only."""

    def __init__(self, h_bins: int = 16, s_bins: int = 16, v_bins: int = 8):
        self.bins = (h_bins, s_bins, v_bins)

    def __call__(self, crop):
        import cv2
        import numpy as np
        if crop is None or getattr(crop, "size", 0) == 0:
            return [0.0]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(np.float32).tolist()


class OSNetEmbedder:
    """torchreid OSNet vehicle/person Re-ID embedding (production). Lazy-loaded."""

    def __init__(self, model_name: str = "osnet_x1_0", device: str = "cuda"):
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError as e:
            raise RuntimeError(
                "OSNetEmbedder needs torchreid. Install it in roadguard-dl "
                "(pip install torchreid) or use HistogramEmbedder.") from e
        self._extractor = FeatureExtractor(model_name=model_name, device=device)

    def __call__(self, crop):
        if crop is None or getattr(crop, "size", 0) == 0:
            return [0.0]
        feats = self._extractor([crop])           # (1, D) tensor
        return feats[0].cpu().numpy().tolist()


def build_track_embeddings(cache: dict, index: dict, frame_provider, embedder, *,
                           track_ids=None, min_area: float = lpr_consumer.DEFAULT_MIN_AREA) -> dict:
    """{track_id: embedding} from each track's SHARPEST large crop. `track_ids` limits the set
    (e.g. only violating + recently-dead tracks); None = all tracks in the index."""
    tids = list(index) if track_ids is None else list(track_ids)
    out = {}
    for tid in tids:
        best, best_crop = -1.0, None
        for frame_id, bbox in index.get(tid, []):
            if lpr_consumer.bbox_area(bbox) <= min_area:
                continue
            frame = frame_provider(frame_id)
            if frame is None:
                continue
            x1, y1, x2, y2 = (int(round(c)) for c in bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if getattr(crop, "size", 0) == 0:
                continue
            score = lpr_consumer.bbox_area(bbox) * lpr_consumer.laplacian_variance(crop)
            if score > best:
                best, best_crop = score, crop
        if best_crop is not None:
            out[int(tid)] = embedder(best_crop)
    return out
