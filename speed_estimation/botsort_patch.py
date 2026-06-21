"""
Two monkey-patches for BotSort:

1. Kalman filter numerical instability fix
   Root cause: floating-point errors accumulate in the covariance matrix over
   many frames until eigenvalues go negative, breaking Cholesky decomposition.
   Fix: when cho_factor fails, project the matrix to the nearest positive
   definite matrix via eigenvalue clipping (symmetrize first, then clip).

2. Scale-aware ReID cutoff
   Root cause: BotSort's ReID appearance features are extracted from the
   detection's bounding box crop. For small/distant objects the crop is only
   ~30x20 pixels — too little information for reliable appearance matching.
   The combined ReID+IoU score falls below the match threshold even when IoU
   alone would have matched successfully, causing the track to be dropped and
   reassigned a new ID on the next frame.
   Fix: for any detection whose bounding box height is below REID_MIN_HEIGHT,
   disable ReID for that detection and fall back to IoU-only matching.
"""

import numpy as np
import scipy.linalg
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.utils import matching

# ── Patch 1: Kalman filter ────────────────────────────────────────────────────

_original_cho_factor = scipy.linalg.cho_factor


def _nearest_positive_definite(a):
    a = (a + a.T) / 2  # force exact symmetry before eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def _robust_cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    try:
        return _original_cho_factor(a, lower=lower, overwrite_a=overwrite_a, check_finite=check_finite)
    except np.linalg.LinAlgError:
        a_fixed = _nearest_positive_definite(a)
        return _original_cho_factor(a_fixed, lower=lower, overwrite_a=False, check_finite=check_finite)


scipy.linalg.cho_factor = _robust_cho_factor

# ── Patch 2: Scale-aware ReID cutoff ─────────────────────────────────────────

# Detections whose bounding box height (in pixels) is below this threshold
# will be matched by IoU only — ReID is disabled for them.
# Tune this value based on your camera resolution and typical vehicle distances.
REID_MIN_HEIGHT = 50

_original_get_dists = BOTSORT.get_dists


def _get_dists_with_size_cutoff(self, tracks, detections):
    dists = matching.iou_distance(tracks, detections)
    dists_mask = dists > (1 - self.proximity_thresh)

    if self.args.fuse_score:
        dists = matching.fuse_score(dists, detections)

    if self.args.with_reid and self.encoder is not None:
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
        emb_dists[dists_mask] = 1.0

        for i, det in enumerate(detections):
            if det.xywh[3] < REID_MIN_HEIGHT:
                emb_dists[:, i] = 1.0  # disable ReID for this detection

        dists = np.minimum(dists, emb_dists)
    return dists


BOTSORT.get_dists = _get_dists_with_size_cutoff
