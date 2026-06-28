"""
Module A -- Heavy Pass wrapper (cache producer).

Thin wrapper around the existing, validated tools/crossing_violation_test.build_cache. It
sets up the YOLOv8-seg lane model, points the vehicle tracker at BoT-SORT (camera-motion
compensation for the moving dashcam) instead of ByteTrack, and writes the per-frame JSON
cache the rest of the pipeline consumes. No per-frame logic lives here -- build_cache owns
all of it; we only inject models and the tracker choice.

CAUTION: switching ByteTrack -> BoT-SORT changes track ids, which invalidates caches built
under the old tracker AND requires re-validating the confidence model. See
rebuild_and_validate.py before trusting confidence numbers from these caches.
"""
from __future__ import annotations

import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_PIPELINE_DIR)
for _p in (_PIPELINE_DIR, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import crossing_violation_test as cvt  # noqa: E402  -- existing heavy pass + paths/loaders

DEFAULT_TRACKER = "botsort.yaml"


def run_heavy_pass(prefix: str, *, video_path: str | None = None,
                   lane_weights: str | None = None, veh_weights: str | None = None,
                   lane_conf: float = 0.25, veh_conf: float = 0.30,
                   tracker: str = DEFAULT_TRACKER, refresh: bool = False) -> dict:
    """Produce (or reuse) the per-frame cache for one clip and return the cache dict.

    refresh=False returns an existing cache if present (fast). refresh=True forces a rebuild
    under `tracker` -- use it after switching trackers, since a stale ByteTrack cache will not
    be detected as wrong automatically.
    """
    lane_weights = lane_weights or cvt.LANE_WEIGHTS
    veh_weights = veh_weights or cvt.VEH_WEIGHTS

    if not refresh:
        cached = cvt.load_cache(prefix)
        if cached is not None:
            print(f"[heavy_pass] {prefix}: cache hit ({cached['total']} frames) -- "
                  f"pass refresh=True to rebuild under {tracker}")
            return cached

    from ultralytics import YOLO  # heavy import deferred to call time (GPU env only)
    lane_model = YOLO(lane_weights, task="segment")
    print(f"[heavy_pass] {prefix}: lane={os.path.basename(lane_weights)} "
          f"veh={os.path.basename(veh_weights)} tracker={tracker}")
    cache = cvt.build_cache(prefix, lane_model, veh_weights, lane_conf, veh_conf,
                            path=video_path, tracker=tracker)
    if cache is None:
        raise FileNotFoundError(
            f"[heavy_pass] no video found for prefix {prefix!r} (looked in {cvt.CLIPS_DIR}); "
            f"pass video_path=... to point at it explicitly.")
    return cache
