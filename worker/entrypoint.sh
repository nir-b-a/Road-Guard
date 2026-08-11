#!/bin/sh
# =============================================================================
# Road Guard worker - container entrypoint.
#
# Its whole job is to make the large third-party weights persist OUTSIDE the
# image, in the models-cache volume, and still appear where the code expects
# them. Two constraints force this shape:
#
#   * Constants.YOLO_VERSION is the bare relative name "yolo11x.pt", and
#     tools/fetch_models.py chdir()s to the repo root before letting ultralytics
#     download it. So the file must be reachable at /app/yolo11x.pt.
#   * Baking those 114 MB into a layer would make every rebuild re-ship them and
#     would pin the image to one upstream release.
#
# So: symlink from /app into the volume, fetch whatever is still missing, then
# move any freshly downloaded real file back onto the volume and re-link. The
# move-after-fetch step is not redundant - torch/ultralytics download to a temp
# file and os.replace() it onto the target, which REPLACES a symlink with a real
# file rather than writing through it.
#
# HOME also lives on the volume: fast_alpr caches its two ONNX models under
# $HOME/.cache, and ultralytics keeps settings.json under $YOLO_CONFIG_DIR.
# Without this every container start re-downloads them.
# =============================================================================
set -e

CACHE="${MODEL_CACHE_DIR:-/models-cache}"

# Weights that are NOT in the repo and are fetched on demand. Keep in sync with
# the `source="ultralytics"` entries in tools/fetch_models.py.
AUTO_WEIGHTS="yolo11x.pt"

export HOME="$CACHE/home"
export YOLO_CONFIG_DIR="$CACHE/ultralytics"
mkdir -p "$HOME" "$YOLO_CONFIG_DIR"

# --- 1. expose anything already cached ---------------------------------------
for w in $AUTO_WEIGHTS; do
    if [ -f "$CACHE/$w" ] && [ ! -e "/app/$w" ]; then
        ln -s "$CACHE/$w" "/app/$w"
        echo "[entrypoint] linked cached $w"
    fi
done

# --- 2. fetch + verify -------------------------------------------------------
# fetch_models.py exits non-zero when a REQUIRED weight is unusable. Failing here
# is deliberate: a worker that starts, claims a job, and only then discovers it
# has no detector is far worse than one that refuses to start. Set
# SKIP_MODEL_FETCH=1 for an air-gapped run where the volume is pre-populated.
if [ "${SKIP_MODEL_FETCH:-0}" = "1" ]; then
    echo "[entrypoint] SKIP_MODEL_FETCH=1 - not verifying models"
else
    echo "[entrypoint] verifying models (this downloads ~114 MB on a cold cache)"
    python tools/fetch_models.py
fi

# --- 3. push anything newly downloaded onto the volume -----------------------
for w in $AUTO_WEIGHTS; do
    if [ -f "/app/$w" ] && [ ! -L "/app/$w" ]; then
        mv "/app/$w" "$CACHE/$w"
        ln -s "$CACHE/$w" "/app/$w"
        echo "[entrypoint] cached $w onto the volume"
    fi
done

exec "$@"
