"""Build-time (and run-time) sanity gate for the worker image.

Run as the last layer of each worker Dockerfile so a broken dependency set fails
the BUILD instead of failing the first job at 3 a.m. It is also invoked by
tests/test_docker_stack.py against the finished images.

    python worker/verify_env.py --expect cpu
    python worker/verify_env.py --expect cuda

What it actually catches:

* the headed/headless OpenCV collision. ultralytics declares a dependency on
  ``opencv-python``; pip installs it happily next to ``opencv-python-headless``
  and the two unpack over each other's ``cv2/``. Whichever lands last wins, so
  the image can silently end up with the headed build that needs libGL.
* a torch that does not match the image. Installing anything from plain PyPI
  after the pinned wheel can swap a cu118 build for a CPU one (GPU inference
  then "works", 40x slower) or the reverse.
* the pipeline modules not being importable at all - a .dockerignore allowlist
  that dropped a package shows up here, not three layers later.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

# Running `python worker/verify_env.py` puts sys.path[0] at worker/, not the repo
# root, so the first-party imports below would fail for a reason that has nothing
# to do with the image. Same fix as the root conftest.py.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Imported for real, not just checked for presence on disk: a numpy/OpenCV ABI
# mismatch only surfaces at import time.
PIPELINE_MODULES = [
    "numpy", "cv2", "torch", "torchvision", "ultralytics",
    "scipy", "pandas", "matplotlib", "boto3", "requests", "onnxruntime",
]

# First-party packages the worker reaches through main.run_pipeline.
FIRST_PARTY = ["Constants", "cloud_env", "video_handler", "Objects.World", "violations.event"]


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    fail.count += 1          # type: ignore[attr-defined]


fail.count = 0               # type: ignore[attr-defined]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", choices=["cpu", "cuda"], required=True,
                    help="which torch build this image is supposed to carry")
    args = ap.parse_args()

    print(f"Road Guard worker image check (expecting torch: {args.expect})")

    installed = {}
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name:
            installed[name] = dist.version

    # --- OpenCV: exactly one flavour, and it must be the headless one ---------
    if "opencv-python-headless" not in installed:
        fail("opencv-python-headless is not installed")
    if "opencv-python" in installed:
        fail("the headed opencv-python is installed and will shadow the headless "
             "cv2 - the image needs libGL it does not have")
    if "opencv-contrib-python" in installed:
        fail("opencv-contrib-python is installed and also provides cv2")

    # --- excluded heavyweights ----------------------------------------------
    for unwanted in ("paddlepaddle", "paddleocr", "easyocr"):
        if unwanted in installed:
            fail(f"{unwanted} was pulled in; it is not on the active LPR path")

    # --- everything imports --------------------------------------------------
    for mod in PIPELINE_MODULES + FIRST_PARTY:
        try:
            __import__(mod)
        except Exception as exc:                       # noqa: BLE001 - report, don't raise
            fail(f"import {mod}: {type(exc).__name__}: {exc}")

    # --- torch flavour matches the image -------------------------------------
    try:
        import torch
        cuda_ver = torch.version.cuda
        print(f"  torch {torch.__version__}  cuda={cuda_ver}")
        if args.expect == "cpu" and cuda_ver is not None:
            fail(f"CPU image carries a CUDA build (cuda={cuda_ver}); the pinned "
                 "CPU wheel was overwritten from PyPI")
        if args.expect == "cuda":
            if cuda_ver is None:
                fail("GPU image carries a CPU-only torch - inference would fall back "
                     "to CPU silently")
            elif not cuda_ver.startswith("11.8"):
                fail(f"GPU image built against CUDA {cuda_ver}, but the base image "
                     "and the GTX-1060-compatible pin are cu118")
    except ImportError:
        pass                                            # already reported above

    import cv2
    print(f"  cv2 {cv2.__version__} (headless)")

    if fail.count:                                      # type: ignore[attr-defined]
        print(f"\n{fail.count} problem(s) - image is not usable.")  # type: ignore[attr-defined]
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
