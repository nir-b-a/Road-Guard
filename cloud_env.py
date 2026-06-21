"""
cloud_env.py -- one place that makes the whole pipeline natively cloud/Colab aware.

It removes the two recurring pains of moving runs between a local machine and Colab:

  1. Headless crashes. On Colab (and any server with no X display) cv2.imshow /
     cv2.waitKey / cv2.destroyAllWindows raise or hang. We detect that environment
     ONCE and monkeypatch those GUI calls to harmless no-ops, so debug-window code
     that used to be hand-commented before every Colab run now silently does nothing.

  2. Slow frame-by-frame writes straight to Google Drive. Drive is a FUSE mount; a
     cv2.VideoWriter writing every frame to it is slow and disconnect-prone.
     staged_output() writes to local Colab NVMe (/content) instead, then shutil.copy's
     the finished file to Drive once, at the very end.

Detection (any of these makes us "colab/headless"):
  * the real google.colab runtime is importable, or
  * Linux with no $DISPLAY / $WAYLAND_DISPLAY, or
  * an explicit  --colab  token in argv, or  ROADGUARD_COLAB=1  in the env.

Call init() once at startup (main.py does this). Everything else keys off the flags
it sets, so the rest of the codebase never has to know whether it is in the cloud.
"""
from __future__ import annotations

import os
import sys
import shutil
from contextlib import contextmanager

import cv2

# Colab's fast local NVMe scratch, and the Drive FUSE mount we stage away from.
_LOCAL_SCRATCH = "/content"
_DRIVE_MARKER = "/content/drive"

# Populated by init(); safe defaults for a plain local desktop run.
IS_COLAB = False
HEADLESS = False
_INITIALIZED = False


def in_colab() -> bool:
    """True iff the real Colab runtime is importable."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _is_headless() -> bool:
    """No usable GUI: Colab, or Linux without a display server."""
    if in_colab():
        return True
    if sys.platform.startswith("linux"):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False        # Windows / macOS desktops always have a display


def _forced() -> bool:
    """Explicit override: --colab argv token or ROADGUARD_COLAB env var."""
    if "--colab" in sys.argv:
        return True
    return os.environ.get("ROADGUARD_COLAB", "").strip().lower() in ("1", "true", "yes", "on")


def _noop(*_args, **_kwargs):
    # cv2.waitKey is expected to return an int (-1 == "no key pressed"); returning -1
    # keeps any caller that branches on the keycode happy.
    return -1


def neutralize_cv2_gui() -> None:
    """Replace every cv2 windowing call with a no-op so headless runs never crash/hang."""
    for fn in ("imshow", "waitKey", "waitKeyEx", "pollKey",
               "destroyAllWindows", "destroyWindow", "namedWindow",
               "setWindowProperty", "setWindowTitle", "getWindowProperty",
               "startWindowThread", "moveWindow", "resizeWindow"):
        if hasattr(cv2, fn):
            setattr(cv2, fn, _noop)


def init(*, force_colab: bool = False, verbose: bool = True) -> bool:
    """Detect the environment once and neutralize cv2 GUI if headless.

    Idempotent: safe to call from several entry points. Returns IS_COLAB.
    """
    global IS_COLAB, HEADLESS, _INITIALIZED
    if _INITIALIZED:
        return IS_COLAB
    forced = _forced() or force_colab
    IS_COLAB = in_colab() or forced
    HEADLESS = _is_headless() or forced
    if HEADLESS:
        neutralize_cv2_gui()
    _INITIALIZED = True
    if verbose:
        print(f"[cloud_env] colab={IS_COLAB} headless={HEADLESS} "
              f"(cv2 GUI {'neutralized' if HEADLESS else 'active'})")
    return IS_COLAB


def on_drive(path: str) -> bool:
    """True if `path` lives under the Google Drive FUSE mount."""
    return os.path.abspath(path).startswith(_DRIVE_MARKER)


@contextmanager
def staged_output(final_path: str, *, enabled: bool | None = None):
    """Write a large output to fast local NVMe, then copy it to its final home on Drive.

    Yields the path the caller should actually WRITE to:
      * On Colab, when `final_path` is on the Drive mount, it yields a local
        /content/<name> path and shutil.copy's the finished file to `final_path`
        (then deletes the local copy) on exit -- so frames are never written to the
        slow Drive FUSE mount one at a time.
      * Otherwise it is a transparent no-op and yields `final_path` unchanged.

    Pass enabled=True/False to force/disable staging regardless of environment.
    """
    if enabled is None:
        enabled = IS_COLAB and on_drive(final_path)

    if not enabled:
        os.makedirs(os.path.dirname(os.path.abspath(final_path)) or ".", exist_ok=True)
        yield final_path
        return

    os.makedirs(_LOCAL_SCRATCH, exist_ok=True)
    local = os.path.join(_LOCAL_SCRATCH, os.path.basename(final_path))
    print(f"[cloud_env] staging on local NVMe: {local} (copies to Drive at the end)")
    try:
        yield local
    finally:
        if os.path.isfile(local):
            os.makedirs(os.path.dirname(os.path.abspath(final_path)) or ".", exist_ok=True)
            print(f"[cloud_env] copying {local} -> {final_path}")
            shutil.copy(local, final_path)
            try:
                os.remove(local)            # free the ephemeral NVMe copy
            except OSError:
                pass
