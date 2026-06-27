"""Unit tests for renderer.scale_params -- the aspect-ratio-aware UI scaling (no cv2 needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import renderer  # noqa: E402


def test_scale_uses_shorter_side_so_vertical_isnt_oversized():
    wide = renderer.scale_params(1280, 720)     # 16:9
    tall = renderer.scale_params(720, 1280)     # 9:16, same short side (720)
    # font/thickness derive from min(w,h)=720 for both -> identical, not blown up on the tall one
    assert wide["font_scale"] == tall["font_scale"]
    assert wide["thick"] == tall["thick"]


def test_scale_grows_with_resolution():
    small = renderer.scale_params(640, 360)
    big = renderer.scale_params(2560, 1440)
    assert big["font_scale"] > small["font_scale"]
    assert big["thick"] >= small["thick"]


def test_scale_has_sane_floors():
    p = renderer.scale_params(120, 90)          # tiny frame
    assert p["thick"] >= 2 and p["font_scale"] >= 0.5 and p["banner_h"] >= 26
