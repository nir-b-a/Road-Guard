"""Tests for evidence.card_layout -- the pure geometry of the evidence page (no cv2 needed; the
shared suite mocks cv2, so the cv2 drawing path is validated by rendering a real clip instead)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evidence  # noqa: E402


def test_layout_slots_fit_inside_canvas():
    for w, h, n in [(1280, 720, 1), (720, 1280, 3), (1080, 1080, 2)]:
        lay = evidence.card_layout(w, h, n)
        assert len(lay["slots"]) == n
        for s in lay["slots"]:
            assert s["x"] >= 0 and s["y"] >= 0
            assert s["x"] + s["w"] <= w
            assert s["y"] + s["h"] <= h
        # title sits above the image row, caption sits below it (no overlap)
        assert lay["title"]["h"] < lay["slots"][0]["y"]
        assert lay["caption"]["y"] >= lay["slots"][0]["y"] + lay["slots"][0]["h"]


def test_layout_more_images_means_narrower_slots():
    one = evidence.card_layout(1280, 720, 1)["slots"][0]["w"]
    three = evidence.card_layout(1280, 720, 3)["slots"][0]["w"]
    assert three < one


def test_layout_clamps_zero_images_to_one_slot():
    assert len(evidence.card_layout(640, 480, 0)["slots"]) == 1


def test_layout_scales_font_with_resolution():
    small = evidence.card_layout(640, 360, 1)["font_scale"]
    big = evidence.card_layout(2560, 1440, 1)["font_scale"]
    assert big > small
