"""
FrameTransform contract tests.

Purpose: when you pull the real `crop_top` and `model_input` from your chosen
UnLanedet/CLRerNet config in Phase 2, drop them into `CONFIGS` below and run
this. If a lane would be shifted vertically (the classic crop/resize bug),
`test_corners_map_to_cropped_region` fails loudly.

Run from the repo root with either:
    python -m unittest tests.test_frame_transform
    python -m pytest tests/test_frame_transform.py
"""

from __future__ import annotations

import unittest

from line_crossing.dl_lane_detector import FrameTransform

# (label, src_w, src_h, model_w, model_h, crop_top)
# Add your real Phase-2 config here and the invariants below will verify it.
CONFIGS = [
    ("clrernet_culane_default", 1920, 1080, 800, 320, 270),
    ("no_crop_square_resize",   1920, 1080, 640, 360, 0),
    ("heavy_crop",              1280, 720,  800, 320, 200),
]


class TestFrameTransform(unittest.TestCase):

    def _transforms(self):
        for label, sw, sh, mw, mh, ct in CONFIGS:
            yield label, FrameTransform(src_w=sw, src_h=sh, model_w=mw, model_h=mh, crop_top=ct)

    def test_round_trip_is_identity(self):
        """invert(forward(p)) == p for points inside the cropped region."""
        for label, t in self._transforms():
            with self.subTest(config=label):
                for x in (0.0, t.src_w * 0.37, t.src_w):
                    for y in (float(t.crop_top), (t.crop_top + t.src_h) / 2, float(t.src_h)):
                        mx, my = t.forward_point(x, y)
                        bx, by = t.invert_point(mx, my)
                        self.assertAlmostEqual(bx, x, places=6)
                        self.assertAlmostEqual(by, y, places=6)

    def test_corners_map_to_cropped_region(self):
        """
        The anti-vertical-shift guard. Model-space corners must invert to the
        corners of the CROPPED full-res region:
          * model top    (my=0)       -> full-res y == crop_top   (NOT 0)
          * model bottom (my=model_h) -> full-res y == src_h
        If crop_top is mis-set, the top corner check fails -> every lane shifted.
        """
        for label, t in self._transforms():
            with self.subTest(config=label):
                # top-left of model input
                x0, y0 = t.invert_point(0.0, 0.0)
                self.assertAlmostEqual(x0, 0.0, places=6)
                self.assertAlmostEqual(y0, float(t.crop_top), places=6)
                # bottom-right of model input
                x1, y1 = t.invert_point(float(t.model_w), float(t.model_h))
                self.assertAlmostEqual(x1, float(t.src_w), places=6)
                self.assertAlmostEqual(y1, float(t.src_h), places=6)

    def test_scale_factors(self):
        for label, t in self._transforms():
            with self.subTest(config=label):
                self.assertAlmostEqual(t.scale_x, t.model_w / t.src_w, places=9)
                self.assertAlmostEqual(t.scale_y, t.model_h / (t.src_h - t.crop_top), places=9)

    def test_cropped_rows_are_excluded(self):
        """A full-res point ABOVE the crop line maps to a negative model y."""
        for label, t in self._transforms():
            if t.crop_top == 0:
                continue
            with self.subTest(config=label):
                _, my = t.forward_point(t.src_w / 2, t.crop_top - 1)
                self.assertLess(my, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
