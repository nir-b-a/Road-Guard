"""
Removable lens-distortion correction for the Pixel 8 dashcam stream.

WHY: the speed pipeline assumes a pinhole camera, but the Pixel 8's recorded 1080p
video has real radial distortion (the checkerboard calibration measured k1=0.33,
k2=-2.45, k3=6.36). That distortion is ~0 on the optical axis and grows toward the
frame EDGES -- exactly where stationary vehicles' speed spikes. This module
undistorts each frame with the measured K + distortion BEFORE detection, so the
off-axis geometry becomes pinhole-valid and the wide-angle blow-up should shrink.

HOW IT'S WIRED: this is a self-contained EXPERIMENT, used by main.py behind a
`--undistort` flag. To remove it completely: delete this file and the two
clearly-marked "UNDISTORT (removable)" blocks in main.py.

IMPORTANT: use this ONLY on clips recorded WITHOUT on-device distortion correction
(the current Pixel 8 clips). If you later enable
CaptureRequest.DISTORTION_CORRECTION_MODE in the app, the recorded frames are
already rectified -- do NOT also undistort here (that double-corrects and bows the
geometry the other way).
"""

import json
import os

import cv2
import numpy as np

# Calibration report written by tools/calibrate_intrinsics.py (fx/fy/cx/cy + distortion).
DEFAULT_REPORT_NAME = "intrinsics_calibration_report.json"
# Fallback report for the Pixel 8 we calibrated (used when none sits next to the video).
_BUNDLED_REPORT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "real_vids", DEFAULT_REPORT_NAME)


def find_report(video_dir):
    """Locate the calibration report: next to the video first, else the bundled one."""
    local = os.path.join(video_dir, DEFAULT_REPORT_NAME)
    if os.path.exists(local):
        return local
    if os.path.exists(_BUNDLED_REPORT):
        return _BUNDLED_REPORT
    return None


class FrameUndistorter:
    """Precomputes a remap once, then rectifies each frame with cv2.remap (cheap).

    The output frames are expressed in the SAME camera matrix K (newCameraMatrix=K),
    so the pipeline keeps using the calibrated fx/fy/cx/cy with NO distortion term --
    the pinhole model is now valid edge to edge. `K_used` exposes that matrix so the
    caller can override its intrinsics to match (see main.py).
    """

    def __init__(self, K, dist, calib_wh):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64)
        self.calib_wh = calib_wh        # (w, h) the calibration was measured at
        self._maps = None               # (map1, map2), built lazily for the real frame size
        self._map_wh = None
        self.K_used = None              # K the OUTPUT (rectified) frames are expressed in
        self.valid_roi = None           # (x1,y1,x2,y2) usable rect in the rectified frame

    @classmethod
    def from_report(cls, report_path):
        with open(report_path) as f:
            r = json.load(f)
        K = [[r["fx"], 0.0, r["cx"]],
             [0.0, r["fy"], r["cy"]],
             [0.0, 0.0, 1.0]]
        d = r["distortion"]
        dist = [d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]]   # OpenCV order [k1,k2,p1,p2,k3]
        res = r.get("resolution", {})
        calib_wh = (int(res.get("width", 0)), int(res.get("height", 0)))
        return cls(K, dist, calib_wh)

    def _build_maps(self, w, h):
        K = self.K.copy()
        # If the frame size differs from the calibration resolution, scale fx/fy/cx/cy
        # (the distortion coeffs are normalized, so they carry over unchanged).
        cw, ch = self.calib_wh
        if cw and ch and (w, h) != (cw, ch):
            sx, sy = w / cw, h / ch
            K[0, 0] *= sx; K[0, 2] *= sx
            K[1, 1] *= sy; K[1, 2] *= sy
            print(f"[undistort] frame {w}x{h} != calib {cw}x{ch}; scaled K by ({sx:.3f},{sy:.3f})")
        # newCameraMatrix = K -> rectified frames keep the SAME intrinsics.
        map1, map2 = cv2.initUndistortRectifyMap(
            K, self.dist, None, K, (w, h), cv2.CV_16SC2)
        self._maps = (map1, map2)
        self._map_wh = (w, h)
        self.K_used = K
        self.valid_roi = self._compute_valid_roi(map1, map2, w, h)
        print(f"[undistort] ENABLED  fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} "
              f"cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}  dist={self.dist.tolist()}")
        print(f"[undistort] usable pixel rect (valid_roi) = {self.valid_roi} "
              f"(frame {w}x{h}); boxes touching the warped invalid band are dropped for speed")

    @staticmethod
    def _compute_valid_roi(map1, map2, w, h):
        """Largest axis-aligned rectangle of VALID (source-covered) output pixels.

        Undistorting with newCameraMatrix=K leaves a curved invalid band where the
        warp samples outside the input image (black border, mostly at the corners).
        We remap an all-valid mask and find the LARGEST all-valid rectangle (classic
        maximal-rectangle-in-a-binary-matrix via a monotonic stack). For speed it is
        solved on a DOWNSAMPLED mask whose coarse cell is valid only when ALL its
        pixels are -- so the returned full-res rectangle is guaranteed fully valid.
        Returns (x1, y1, x2, y2); the full frame if there is no invalid band.
        """
        mask = np.full((h, w), 255, np.uint8)
        valid = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0) == 255
        if valid.all():
            return (0, 0, w, h)

        ds = 4                                  # downsample step (px) for the search
        hh, ww = h // ds, w // ds
        if hh < 2 or ww < 2:
            return (0, 0, w, h)
        # coarse cell valid only if EVERY full-res pixel in it is valid (conservative)
        small = valid[:hh * ds, :ww * ds].reshape(hh, ds, ww, ds).all(axis=(1, 3))

        heights = np.zeros(ww, dtype=np.int64)
        best_area = 0
        best = None
        for y in range(hh):
            heights = np.where(small[y], heights + 1, 0)
            hl = heights.tolist()
            stack: list[tuple[int, int]] = []   # (start_x, bar_height)
            for x in range(ww + 1):
                cur = hl[x] if x < ww else 0
                start = x
                while stack and stack[-1][1] >= cur:
                    idx, ht = stack.pop()
                    area = ht * (x - idx)
                    if area > best_area:
                        best_area = area
                        best = (idx, y - ht + 1, x, y + 1)
                    start = idx
                stack.append((start, cur))
        if best is None:
            return (0, 0, w, h)
        sx1, sy1, sx2, sy2 = best            # in coarse cells; each cell fully valid
        return (sx1 * ds, sy1 * ds, sx2 * ds, sy2 * ds)

    def __call__(self, frame):
        if frame is None:
            return frame
        h, w = frame.shape[:2]
        if self._maps is None or self._map_wh != (w, h):
            self._build_maps(w, h)
        return cv2.remap(frame, self._maps[0], self._maps[1], interpolation=cv2.INTER_LINEAR)
