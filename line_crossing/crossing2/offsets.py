"""Stage 2 - the signed lateral offset.

Collapses (vehicle, line, frame) into one scalar with physical meaning:

    u = (x_vehicle - x_line(y_contact)) / W(y_contact)

Three properties earn this its place as the only quantity Stage 3 looks at:

  * The SIGN is defined by the line, not by the image. Positive means the vehicle
    is to the right of that marking, everywhere in the frame, for both crossing
    directions. No "which side of image centre is the line on" special-casing, and
    no direction the test is blind to.
  * EGO MOTION cancels to first order. Yaw, roll and suspension move the vehicle
    and the marking together in the image; their difference barely moves. That is
    why the offset is the right invariant and a raw image-space trajectory is not,
    and why no optical-flow compensation is needed anywhere in this module.
  * SCALE cancels. Dividing by the local lane width makes u dimensionless: u = 0.15
    is about half a metre at any distance, any focal length, any resolution.

Two details here are silent recall killers if skipped: extrapolating a lane track
slightly below its detected extent (a close vehicle's contact row is often lower
than the lowest row the lane model emitted - exactly the samples that matter most),
and interpolating short gaps so one crossing does not split into two half-events.
"""
from __future__ import annotations

import numpy as np

from .config import CrossingConfig
from .smoothing import interp_gaps, nan_median_filter
from .types import LineTrack, OffsetSeries, VehicleAnchors


def _row_extent(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, the first/last row index where the track has geometry."""
    ok = np.isfinite(X)
    any_ok = ok.any(axis=1)
    r_lo = np.where(any_ok, np.argmax(ok, axis=1), -1)
    r_hi = np.where(any_ok, X.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1), -1)
    return r_lo, r_hi


def _x_at(X: np.ndarray, i: int, row: int, r_lo: int, r_hi: int,
          max_extrap: int) -> tuple[float, bool]:
    """Line x at (frame i, row), extending DOWN the bottom tangent when the vehicle's
    contact row falls below the track's detected extent. Returns (x, extrapolated)."""
    if r_hi < 0:
        return np.nan, False
    if r_lo <= row <= r_hi:
        v = X[i, row]
        if np.isfinite(v):
            return float(v), False
    if row > r_hi and (row - r_hi) <= max_extrap:
        r_a = max(r_lo, r_hi - 5)
        if r_a < r_hi and np.isfinite(X[i, r_a]) and np.isfinite(X[i, r_hi]):
            slope = (X[i, r_hi] - X[i, r_a]) / float(r_hi - r_a)
            return float(X[i, r_hi] + slope * (row - r_hi)), True
    return np.nan, False


def build_offsets(anchors: dict[int, VehicleAnchors], tracks: list[LineTrack],
                  lane_width: np.ndarray, ys: np.ndarray, fps: float,
                  cfg: CrossingConfig, stats: dict | None = None) -> list[OffsetSeries]:
    """Run Stage 2 for every (vehicle, SOLID line) pair that ever comes close."""
    step = float(cfg.row_grid_step)
    tally = {k: 0 for k in ("pairs", "too_few_samples", "never_near", "kept",
                            "no_line_geometry", "no_lane_width", "extrapolated")}
    n_rows = lane_width.shape[1]
    # 0 must mean 0: with the continuation switched off, a contact row below the
    # track's detected extent is discarded rather than guessed at.
    max_extrap = (max(1, int(round(cfg.extrapolate_frac * n_rows)))
                  if cfg.extrapolate_frac > 0.0 else 0)
    gap = cfg.frames(cfg.gap_interp_sec, fps)
    solid = [t for t in tracks if t.is_solid and not t.suppressed]
    r_lo_cache = {t.track_id: _row_extent(t.X) for t in solid}
    out: list[OffsetSeries] = []

    for va in anchors.values():
        if not va.valid.any():
            continue
        mult = cfg.width_mult.get(va.cls_name or "", cfg.default_width_mult)

        for tr in solid:
            r_lo_all, r_hi_all = r_lo_cache[tr.track_id]
            t = len(va.idx)
            u = np.full((t, 3), np.nan, dtype=np.float64)
            extrap = np.zeros(t, dtype=bool)

            for k in range(t):
                if not va.valid[k]:
                    continue
                i = int(va.idx[k])
                row = int(np.clip(round(va.y_c[k] / step), 0, n_rows - 1))
                x_l, ex = _x_at(tr.X, i, row, int(r_lo_all[i]), int(r_hi_all[i]), max_extrap)
                if not np.isfinite(x_l):
                    tally["no_line_geometry"] += 1
                    continue
                w = lane_width[i, row]
                if not np.isfinite(w) or w <= 1.0:
                    w = mult * va.w_box[k]                 # single line visible: car as the ruler
                    tally["no_lane_width"] += 1
                if not np.isfinite(w) or w <= 1.0:
                    continue
                u[k] = ((va.x_c[k] - x_l) / w,
                        (va.x_left[k] - x_l) / w,
                        (va.x_right[k] - x_l) / w)
                extrap[k] = ex
                tally["extrapolated"] += int(ex)

            tally["pairs"] += 1
            finite = np.isfinite(u[:, 0])
            if finite.sum() < cfg.min_valid_samples:
                tally["too_few_samples"] += 1
                continue
            if float(np.abs(u[finite, 0]).min()) > cfg.max_pair_offset:
                tally["never_near"] += 1
                continue                                    # never near this marking
            tally["kept"] += 1

            u = interp_gaps(u, gap)
            u = nan_median_filter(u, 1)                     # kill row-quantization spikes
            out.append(OffsetSeries(
                vehicle_id=va.vehicle_id, track_id=tr.track_id,
                idx=va.idx, frame_ids=va.frame_ids,
                u_c=u[:, 0], u_l=u[:, 1], u_r=u[:, 2], extrapolated=extrap,
                aspect=va.aspect))
    if stats is not None:
        tally["solid_tracks"] = len(solid)
        stats["stage2"] = tally
    return out
