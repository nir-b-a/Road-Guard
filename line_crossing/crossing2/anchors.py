"""Stage 1 - vehicle bbox -> road contact.

Two jobs, and the second matters as much as the first:

  1. turn each box into a ground anchor plus a corrected FOOTPRINT interval, and
  2. mark the frames where a box simply cannot say where the car is.

On (1): the bbox bottom edge sits at the row of the NEAREST ground contact. Along
that row toward the far side of the vehicle, the corresponding road points are
closer to the camera than the vehicle actually is - road the car is not standing
on - so the box overhangs the true footprint on that side, worst at close range.
We identify the near side from the vanishing point (a vehicle left of it shows us
its right flank, so its bottom-right corner is a genuine wheel contact) and pull
the FAR edge inward by a fixed fraction of the box width. One parameter, tunable
by eye on the overlay, no ground homography required.

On (2): when the box is truncated at a frame border, or its bottom is hidden by a
nearer vehicle, y2 is not a ground contact at all - it is a point on bodywork, and
it can land anywhere. Those samples are dropped, not down-weighted. The near-field
gate (bbox width vs local lane width) lives here too: distant vehicles are neither
detected nor typed reliably, and they are not what this product is for.

Smoothing uses the SAME centered windows as Stage 0, so the ego-motion component
common to the vehicle and the lane cancels cleanly in the Stage 2 difference.
"""
from __future__ import annotations

import numpy as np

from .config import CrossingConfig
from .smoothing import smooth
from .types import FrameObservation, VehicleAnchors


def _bottom_occluded(box, others, cfg: CrossingConfig) -> bool:
    """True if a NEARER box (larger y2) covers the bottom strip of `box`, i.e. the
    wheels are hidden and y2 has ridden up onto bodywork."""
    x1, y1, x2, y2 = box
    h = max(1, y2 - y1)
    strip_top = y2 - 0.25 * h
    area = max(1.0, (x2 - x1) * (y2 - strip_top))
    for ox1, oy1, ox2, oy2 in others:
        if oy2 <= y2:                     # not nearer than us
            continue
        ox = max(0.0, min(x2, ox2) - max(x1, ox1))
        oy = max(0.0, min(y2, oy2) - max(strip_top, oy1))
        if ox * oy / area > cfg.occlusion_overlap:
            return True
    return False


def build_anchors(obs: list[FrameObservation], lane_width: np.ndarray, ys: np.ndarray,
                  van: np.ndarray, frame_width: int, frame_height: int,
                  cfg: CrossingConfig,
                  stats: dict | None = None) -> dict[int, VehicleAnchors]:
    """Run Stage 1 over every tracked vehicle. Returns vehicle id -> anchors.

    `stats`, when given, is filled with per-reason rejection counts - the only way to
    tell "the detector found nothing" from "the gates threw everything away"."""
    step = float(cfg.row_grid_step)
    tally = {k: 0 for k in ("total", "degenerate", "aspect", "truncated", "occluded",
                            "far_lane_gate", "far_abs_gate", "guard", "kept")}
    n_rows = lane_width.shape[1]
    per_vehicle: dict[int, list[tuple]] = {}
    boxes_by_frame = [[v.bbox for v in ob.vehicles] for ob in obs]

    for i, ob in enumerate(obs):
        others_all = boxes_by_frame[i]
        for v in ob.vehicles:
            x1, y1, x2, y2 = v.bbox
            w, h = x2 - x1, y2 - y1
            valid = True
            tally["total"] += 1

            if w <= 1 or h <= 1:
                valid, why = False, "degenerate"
            elif not (cfg.min_aspect <= w / max(h, 1) <= cfg.max_aspect):
                valid, why = False, "aspect"                   # merged or split box
            elif (x1 <= cfg.edge_margin_px or y1 <= cfg.edge_margin_px
                  or x2 >= frame_width - cfg.edge_margin_px
                  or y2 >= frame_height - cfg.edge_margin_px):
                valid, why = False, "truncated"                # y2 is not a contact
            elif _bottom_occluded(v.bbox, [b for b in others_all if b != v.bbox], cfg):
                valid, why = False, "occluded"                 # wheels hidden by a nearer car

            r = int(np.clip(round(y2 / step), 0, n_rows - 1))
            lw = float(lane_width[i, r]) if np.isfinite(lane_width[i, r]) else np.nan
            if valid and w < cfg.min_bbox_width_frac * frame_width:
                valid, why = False, "far_abs_gate"             # too small to judge, ruler or not
            if valid and np.isfinite(lw) and w < cfg.min_bbox_lane_frac * lw:
                valid, why = False, "far_lane_gate"            # near-field gate
            if valid:
                tally["kept"] += 1
            else:
                tally[why] += 1

            x_vp = van[i, 0] if np.isfinite(van[i, 0]) else frame_width / 2.0
            x_c = 0.5 * (x1 + x2)
            # Near side = the flank we can see. Left of the vanishing point we see
            # the vehicle's right side, so (x2, y2) is the real wheel contact.
            near_right = x_c < x_vp
            shrink = cfg.footprint_shrink * w
            x_l = x1 + (shrink if near_right else 0.0)
            x_r = x2 - (0.0 if near_right else shrink)

            per_vehicle.setdefault(v.track_id, []).append(
                (i, ob.frame_id, float(y2), x_c, x_l, x_r, float(w),
                 np.int8(1 if near_right else -1), valid, v.bbox, v.cls_name,
                 float(w) / max(float(h), 1.0)))

    out: dict[int, VehicleAnchors] = {}
    for vid, rows in per_vehicle.items():
        t = len(rows)
        idx = np.array([r[0] for r in rows], dtype=np.int64)
        fid = np.array([r[1] for r in rows], dtype=np.int64)
        valid = np.array([r[8] for r in rows], dtype=bool)

        # A track that appears already straddling is usually a fresh detection, not
        # a crossing; likewise the frames where it is about to be lost.
        g = cfg.track_edge_guard
        before = int(valid.sum())
        if g > 0 and t > 2 * g:
            valid[:g] = False
            valid[-g:] = False
        elif g > 0:
            valid[:] = False
        tally["guard"] += before - int(valid.sum())

        series = np.full((t, 5), np.nan, dtype=np.float64)
        for k, r in enumerate(rows):
            if valid[k]:
                series[k] = (r[2], r[3], r[4], r[5], r[6])
        series = smooth(series, cfg.smooth_median_half, cfg.smooth_mean_half)
        valid &= np.isfinite(series).all(axis=1)

        out[vid] = VehicleAnchors(
            vehicle_id=vid, idx=idx, frame_ids=fid,
            y_c=series[:, 0], x_c=series[:, 1], x_left=series[:, 2],
            x_right=series[:, 3], w_box=series[:, 4],
            aspect=np.array([r[11] for r in rows], dtype=np.float64),
            near_side=np.array([r[7] for r in rows], dtype=np.int8),
            valid=valid,
            cls_name=next((r[10] for r in rows if r[10]), None),
            boxes={r[1]: r[9] for r in rows},
        )
    if stats is not None:
        tally["vehicles"] = len(out)
        tally["vehicles_usable"] = sum(1 for v in out.values()
                                       if int(v.valid.sum()) >= cfg.min_valid_samples)
        stats["stage1"] = tally
    return out
