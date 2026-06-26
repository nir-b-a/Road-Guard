"""
Module D -- Joiner + Ego-Compensated Handshake.

Attaches each violation EVENT (Module C) to a license PLATE (Module B) by track_id.

When the violator's track has no plate -- typically because the tracker switched IDs
mid-life (the car was close & readable under id 7, then re-acquired as id 8 right when it
crossed the line) -- it tries to INHERIT the plate from a track that died 3-5 frames
earlier and whose final box, AFTER compensating for camera ego-motion, still overlaps the
violator's first box (IoU > 0.8).

Why ego compensation: on a moving dashcam the whole image translates between frames, so a
raw IoU across a multi-frame gap fails even for the same physical car. We sum the cached
per-frame `shift` vectors over the gap, translate the dead track's last box by that sum,
and only then measure overlap. The cache already stores `shift`, so this is free.

Consumes the unchanged cache schema (`frame`, `shift`, `vehicles[].track_id/bbox`) and
the Module B plate map (track_id -> {"plate_candidate", "plate_confidence_score"}).
"""
from __future__ import annotations

from typing import Optional

from lpr_consumer import build_track_index   # sibling import (pipeline dir on sys.path)

GAP_MIN = 3              # a predecessor must have died at least this many frames before
GAP_MAX = 5              # ...and at most this many (else it's not the same car re-acquired)
IOU_THRESHOLD = 0.8      # ego-compensated overlap required to inherit a plate


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def translate(bbox, dx: float, dy: float) -> list:
    return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]


def sum_shift(cache: dict, f_start: int, f_end: int) -> "tuple[float, float]":
    """Summed ego-motion over frames [f_start, f_end] inclusive (a static point's drift)."""
    shift_by_frame = {fr["frame"]: fr.get("shift", [0.0, 0.0]) for fr in cache["frames"]}
    dx = dy = 0.0
    for f in range(f_start, f_end + 1):
        s = shift_by_frame.get(f, [0.0, 0.0])
        dx += s[0]
        dy += s[1]
    return dx, dy


# --------------------------------------------------------------------------- #
# track endpoints + the two IoU flavours
# --------------------------------------------------------------------------- #
def track_endpoints(index: dict, tid: int):
    """(first_frame, first_bbox, last_frame, last_bbox) for a track, or None if absent."""
    seq = index.get(tid)
    if not seq:
        return None
    (f0, b0), (f1, b1) = seq[0], seq[-1]
    return f0, b0, f1, b1


def raw_iou(index: dict, old_tid: int, new_tid: int) -> float:
    """Naive overlap of the dead track's last box vs the new track's first box (no ego comp)."""
    old = track_endpoints(index, old_tid)
    new = track_endpoints(index, new_tid)
    if not old or not new:
        return 0.0
    return iou(old[3], new[1])


def ego_compensated_iou(cache: dict, index: dict, old_tid: int, new_tid: int) -> float:
    """Same overlap, but the dead box is first translated by the summed ego-motion over the gap."""
    old = track_endpoints(index, old_tid)
    new = track_endpoints(index, new_tid)
    if not old or not new:
        return 0.0
    old_last_frame, old_last_bbox = old[2], old[3]
    new_first_frame, new_first_bbox = new[0], new[1]
    dx, dy = sum_shift(cache, old_last_frame + 1, new_first_frame)
    return iou(translate(old_last_bbox, dx, dy), new_first_bbox)


# --------------------------------------------------------------------------- #
# handshake + join
# --------------------------------------------------------------------------- #
def find_predecessor(
    cache: dict,
    index: dict,
    new_tid: int,
    plate_map: dict,
    *,
    gap_min: int = GAP_MIN,
    gap_max: int = GAP_MAX,
    iou_threshold: float = IOU_THRESHOLD,
):
    """Best plate-bearing track that died 3-5 frames before `new_tid` was born and whose
    ego-compensated last box overlaps `new_tid`'s first box. Returns (old_tid, iou, plate_data)
    or None."""
    new = track_endpoints(index, new_tid)
    if not new:
        return None
    new_first_frame = new[0]
    best = None
    for old_tid, seq in index.items():
        if old_tid == new_tid:
            continue
        old_last_frame = seq[-1][0]
        gap = new_first_frame - old_last_frame
        if not (gap_min <= gap <= gap_max):
            continue
        pdata = plate_map.get(old_tid)
        if not pdata or not pdata.get("plate_candidate"):   # nothing worth inheriting
            continue
        score = ego_compensated_iou(cache, index, old_tid, new_tid)
        if score >= iou_threshold and (best is None or score > best[1]):
            best = (old_tid, score, pdata)
    return best


def join_events(
    events: list,
    plate_map: dict,
    cache: dict,
    *,
    gap_min: int = GAP_MIN,
    gap_max: int = GAP_MAX,
    iou_threshold: float = IOU_THRESHOLD,
) -> list:
    """Enrich each violation event with plate data: its own track's plate if present, else
    an inherited plate via the ego-compensated handshake, else nulls.

    Input event: {"violation_id", "track_id", "start_frame", "end_frame", "confidence"}
    Output adds: "plate_candidate", "plate_confidence_score", "plate_source"
                 (+ "inherited_from"/"inherited_iou" when a handshake fired).
    """
    index = build_track_index(cache)
    joined = []
    for ev in events:
        result = dict(ev)
        own = plate_map.get(ev["track_id"])
        if own and own.get("plate_candidate"):
            result.update(plate_candidate=own["plate_candidate"],
                          plate_confidence_score=own["plate_confidence_score"],
                          plate_source="own")
        else:
            pred = find_predecessor(cache, index, ev["track_id"], plate_map,
                                    gap_min=gap_min, gap_max=gap_max, iou_threshold=iou_threshold)
            if pred:
                old_tid, score, pdata = pred
                result.update(plate_candidate=pdata["plate_candidate"],
                              plate_confidence_score=pdata["plate_confidence_score"],
                              plate_source=f"inherited:{old_tid}",
                              inherited_from=old_tid,
                              inherited_iou=round(score, 4))
            else:
                result.update(plate_candidate=None, plate_confidence_score=0.0,
                              plate_source="none")
        joined.append(result)
    return joined
