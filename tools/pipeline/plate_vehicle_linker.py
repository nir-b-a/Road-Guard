"""
Dual-graph linking: associate license-plate tracks with vehicle tracks by spatial containment.

Plates are detected & tracked as their OWN objects (a second tracker over israeli_plates.pt
detections), decoupled from the vehicle tracker. A plate box in a frame links to the vehicle whose
box best CONTAINS it (containment = intersection / plate_area). Votes accumulate per
(plate_track, vehicle_track) across frames; each plate track is assigned the vehicle it sits inside
most often.

Why: this decouples WHEN the plate is readable from WHEN the vehicle violates. A plate read at any
moment in the vehicle's life attaches to that vehicle (hence to its violations) -- even if the
vehicle tracker dropped/switched its own id, the plate track bridges it. Pure logic, CPU-testable.
"""
from __future__ import annotations

from collections import defaultdict


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(plate_bbox, veh_bbox) -> float:
    """Fraction of the plate box covered by the vehicle box (1.0 = plate fully inside vehicle)."""
    ix1, iy1 = max(plate_bbox[0], veh_bbox[0]), max(plate_bbox[1], veh_bbox[1])
    ix2, iy2 = min(plate_bbox[2], veh_bbox[2]), min(plate_bbox[3], veh_bbox[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pa = _area(plate_bbox)
    return inter / pa if pa > 0 else 0.0


def link_frame(plate_dets, vehicle_dets, *, min_containment: float = 0.6):
    """For each plate detection in a frame, the best-containing vehicle (>= threshold).
    plate_dets/vehicle_dets: [{"track_id", "bbox"}]. Yields (plate_tid, veh_tid, containment)."""
    links = []
    for p in plate_dets:
        best_v, best_c = None, 0.0
        for v in vehicle_dets:
            c = containment(p["bbox"], v["bbox"])
            if c > best_c:
                best_v, best_c = v["track_id"], c
        if best_v is not None and best_c >= min_containment:
            links.append((p["track_id"], best_v, best_c))
    return links


def build_plate_vehicle_registry(frames, *, min_containment: float = 0.6) -> dict[int, int]:
    """Assign each plate track to the vehicle track it sits inside most (containment-weighted).

    frames: [{"plates": [{track_id,bbox}], "vehicles": [{track_id,bbox}]}].
    Returns: {plate_track_id: vehicle_track_id}.
    """
    votes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for fr in frames:
        for plate_tid, veh_tid, c in link_frame(fr.get("plates", []), fr.get("vehicles", []),
                                                 min_containment=min_containment):
            votes[plate_tid][veh_tid] += c
    return {ptid: max(vw, key=lambda k: (vw[k], k)) for ptid, vw in votes.items()}


def vehicle_to_plate_tracks(registry: dict[int, int]) -> dict[int, list[int]]:
    """Invert the registry: vehicle_track_id -> [plate_track_ids assigned to it]."""
    out: dict[int, list[int]] = defaultdict(list)
    for plate_tid, veh_tid in registry.items():
        out[veh_tid].append(plate_tid)
    return dict(out)
