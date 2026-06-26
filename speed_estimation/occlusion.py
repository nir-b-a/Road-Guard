"""
occlusion -- per-(vehicle, frame) occlusion fraction from inter-vehicle bbox
overlap, so the speed path can DOWN-WEIGHT occluded frames (note #2).

A vehicle partially hidden behind a NEARER vehicle has a truncated silhouette: its
bbox height/width (-> depth) and center (-> lateral) are corrupted on those frames,
which differentiates into a speed spike. We detect it geometrically, per frame:
for any two overlapping vehicle boxes the one whose BOTTOM edge (y2) sits LOWER in
the image is nearer (ground-plane projection -- nearer things project lower), so
the OTHER (higher y2) is the occluded one. Its occlusion fraction is the
intersection area over its OWN box area (how much of it is hidden); when several
vehicles occlude it, the worst single occluder wins.

The resulting {vehicle_id: {frame: fraction}} map feeds
speed_estimator.build_world_runs as a per-frame measurement-noise multiplier (the
existing noise_scale channel) -- the Kalman then coasts through occluded frames on
its model instead of chasing the corrupted box. SPEED only; the distance path
(estimateDistance / smooth_distances) is untouched.

Remove cleanly by deleting this file + the occ_by_frame/occlusion_gate plumbing in
speed_estimator and the --occlusion-gate flag in main.
"""

import Constants


def _inter_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Pixel area of the intersection of two xyxy boxes (0 if disjoint)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def compute_occlusion(world,
                      min_fraction: float = Constants.OCC_MIN_FRACTION
                      ) -> dict[int, dict[int, float]]:
    """Return {vehicle_id: {frame: occ_fraction}} for occluded (vehicle, frame)s only.

    occ_fraction in (0, 1] is the share of the vehicle's OWN bbox hidden behind the
    worst single nearer vehicle in that frame; entries below `min_fraction` are
    dropped (sparse map). A vehicle never occludes itself, and the frontmost
    vehicle in a frame is never occluded.
    """
    # Gather valid bboxes per frame: {frame: [(vid, bbox), ...]}.
    per_frame: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for vid, vehicle in world.vehicles.items():
        for f, b in vehicle.bounding_box.items():
            if b != (0, 0, 0, 0):
                per_frame.setdefault(f, []).append((vid, b))

    occ: dict[int, dict[int, float]] = {}
    for f, items in per_frame.items():
        if len(items) < 2:
            continue
        for vid_i, bi in items:
            area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
            if area_i <= 0:
                continue
            worst = 0.0
            for vid_j, bj in items:
                if vid_j == vid_i:
                    continue
                if bj[3] <= bi[3]:          # j must be NEARER (lower bottom edge) to occlude i
                    continue
                frac = _inter_area(bi, bj) / area_i
                if frac > worst:
                    worst = frac
            if worst >= min_fraction:
                occ.setdefault(vid_i, {})[f] = float(min(worst, 1.0))

    if occ:
        n_frames = sum(len(d) for d in occ.values())
        print(f"[occlusion] {n_frames} occluded (vehicle,frame)(s) across {len(occ)} vehicle(s) "
              f"(>= {min_fraction:.0%} hidden)")
    else:
        print("[occlusion] no occluded frames detected")
    return occ
