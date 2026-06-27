"""
Linear interpolation of short bounding-box gaps in a Vehicle's track.

When YOLO misses a vehicle for a few consecutive frames (a lamppost
occlusion, motion blur, partial occlusion behind a truck) BotSort may
still keep the track alive but no bbox is recorded.  This module fills
those short gaps via per-corner linear interpolation between the two
surrounding real detections.

Gaps longer than `max_gap_seconds` are LEFT EMPTY because linear
interpolation cannot be trusted over long periods -- the vehicle may
have changed speed, direction, or even identity during the gap.

Run order:
    process all frames with YOLO   ->
    interpolate_bbox_gaps(world)   ->
    estimateDistance / estimate_world_speeds
"""

from Objects.World import World


def _is_placeholder(bbox):
    return bbox is None or bbox == (0, 0, 0, 0)


def interpolate_bbox_gaps(world: World, fps: float, max_gap_seconds: float = 0.5):
    """
    Fill short bbox gaps in every vehicle track with linear interpolation.

    Only frames between the vehicle's first and last real detection are
    considered -- the track is never extrapolated forward or backward.
    """
    max_gap_frames = int(max_gap_seconds * fps)

    filled_total = 0
    skipped_total = 0

    for vehicle in world.vehicles.values():
        real_frames = sorted(
            f for f, b in vehicle.bounding_box.items()
            if not _is_placeholder(b)
        )
        if len(real_frames) < 2:
            continue

        for i in range(len(real_frames) - 1):
            f0, f1 = real_frames[i], real_frames[i + 1]
            gap = f1 - f0 - 1
            if gap <= 0:
                continue
            if gap > max_gap_frames:
                skipped_total += gap
                continue

            b0 = vehicle.bounding_box[f0]
            b1 = vehicle.bounding_box[f1]
            for j in range(1, gap + 1):
                t = j / (gap + 1)
                interp = tuple(
                    int(round(b0[k] * (1 - t) + b1[k] * t))
                    for k in range(4)
                )
                vehicle.bounding_box[f0 + j] = interp
                filled_total += 1

    print(f"[bbox_interpolator] filled {filled_total} frames; "
          f"left {skipped_total} frames in gaps longer than {max_gap_seconds}s.")
