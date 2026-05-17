from Objects.Vehicle import Vehicle


def check_crossing(vehicle: Vehicle, detected_lines: dict[int, dict[str, tuple]]) -> int | None:
    """
    Checks if a vehicle ever crossed a solid line during its tracked lifetime.
    Returns the frame_id of the first detected crossing, or None if no crossing occurred.

    detected_lines: world.detected_lines — maps frame_id to detected line coordinates,
                    e.g. {42: {"solid_line_left": (x1, y1, x2, y2), ...}}

    Design note: both the previous and current point are tested against the *current*
    frame's line coordinates. This keeps the reference line consistent for each comparison
    and correctly handles the dynamic dashcam case where line coords shift every frame.
    """
    frame_ids = sorted(vehicle.bounding_box.keys())
    prev_point: tuple[int, int] | None = None

    for frame_id in frame_ids:
        bbox = vehicle.bounding_box.get(frame_id)
        if bbox is None or bbox == (0, 0, 0, 0):
            # Vehicle was absent this frame — reset so we don't compare across a gap
            prev_point = None
            continue

        curr_point = vehicle.bottomCenterInFrame(frame_id)

        if prev_point is not None:
            for lx1, ly1, lx2, ly2 in detected_lines.get(frame_id, {}).values():
                prev_side = _side_of_line(lx1, ly1, lx2, ly2, *prev_point)
                curr_side = _side_of_line(lx1, ly1, lx2, ly2, *curr_point)
                # Sign flip → bottom-center of vehicle moved across the line
                if prev_side != 0 and curr_side != 0 and prev_side * curr_side < 0:
                    return frame_id

        prev_point = curr_point

    return None


def _side_of_line(x1: int, y1: int, x2: int, y2: int, px: int, py: int) -> float:
    """
    Signed cross product of vector (x1,y1)→(x2,y2) and vector (x1,y1)→(px,py).
    Positive and negative values indicate opposite sides of the line.
    """
    return float((x2 - x1) * (py - y1) - (y2 - y1) * (px - x1))
