import math
from Objects.World import World


def _default_focal_length():
    image_width = 1920
    FOV_horizontal = 90.0
    return (image_width / 2) / math.tan(math.radians(FOV_horizontal / 2))


def estimateDistance(world: World, focal_length_px: float | None = None):
    """
    Estimate distance for every vehicle in every frame.

    focal_length_px: focal length of the actual camera in pixels.
      - For CARLA simulations: leave as None (uses the FOV from calibration).
      - For real-world video: pass the value from the Android app so that
        distances are scaled correctly for the real camera.

    Uses a calibrated model (distance_model.py) when available.
    Falls back to the naive pinhole formula if no calibration exists.
    """
    try:
        import distance_model as dm

        fl_px: float = focal_length_px if focal_length_px is not None else getattr(dm, "CALIBRATION_FOCAL_LENGTH_PX", _default_focal_length())

        best_model = getattr(dm, "BEST_MODEL", "corrected_pinhole")
        if best_model == "corrected_pinhole":
            # Recompute A using the actual camera's focal length so that a
            # different camera does not introduce a proportional speed error.
            h_eff = getattr(dm, "H_EFF", dm.A_CP / fl_px)  # type: ignore[attr-defined]
            A = h_eff * fl_px
            b = getattr(dm, "B_CP", 0.0)
            def _dist(pixel_height):
                return A / (pixel_height + b)
        else:
            a_pl = getattr(dm, "A_PL", 1.0)
            b_pl = getattr(dm, "B_PL", -1.0)
            def _dist(pixel_height):
                return a_pl * (pixel_height ** b_pl)

        print(f"[speed_estimator] calibrated model={best_model}  fl={fl_px:.1f}px")
    except ImportError:
        fl = focal_length_px or _default_focal_length()
        def _dist(pixel_height):
            return 1.5 * fl / pixel_height
        print("[speed_estimator] WARNING: no distance_model.py — using uncalibrated pinhole")

    for vehicle in world.vehicles.values():
        for frame, (_, y1, _, y2) in vehicle.bounding_box.items():
            pixel_height = y2 - y1
            if pixel_height <= 0:
                vehicle.dist_per_frame[frame] = 0.0
            else:
                vehicle.dist_per_frame[frame] = _dist(pixel_height)
