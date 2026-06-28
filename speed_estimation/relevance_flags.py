"""
relevance_flags -- decide which tracked vehicles are probably NOT on our road, so
the speed path can REJECT them (note #1). Reconstructs the SAME world geometry the
speed estimator uses (same per-class estimator, ego pose, handedness), so
depth/lateral/Tx/Ty match the speed path exactly.

Two reject criteria, each judged over the vehicle's tracked life and OR'd; a
criterion fires only if it holds for >= REJECT_LIFE_FRACTION of life (so a
momentary wide bearing or one noisy frame can't reject a real target):

    "big_lateral"  -- |cross-track offset from the optical axis| > REJECT_LATERAL_M.
                      A large, PERSISTENT lateral offset suggests another
                      road/lane far from ego, not our lane.
    "direction"    -- the target's OWN motion pointed TOWARD the ego (its world
                      velocity has a positive component along target->ego). That
                      is oncoming traffic, usually on the other carriageway.
                      Near-stationary frames (< TOWARD_MIN_SPEED_MPS) don't vote.

A vehicle that fires either criterion is rejected: main.run_speed_estimation feeds
the id set to estimate_world_speeds, which skips it (no speed -> no plot, no
overspeed). The reason strings are for logging only. This is a HARD filter
(Constants warns it can backfire on far on-road traffic / curves).

Remove cleanly by deleting this file + its call in main.run_speed_estimation +
the reject_ids plumbing in speed_estimator.estimate_world_speeds.
"""

import math
import numpy as np

import Constants
from Constants import VEHICLE_HEIGHTS, DEFAULT_VEHICLE_HEIGHT_M
from speed_estimation.speed_estimator import _build_estimator, _contiguous_runs
from speed_estimation.smoothers import kalman_velocity_1d

TOWARD_MIN_SPEED_MPS = 0.5   # frames slower than this don't vote on direction (sign is noise)


def compute_relevance_flags(world,
                            ego_pos: dict[int, tuple[float, float]],
                            ego_heading: dict[int, float],
                            *,
                            fx: float, fy: float, cx: float, cy: float, fps: float,
                            method: str, lateral_ref: str, lat_sign: int,
                            camera_height_m: float,
                            reject_lateral: bool = Constants.REJECT_BIG_LATERAL,
                            reject_direction: bool = Constants.REJECT_DIRECTION,
                            big_lateral_m: float = Constants.REJECT_LATERAL_M,
                            min_fraction: float = Constants.REJECT_LIFE_FRACTION,
                            toward_min_speed: float = TOWARD_MIN_SPEED_MPS,
                            ) -> dict[int, list[str]]:
    """Return {vehicle_id: [reason, ...]} for vehicles that should be REJECTED.

    `life` for the fractions is the number of frames whose geometry resolves
    (a real bbox the estimator can turn into depth/lateral). Direction needs a
    velocity, so it is measured per contiguous run (>= 2 frames); the dot-product
    SIGN is scale-invariant, so the differentiation timebase (1/fps) does not
    affect it.
    """
    flags_by_id: dict[int, list[str]] = {}

    if not reject_lateral and not reject_direction:
        print("[relevance] both criteria disabled -> no vehicles rejected")
        return flags_by_id

    for vid, vehicle in world.vehicles.items():
        height_m = VEHICLE_HEIGHTS.get(vehicle.vehicle_type, DEFAULT_VEHICLE_HEIGHT_M)
        estimator = _build_estimator(method, fx, fy, cx, cy,
                                     height_m, camera_height_m, lateral_ref)

        valid = sorted(f for f, b in vehicle.bounding_box.items() if b != (0, 0, 0, 0))

        life = 0            # frames with resolvable geometry
        big_lateral = 0     # frames with |lateral| > big_lateral_m
        toward = 0          # frames whose motion points at the ego (and moving)

        for run in _contiguous_runs(valid):
            Tx, Ty, ex_l, ey_l, laterals = [], [], [], [], []
            for f in run:
                r = estimator(vehicle.bounding_box[f])
                if r is None:
                    continue
                depth, lateral = r
                theta = ego_heading.get(f, 0.0)
                ex, ey = ego_pos.get(f, (0.0, 0.0))
                c, s = math.cos(theta), math.sin(theta)
                Tx.append(ex + depth * c + lat_sign * lateral * s)
                Ty.append(ey + depth * s - lat_sign * lateral * c)
                ex_l.append(ex); ey_l.append(ey); laterals.append(lateral)

            n = len(Tx)
            if n == 0:
                continue
            life += n
            if reject_lateral:
                big_lateral += sum(1 for lat in laterals if abs(lat) > big_lateral_m)

            if reject_direction and n >= 2:   # need >= 2 points to have a velocity
                ax, ay = np.array(Tx), np.array(Ty)
                _, vx, _ = kalman_velocity_1d(ax, fps)
                _, vy, _ = kalman_velocity_1d(ay, fps)
                for i in range(n):
                    if math.hypot(vx[i], vy[i]) < toward_min_speed:
                        continue
                    # target -> ego direction; positive dot => moving toward us
                    dx, dy = ex_l[i] - ax[i], ey_l[i] - ay[i]
                    if vx[i] * dx + vy[i] * dy > 0:
                        toward += 1

        if life == 0:
            continue
        tags: list[str] = []
        if reject_lateral and big_lateral / life >= min_fraction:
            tags.append("big_lateral")
        if reject_direction and toward / life >= min_fraction:
            tags.append("direction")
        if tags:
            flags_by_id[vid] = tags

    if flags_by_id:
        summary = ", ".join(f"{vid}:{'+'.join(t)}" for vid, t in sorted(flags_by_id.items()))
        print(f"[relevance] rejecting {len(flags_by_id)} vehicle(s) (off-road/oncoming): {summary}")
    else:
        print("[relevance] no vehicles rejected")
    return flags_by_id
