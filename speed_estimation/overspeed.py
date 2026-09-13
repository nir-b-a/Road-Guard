"""
overspeed -- flag tracked vehicles whose ESTIMATED speed exceeds the road limit.

Where the numbers come from
---------------------------
* Estimated per-vehicle speed: ``vehicle.speed_per_frame`` (a {frame: speed} dict),
  produced by the speed estimator (``speed_estimator.estimate_world_speeds``) and stored
  on each Vehicle. Those values are in **m/s** (the smoothers return m/s), so we convert x3.6 to
  match ``speed_limit_lookup`` which is in **km/h**. (On the backend the same quantity
  is persisted as ``Drive.speedSamples[].speedKmh`` -- already km/h, no conversion.)
* Road limit: ``speed_limit_lookup.get_speed_limit(lat, lon, bearing)`` (km/h).

The limit-location proxy
------------------------
We only have GPS for the EGO vehicle, not for the other cars the dashcam tracks. The
tracked vehicles share the ego's road, so the limit at the ego's position for a frame is
the best available limit for a vehicle seen in that frame. ``ego_track`` therefore maps
frame -> (lat, lon[, bearing]); use :func:`build_ego_track` to build it from an Android
clip's frames.csv + gps.csv.

Estimation is approximate (monocular geometry tends to OVER-estimate), so a vehicle is
only flagged when it is over the limit by more than ``OVERSPEED_MARGIN_KMH``, and we
report the (conservative) peak exceedance per vehicle, not every noisy frame.
"""

import csv
import math
from dataclasses import dataclass

from speed_estimation.speed_limit_lookup import get_track_speed_limits

MPS_TO_KMH = 3.6

# The estimator over-estimates, so require this much OVER the limit before flagging --
# below it the "violation" is within the noise of the estimate. Tune on real footage.
OVERSPEED_MARGIN_KMH = 10.0

# The GPS heading only means something while the car moves: Android writes bearing 0.0 (due north)
# when it has none, and near a standstill the heading is noise. A wrong heading makes the
# direction-aware road match prefer the opposite carriageway, so below this GPS speed the bearing is
# dropped and the match falls back to the nearest road. (gps.csv without speed_mps keeps the bearing.)
MIN_BEARING_SPEED_MPS = 3.0     # ~11 km/h


@dataclass
class OverspeedEvent:
    """One flagged vehicle, reported at the frame of its PEAK exceedance."""
    vehicle_id: int
    frame: int
    lat: float
    lon: float
    est_speed_kmh: float      # estimated speed at that frame (km/h)
    speed_limit_kmh: int      # road limit there (km/h)
    over_by_kmh: float        # est_speed - limit  (> margin)
    n_frames_over: int        # how many frames this vehicle was over the limit+margin


def build_ego_track(frames_csv: str, gps_csv: str) -> dict[int, tuple[float, float, float | None]]:
    """frame -> (lat, lon, bearing) using the GPS fix nearest each frame's timestamp.

    Reuses ego_yaw.load_frame_timestamps for the frame clock. Nearest-fix (not
    interpolation) keeps a real (lat, lon, bearing) triple -- good enough since the
    speed-limit lookup already rounds coordinates to ~11 m.
    """
    from speed_estimation import ego_yaw   # local import: avoids a heavy import at module load

    frame_ts = ego_yaw.load_frame_timestamps(frames_csv)
    if not frame_ts:
        return {}

    fixes: list[tuple[int, float, float, float | None]] = []
    with open(gps_csv, "r") as f:
        for row in csv.DictReader(f):
            try:
                ts_ns, lat, lon = int(row["timestamp_ns"]), float(row["lat"]), float(row["lon"])
                bearing = float(row["bearing_deg"])
            except (KeyError, ValueError):
                continue
            try:
                speed = float(row.get("speed_mps") or "nan")
            except ValueError:
                speed = math.nan
            if not math.isfinite(bearing) or (math.isfinite(speed) and speed < MIN_BEARING_SPEED_MPS):
                bearing = None          # no usable heading -> nearest-road match for this fix
            fixes.append((ts_ns, lat, lon, bearing))
    if not fixes:
        return {}
    fixes.sort(key=lambda r: r[0])
    ts = [r[0] for r in fixes]

    import bisect
    out: dict[int, tuple[float, float, float | None]] = {}
    for frame, t in frame_ts.items():
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            j = 0
        elif i >= len(ts):
            j = len(ts) - 1
        else:                                   # pick the closer of the two neighbours
            j = i if (ts[i] - t) < (t - ts[i - 1]) else i - 1
        _, lat, lon, brg = fixes[j]
        out[frame] = (lat, lon, brg)
    return out


def flag_overspeed_vehicles(world,
                            ego_track: dict[int, tuple],
                            *,
                            margin_kmh: float = OVERSPEED_MARGIN_KMH,
                            min_frames_over: int = 1,
                            **lookup_kwargs) -> list[OverspeedEvent]:
    """Flag vehicles whose estimated speed exceeds the road limit (+ margin).

    Args:
        world:           a World whose vehicles carry ``speed_per_frame`` (m/s), i.e.
                         after estimate_world_speeds has run.
        ego_track:       frame -> (lat, lon) or (lat, lon, bearing); see build_ego_track.
        margin_kmh:      how far OVER the limit (km/h) a vehicle must be to be flagged,
                         absorbing the estimator's tendency to over-estimate.
        min_frames_over: ignore one-off spikes -- require this many over-limit frames.
        lookup_kwargs:   forwarded to get_track_speed_limits (radius_m, timeout,
                         bbox_margin_deg).

    Returns:
        One OverspeedEvent per offending vehicle (at its peak exceedance frame), sorted
        worst-first. Every frame's road limit is resolved up front from a SINGLE Overpass
        bbox query (get_track_speed_limits) -- not one request per frame -- so this stays
        fast and can't hang on a slow public API.
    """
    # Resolve the limit at every ego frame in one shot (one network query for the
    # whole track), then it's just a dict lookup per (vehicle, frame).
    track_points = [(frame, fix[0], fix[1], fix[2] if len(fix) > 2 else None)
                    for frame, fix in ego_track.items()]
    limit_by_frame = get_track_speed_limits(track_points, **lookup_kwargs)

    def limit_at(frame: int):
        return limit_by_frame.get(frame)

    events: list[OverspeedEvent] = []
    for vid, vehicle in world.vehicles.items():
        best = None                 # (frame, over_by, est_kmh, limit)
        n_over = 0
        for frame, sp_mps in vehicle.speed_per_frame.items():
            if sp_mps is None or sp_mps <= 0:
                continue
            limit = limit_at(frame)
            if limit is None:
                continue
            est_kmh = sp_mps * MPS_TO_KMH
            over = est_kmh - limit
            if over > margin_kmh:
                n_over += 1
                if best is None or over > best[1]:
                    best = (frame, over, est_kmh, limit)

        if best is not None and n_over >= min_frames_over:
            frame, over, est_kmh, limit = best
            lat, lon = ego_track[frame][0], ego_track[frame][1]
            events.append(OverspeedEvent(
                vehicle_id=vid, frame=frame, lat=lat, lon=lon,
                est_speed_kmh=round(est_kmh, 1), speed_limit_kmh=limit,
                over_by_kmh=round(over, 1), n_frames_over=n_over))

    events.sort(key=lambda e: e.over_by_kmh, reverse=True)
    return events


@dataclass
class SpeedingEvent:
    """One speeding EPISODE for one vehicle against a FIXED posted limit (the offline-baseline
    path -- no GPS limit lookup). Reported ONCE at the onset frame; carries the episode bounds and
    the peak speed reached inside that limit zone."""
    vehicle_id: int
    onset_frame: int          # the frame the offence began -> the "report once" key frame
    start_frame: int          # episode bounds (== onset_frame, kept explicit for the report)
    end_frame: int
    est_speed_kmh: float      # speed at the onset frame
    max_speed_kmh: float      # PEAK speed reached during the episode (within the limit zone)
    speed_limit_kmh: float
    over_by_kmh: float        # max_speed - limit
    n_frames_over: int


def flag_speeding_fixed_limit(world,
                              *,
                              limit_kmh: float,
                              threshold_kmh: float,
                              fps: float,
                              cooldown_sec: float = 30.0,
                              gap_close_sec: float = 0.5,
                              min_frames_over: int = 1) -> list[SpeedingEvent]:
    """Flag speeding against a FIXED limit, using each vehicle's ``speed_per_frame`` (m/s).

    Unlike :func:`flag_overspeed_vehicles` (which needs a per-frame GPS-derived limit), this uses
    one posted ``limit_kmh`` for the whole clip -- the offline simulation/baseline scenario.

    Logic per vehicle (Tal's spec):
      * an "over" frame is one whose speed >= ``threshold_kmh`` (= limit * 1.1, the violation line),
      * consecutive over-frames form an EPISODE; a gap of up to ``gap_close_sec`` is bridged so a
        one-frame dip doesn't split one offence into two,
      * the episode is REPORTED ONCE at its onset frame (the red-box trigger moment),
      * ``max_speed_kmh`` is the PEAK speed over the whole episode (the value the data report keeps
        "within that specific speed limit zone"),
      * a 30 s per-vehicle COOLDOWN (``cooldown_sec``) suppresses a fresh report that starts within
        that window of the previous episode's onset.

    Returns one SpeedingEvent per reported episode, sorted by onset frame.
    """
    gap = max(0, round(gap_close_sec * fps))
    cooldown = cooldown_sec * fps
    events: list[SpeedingEvent] = []

    for vid, vehicle in world.vehicles.items():
        # frames where this vehicle is over the violation line, in time order
        over_frames = sorted(f for f, sp in vehicle.speed_per_frame.items()
                             if sp is not None and sp * MPS_TO_KMH >= threshold_kmh)
        if not over_frames:
            continue

        # group into episodes, bridging gaps <= gap frames
        episodes: list[list[int]] = []
        cur = [over_frames[0]]
        for f in over_frames[1:]:
            if f - cur[-1] <= gap + 1:
                cur.append(f)
            else:
                episodes.append(cur)
                cur = [f]
        episodes.append(cur)

        last_onset = None
        for ep in episodes:
            start, end = ep[0], ep[-1]
            if last_onset is not None and (start - last_onset) <= cooldown:
                continue                       # within the per-vehicle cooldown -> not re-reported
            last_onset = start
            # peak speed across the CONTIGUOUS span of the episode (not just the over-frames),
            # so the recorded max reflects everything the car did during the offence window.
            span_speeds = [vehicle.speed_per_frame[f] * MPS_TO_KMH
                           for f in range(start, end + 1)
                           if f in vehicle.speed_per_frame and vehicle.speed_per_frame[f] is not None]
            max_kmh = max(span_speeds) if span_speeds else threshold_kmh
            onset_kmh = vehicle.speed_per_frame[start] * MPS_TO_KMH
            if len(ep) < min_frames_over:
                continue
            events.append(SpeedingEvent(
                vehicle_id=vid, onset_frame=start, start_frame=start, end_frame=end,
                est_speed_kmh=round(onset_kmh, 1), max_speed_kmh=round(max_kmh, 1),
                speed_limit_kmh=round(limit_kmh, 1),
                over_by_kmh=round(max_kmh - limit_kmh, 1), n_frames_over=len(ep)))

    events.sort(key=lambda e: e.onset_frame)
    return events


def format_overspeed_report(events: list[OverspeedEvent]) -> str:
    """Human-readable summary: which vehicle, where, by how much (km/h)."""
    if not events:
        return "No vehicles exceeded the speed limit (beyond the estimation margin)."
    lines = [f"{len(events)} vehicle(s) over the limit (est. speeds; may over-estimate):"]
    for e in events:
        lines.append(
            f"  vehicle {e.vehicle_id}: ~{e.est_speed_kmh:.0f} km/h in a "
            f"{e.speed_limit_kmh} km/h zone  (+{e.over_by_kmh:.0f} km/h) "
            f"at frame {e.frame} ~({e.lat:.5f}, {e.lon:.5f}), {e.n_frames_over} frame(s) over")
    return "\n".join(lines)


def write_overspeed_csv(events: list[OverspeedEvent], path: str) -> None:
    """Write one row per flagged vehicle (machine-readable companion to the report).

    Always writes the header, so an empty file unambiguously means "no vehicle was
    over the limit" rather than "the step didn't run".
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vehicle_id", "frame", "lat", "lon",
                         "est_speed_kmh", "speed_limit_kmh", "over_by_kmh", "n_frames_over"])
        for e in events:
            writer.writerow([e.vehicle_id, e.frame, f"{e.lat:.6f}", f"{e.lon:.6f}",
                             e.est_speed_kmh, e.speed_limit_kmh, e.over_by_kmh, e.n_frames_over])


# ============================================================================
# Offline self-test (no network: get_speed_limit is monkeypatched).
# ============================================================================

if __name__ == "__main__":
    import speed_estimation.overspeed as me

    # Fake a World with two vehicles' speed_per_frame (m/s).
    class _V:
        def __init__(self, spf): self.speed_per_frame = spf
    class _W:
        def __init__(self, vehicles): self.vehicles = vehicles

    speeder = _V({0: 25.0, 1: 26.0, 2: 27.0})   # ~90-97 km/h
    cruiser = _V({0: 13.0, 1: 13.5})            # ~47-49 km/h

    world = _W({101: speeder, 102: cruiser})
    ego_track = {0: (32.0, 34.9, 90.0), 1: (32.0001, 34.9, 90.0), 2: (32.0002, 34.9, 90.0)}

    # 50 km/h zone for every frame, offline (no Overpass).
    me.get_track_speed_limits = lambda points, **kw: {p[0]: 50 for p in points}

    evts = me.flag_overspeed_vehicles(world, ego_track, margin_kmh=10.0)
    print(format_overspeed_report(evts))
    assert len(evts) == 1 and evts[0].vehicle_id == 101, evts   # cruiser stays under 50+10
    print("self-test OK")
