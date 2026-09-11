"""The v3 half-edge contact rule, applied to YELLOW solid lines only, for main.py.

Why this module exists
----------------------
The solid-line crossing rule is split by marking colour:

  * yellow_solid_lane                    -> v3 (this module)
  * solid_white_lane + traffic_island    -> v1, i.e. main.py's `evaluate_crossing`
                                            (ghost_mask verdict timeline -> Stage-2
                                            tire cascade). NOT touched by anything here.

That split is exact, not approximate: `ghost_mask.VIOLATION_CLASSES` is literally
{"solid_white_lane", "traffic_island"}, so v1 already owns the white lines and the
painted gore areas and has never looked at a yellow line. This module adds the
colour v1 skips; it removes nothing from v1.

What it guarantees for the rest of the program
----------------------------------------------
A crossing found here is emitted as the SAME `ViolationEvent`
(`ViolationType.SOLID_LINE_CROSSING`, same `details` keys) that `evaluate_crossing`
emits, so every downstream stage - evidence/LPR, the violations CSV, the annotated
video, the clip export and the backend/R2 push - handles it without knowing which
rule fired. Nothing here opens a socket, writes outside the caller's control or
deletes anything: offline runs stay offline, and an online run pushes a yellow
crossing through exactly the path a white one already takes.

Two behaviours are deliberately matched to v1 rather than to the standalone runner:

  * the same per-vehicle cooldown collapse `motion_filter.merge_events` applies to
    the v1 incidents (one reportable incident per vehicle per `cooldown_sec`), so
    the yellow channel does not report at a different granularity from the white one;
  * `drop_duplicate_crossings`, so a vehicle that trips both channels inside that
    same window still produces ONE violation, as it does with v1 alone.

Line continuation is OFF (`--no-extrap`): the lane geometry stops where the model
saw paint stop, and is never extended past it.
"""
from __future__ import annotations

from dataclasses import replace

from .config import CrossingConfig
from .detector import to_violation_events
from .edge_contact import EdgeContactConfig, detect_edge_contacts
from .lane_tracks import contour_to_lane
from .types import FrameObservation, VehicleBox

# The lane-seg head's class names -> the lane_types contract. Same map
# tools/run_crossing2_video.py uses; a traffic island is an AREA, not a line, and
# belongs to v1 either way, so it is not in here.
SEG_TO_LANETYPE = {
    "solid_white_lane": "solid_white",
    "yellow_solid_lane": "solid_yellow",
    "dashed_lane": "dashed",
}

# The only marking type this rule may fire on. Everything else is v1's.
YELLOW_LANE_TYPES: tuple[str, ...] = ("solid_yellow",)

DEFAULT_COOLDOWN_SEC = 3.0        # matches evaluate_crossing's merge_events call


def yellow_config() -> CrossingConfig:
    """Stage 0/1 tunables for this rule: stock, except that the line CONTINUATION is
    off - the two places crossing2 extends a marking past the paint the model actually
    segmented (Stage 0's per-frame row fill, Stage 2's bottom-tangent reach). Interior
    gaps between two real observations are still bridged; that is interpolation, not a
    guess about where the marking goes next."""
    cfg = CrossingConfig()
    cfg.row_fill_extrap_rows = 0
    cfg.extrapolate_frac = 0.0
    return cfg


def yellow_ec_config() -> EdgeContactConfig:
    """The v3 rule exactly as it runs in the standalone runner's defaults, restricted
    to yellow markings."""
    return EdgeContactConfig(lane_types=YELLOW_LANE_TYPES)


def records_to_lanes(records, frame_width: int) -> list:
    """One frame of `seg_lanes` records -> `lane_types.Lane` centerlines.

    EVERY lane class is converted, not just yellow: Stage 0 derives the lane-width
    ruler, the association gates and the double-line suppression from the whole set
    of markings in the scene. The yellow-only restriction is applied later, on which
    tracks may fire (`EdgeContactConfig.lane_types`), so those three stay identical
    to a full-scene run.
    """
    lanes = []
    for r in records or ():
        lt = SEG_TO_LANETYPE.get(r.get("cls"))
        if lt is None:
            continue
        lane = contour_to_lane(r.get("contour"), frame_width,
                               lane_type=lt, score=float(r.get("conf", 1.0)))
        if lane is not None:
            lanes.append(lane)
    return lanes


def observations_from_seg_frames(seg_frames: list, frame_width: int) -> list[FrameObservation]:
    """main.py's live lane cache -> the crossing2 input contract.

    `seg_frames` records are `{"frame", "vehicles": [{"track_id", "bbox"}], "lanes",
    "shift"}`. The optical-flow `shift` is not used: v3 compares a bbox against a
    marking in the SAME frame, so ego motion moves both together and cancels.
    """
    obs: list[FrameObservation] = []
    for fr in seg_frames:
        vehicles = [VehicleBox(int(v["track_id"]), tuple(int(c) for c in v["bbox"]))
                    for v in fr.get("vehicles", ()) if v.get("bbox") is not None]
        obs.append(FrameObservation(int(fr["frame"]),
                                    records_to_lanes(fr.get("lanes"), frame_width),
                                    vehicles))
    return obs


def _collapse(events: list, fps: float, cooldown_sec: float) -> list:
    """Per-vehicle cooldown collapse, the same one `motion_filter.merge_events`
    applies to the v1 incidents: contacts from one vehicle less than `cooldown_sec`
    apart are one incident. The representative kept is the frame where the largest
    share of the vehicle was across the line."""
    gap = cooldown_sec * max(fps, 1e-6)
    runs: dict[int, list[list]] = {}
    for e in sorted(events, key=lambda e: (e.vehicle_id, e.start_frame)):
        group = runs.setdefault(e.vehicle_id, [])
        if group and e.start_frame - group[-1][-1].end_frame <= gap:
            group[-1].append(e)
        else:
            group.append([e])

    out = []
    for group in runs.values():
        for run in group:
            rep = max(run, key=lambda e: e.peak_overlap)
            start = min(e.start_frame for e in run)
            end = max(e.end_frame for e in run)
            out.append(replace(
                rep, start_frame=start, end_frame=end,
                confidence=max(e.confidence for e in run),
                details={**rep.details, "n_frames": sum(e.details.get("n_frames", 0) for e in run),
                         "merged_contacts": len(run)}))
    out.sort(key=lambda e: (e.start_frame, e.vehicle_id))
    return out


def evaluate_yellow_crossing(seg_frames: list, fps: float, frame_height: int,
                             frame_width: int, *,
                             cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
                             config: CrossingConfig | None = None,
                             ec_config: EdgeContactConfig | None = None,
                             stats: dict | None = None) -> list:
    """Yellow solid-line crossings over a whole clip -> `ViolationEvent`s.

    Argument order matches main.py's `evaluate_crossing(seg_frames, fps, frame_height,
    frame_width, ...)` on purpose, so the two calls cannot be mixed up at the call site.
    Returns [] - never raises - when there is nothing to work with.
    """
    if not seg_frames or frame_width <= 0 or frame_height <= 0:
        return []

    obs = observations_from_seg_frames(seg_frames, frame_width)
    result = detect_edge_contacts(obs, frame_width, frame_height, fps,
                                  config or yellow_config(),
                                  ec_config or yellow_ec_config(), stats)
    raw = [e for e in result.events if e.kind == "crossing"]
    merged = _collapse(raw, fps, cooldown_sec)
    events = to_violation_events(merged)
    for ve in events:
        ve.details["rule"] = "v3_half_edge_contact"
        ve.details["marking"] = "yellow_solid"
    print(f"[crossing/yellow] v3 on yellow lines: {len(raw)} contact run(s) -> "
          f"{len(events)} event(s)")
    return events


def drop_duplicate_crossings(primary: list, extra: list, fps: float,
                             cooldown_sec: float = DEFAULT_COOLDOWN_SEC) -> list:
    """Remove `extra` events whose vehicle already has a `primary` event within the
    same cooldown window.

    Without this, a vehicle that cuts a yellow line and a white one within a few
    frames would produce two SOLID_LINE_CROSSING records - two evidence bundles, two
    uploads - where v1 on its own produces one. Only the cross-channel duplicate is
    dropped; a genuinely separate incident later in the clip survives.
    """
    if not (primary and extra):
        return list(extra)
    gap = cooldown_sec * max(fps, 1e-6)
    spans: dict[int, list[tuple[float, float]]] = {}
    for e in primary:
        s = float(e.details.get("start_frame", e.key_frame))
        t = float(e.details.get("end_frame", e.key_frame))
        spans.setdefault(e.vehicle_id, []).append((s, t))

    kept = []
    for e in extra:
        s = float(e.details.get("start_frame", e.key_frame))
        t = float(e.details.get("end_frame", e.key_frame))
        if any(s - b <= gap and a - t <= gap for a, b in spans.get(e.vehicle_id, ())):
            continue
        kept.append(e)
    if len(kept) != len(extra):
        print(f"[crossing/yellow] {len(extra) - len(kept)} event(s) dropped as the same "
              f"incident v1 already reported for that vehicle")
    return kept
