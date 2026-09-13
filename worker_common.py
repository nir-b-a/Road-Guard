"""
worker_common.py -- the pieces both workers need.

`worker.py` (online: R2 + the REST API) and `worker_offline.py` (local disk only) do the same
work with different transports. Everything that is NOT transport lives here, so the two can
never drift apart:

  * what a session directory looks like, and which mode it implies (android / simulation / video-only)
  * loading the four models ONCE and building a fresh evidence collector per clip
  * turning the pipeline's export bundle into a per-violation folder tree
  * turning that tree into the violation records the backend stores

Nothing here imports `main` at module level: both workers import it lazily so a missing GPU
stack fails at job time, not at start-up.
"""
from __future__ import annotations

import bisect
import json
import os
import re
import shutil
import tarfile
from collections import namedtuple

# The six fixed-name artefacts of the "7-file" session contract (the 7th is the video).
SENSOR_FILES = ("frames.csv", "gps.csv", "gravity.csv", "gyro.csv", "linacc.csv", "intrinsics.json")
# Without these four there is no ego pose, so no world-frame speed and no speeding rule.
REQUIRED_FOR_SPEED = ("frames.csv", "gyro.csv", "gravity.csv", "gps.csv")
VIDEO_EXTS = (".mp4", ".mov", ".m4v", ".mpeg", ".avi", ".mkv")

# A per-violation clip left in place by an earlier run (violations.export.violation_id shape),
# e.g. v100_SPEEDING_f1200.mp4 -- an output, never an input.
VIOLATION_CLIP_RE = re.compile(r"^v\d+_.+_f\d+\.(mp4|mov|m4v)$", re.I)

# Pipeline violation type -> the backend's Violation.violationType enum.
BACKEND_TYPE = {
    "SPEEDING": "speeding",
    "SOLID_LINE_CROSSING": "lane_crossing",
    "YELLOW_LINE_RIGHT": "lane_crossing",
    "TRAFFIC_ISLAND": "lane_crossing",
    "WRONG_WAY": "lane_crossing",
}

# What the pipeline can reconstruct for a given session.
ANDROID = "android"          # video + frames/gyro/gravity/gps -> full pipeline
SIMULATION = "simulation"    # video + telemetry.csv (CARLA)
VIDEO_ONLY = "video-only"    # video alone -> lane rules + plates, but no speed of any kind

Models = namedtuple("Models", "yolo lane tire plate_reader")


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def find_video(session_dir: str) -> str | None:
    """The session's source video: the largest media file directly in the folder, ignoring
    anything a previous run produced (annotated renders, per-violation clips)."""
    best, best_size = None, -1
    for name in sorted(os.listdir(session_dir)):
        path = os.path.join(session_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in VIDEO_EXTS:
            continue
        if (name.endswith("_annotated.mp4") or name == "evidence_5s.mp4"
                or VIOLATION_CLIP_RE.match(name)):
            continue                                     # our own output, not an input
        size = os.path.getsize(path)
        if size > best_size:
            best, best_size = path, size
    return best


def has_sensor_csvs(session_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(session_dir, n)) for n in REQUIRED_FOR_SPEED)


def session_mode(session_dir: str, *, simulation: bool = False) -> str:
    """Which reconstruction the files in `session_dir` allow. `simulation` forces the CARLA path
    (main.py reads telemetry.csv only when it is told the clip is a simulation)."""
    if simulation:
        return SIMULATION
    return ANDROID if has_sensor_csvs(session_dir) else VIDEO_ONLY


def session_files(session_dir: str, video_path: str) -> dict:
    """The Drive.files record, with local paths where the R2 keys would be (None when absent)."""
    files = {"video": video_path}
    for name in SENSOR_FILES:
        candidate = os.path.join(session_dir, name)
        files[os.path.splitext(name)[0]] = candidate if os.path.isfile(candidate) else None
    return files


def describe_mode(mode: str) -> str:
    """One line explaining what this session will and will not produce."""
    if mode == VIDEO_ONLY:
        return ("video only (no sensor CSVs) -> lane-crossing/yellow rules + plate reading run; "
                "NO speed, NO speeding, NO speed plots, and violations carry lat/lon 0")
    if mode == SIMULATION:
        return "CARLA simulation (telemetry.csv) -> full pipeline"
    return "Android session (frames/gyro/gravity/gps) -> full pipeline"


# --------------------------------------------------------------------------- #
# Models -- loaded ONCE per process and reused for every clip
# --------------------------------------------------------------------------- #
def load_models(main, *, yellow: bool = True, stage2: bool = True, evidence: bool = True,
                repo_root: str | None = None, log=print) -> Models:
    """Load the vehicle detector, the lane-seg model, the Stage-2 tire model and the plate
    reader. Mirrors main.main()'s set-up; each optional model degrades to None with a reason."""
    repo_root = repo_root or os.path.dirname(os.path.abspath(__file__))

    log("[models] loading YOLO vehicle model ...")
    yolo = main.loadYoloModel()
    if yolo is None:
        raise RuntimeError("the vehicle detector failed to load")

    lane = None
    if yellow and main._YELLOW_IMPORT_OK:
        weights = main._cli_value("--lane-weights",
                                  os.path.join(repo_root, "weights", "phase3_v3_yellowprotect.pt"))
        if os.path.isfile(weights):
            lane = main.loadLaneModel(weights)
        else:
            log(f"[models] lane weights not found ({weights}); yellow/solid rules disabled")

    tire = None
    if stage2:
        weights = main._cli_value("--tire-weights", os.path.join(repo_root, "models", "tire_yolo11n.pt"))
        if os.path.isfile(weights):
            try:
                from violations.cascade.tire_model import TireModel
                tire = TireModel(weights)
                log(f"[models] Stage-2 tire model loaded: {weights}")
            except Exception as e:
                log(f"[models] tire model failed to load ({e}); Stage-2 disabled")
        else:
            log(f"[models] tire weights not found ({weights}); Stage-2 disabled")

    plate_reader = None
    if evidence and main._EVIDENCE_IMPORT_OK:
        try:
            from lpr.reader import FastALPRReader
            plate_reader = FastALPRReader()
            log("[models] FastALPR plate reader ready")
        except Exception as e:
            log(f"[models] plate reading disabled (reader init failed: {e})")

    return Models(yolo, lane, tire, plate_reader)


def attach_models(main, models: Models) -> None:
    """Point main.py's module globals at this process's models and give the clip a FRESH
    evidence collector: its crop buffers are keyed by track id, and track ids restart at 1 for
    every video, so reusing one across clips would leak crops between them. The OCR reader
    itself (the expensive part) is reused."""
    main.LANE_MODEL, main.TIRE_MODEL = models.lane, models.tire
    if models.plate_reader is None:
        main.EVIDENCE_COLLECTOR = None
        return
    from lpr.evidence import EvidenceCollector
    from Constants import LPR
    main.EVIDENCE_COLLECTOR = EvidenceCollector(models.plate_reader.read_plate_with_conf,
                                                min_area=LPR.MIN_VEHICLE_AREA)


def reset_speed_limit_lookup(log=print) -> None:
    """Re-arm the Overpass (OpenStreetMap) speed-limit lookup before each drive.

    speed_limit_lookup keeps a module-level circuit breaker: after a few failed lookups in a row it
    stops calling Overpass for the rest of the PROCESS. A worker process lives for many drives, so
    without this a short Overpass outage would switch speed limits off for every later drive until
    the worker restarts. The price: while Overpass really is down, each drive spends one more failed
    lookup (every mirror tried, up to ~2.5 min in the worst case) before giving up."""
    try:
        from speed_estimation import speed_limit_lookup
    except ImportError as e:                  # e.g. no `requests`: the speeding stage cannot run anyway
        log(f"[models] speed-limit lookup unavailable ({e})")
        return
    speed_limit_lookup.clear_cache()


def patch_speed_limit(limit_kmh: float, log=print) -> None:
    """Replace the Overpass (OpenStreetMap) lookup with a constant limit.

    The speed-limit query is the ONLY network call in the pipeline. Without it the speeding
    stage is skipped; with this patch the same overspeed logic (margin, peak exceedance,
    per-vehicle reporting) runs against a limit you supply."""
    from speed_estimation import overspeed

    def _fixed(track_points, **_kwargs):
        return {int(tp[0]): limit_kmh for tp in track_points}

    overspeed.get_track_speed_limits = _fixed
    log(f"[models] speed limit pinned to {limit_kmh:.0f} km/h (no Overpass lookup)")


# --------------------------------------------------------------------------- #
# Export bundle -> a per-violation folder tree
# --------------------------------------------------------------------------- #
def unpack_bundle(bundle_path: str, violations_dir: str) -> dict | None:
    """Extract the evidence bundle into violations/<violation_id>/... and return the manifest.

    The per-violation clip is skipped here: the pipeline already wrote it as a loose
    <violation_id>.mp4 next to the bundle, and move_clips() moves that one file in (so the same
    video is not stored twice)."""
    if not os.path.isfile(bundle_path):
        return None
    os.makedirs(violations_dir, exist_ok=True)
    manifest = None
    with tarfile.open(bundle_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = member.name.replace("\\", "/").lstrip("/")
            if ".." in name.split("/"):
                continue                                  # defensive: never escape the folder
            payload = tar.extractfile(member)
            if payload is None:
                continue
            data = payload.read()
            if name == "manifest.json":
                manifest = json.loads(data.decode("utf-8"))
            elif os.path.basename(name).startswith("clip."):
                continue
            dest = os.path.join(violations_dir, *name.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as fh:
                fh.write(data)
    return manifest


def move_clips(manifest: dict, out_dir: str, violations_dir: str) -> dict:
    """Move each loose <violation_id>.mp4 into violations/<violation_id>/clip.mp4.
    -> {violation_id: path relative to out_dir} for the ones that exist."""
    clips = {}
    for rec in (manifest or {}).get("violations", []):
        stem = rec.get("violation_id")
        if not stem:
            continue
        flat = os.path.join(out_dir, f"{stem}.mp4")
        dest_dir = os.path.join(violations_dir, stem)
        dest = os.path.join(dest_dir, "clip.mp4")
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.isfile(flat):
            shutil.move(flat, dest)
        if os.path.isfile(dest):
            clips[stem] = os.path.relpath(dest, out_dir).replace("\\", "/")
    return clips


def collect_speed_plots(out_dir: str, video_name: str) -> dict:
    """Move the per-vehicle speed/distance PNGs and the ego-speed PNG into speed_plots/.
    -> {"vehicles": {track_id: relpath}, "ego": relpath|None}. Both are empty on a video-only
    clip: with no ego pose there are no speeds to plot."""
    plots_dir = os.path.join(out_dir, "speed_plots")
    vehicles, ego = {}, None
    for name in sorted(os.listdir(out_dir)):
        if not name.lower().endswith(".png"):
            continue
        is_vehicle = name.startswith(f"{video_name}_vehicle_")
        is_ego = name == f"{video_name}_ego_speed.png"
        if not (is_vehicle or is_ego):
            continue
        os.makedirs(plots_dir, exist_ok=True)
        dest = os.path.join(plots_dir, name)
        shutil.move(os.path.join(out_dir, name), dest)
        rel = os.path.relpath(dest, out_dir).replace("\\", "/")
        if is_ego:
            ego = rel
        else:
            vehicles[name[len(f"{video_name}_vehicle_"):-len(".png")]] = rel
    return {"vehicles": vehicles, "ego": ego}


# --------------------------------------------------------------------------- #
# Violation records
# --------------------------------------------------------------------------- #
def ego_track_for(session_dir: str, log=print) -> dict:
    """frame -> (lat, lon, bearing) from frames.csv + gps.csv, or {} when there is no GPS."""
    frames_csv = os.path.join(session_dir, "frames.csv")
    gps_csv = os.path.join(session_dir, "gps.csv")
    if not (os.path.isfile(frames_csv) and os.path.isfile(gps_csv)):
        return {}
    try:
        from speed_estimation import overspeed
        return overspeed.build_ego_track(frames_csv, gps_csv)
    except Exception as e:
        log(f"[violations] could not build the ego GPS track ({e}); violations get lat/lon 0")
        return {}


def fix_at(track: dict, sorted_frames: list, frame: int):
    """The GPS fix nearest `frame` (the violation's key frame) -> (lat, lon)."""
    if not sorted_frames:
        return 0.0, 0.0
    i = bisect.bisect_left(sorted_frames, frame)
    if i == 0:
        best = sorted_frames[0]
    elif i >= len(sorted_frames):
        best = sorted_frames[-1]
    else:
        lo, hi = sorted_frames[i - 1], sorted_frames[i]
        best = hi if (hi - frame) < (frame - lo) else lo
    fix = track[best]
    return float(fix[0]), float(fix[1])


def build_violation_payloads(manifest: dict, clips: dict, *, drive_id: str, session_id: str,
                             session_dir: str, detected_at: str, log=print) -> list:
    """One record per violation: the exact body POSTed to /api/internal/violation, what the
    backend adds when it stores the row, and every artefact path.

    `clips` maps violation_id -> wherever the clip now lives (an R2 key online, a path relative
    to the output folder offline). `lat`/`lon` are the EGO's GPS fix at the violation frame --
    the system never has GPS for the other car, the same proxy the speed-limit lookup uses --
    and 0.0 when the session has no gps.csv."""
    track = ego_track_for(session_dir, log=log)
    frames = sorted(track)
    payloads = []

    def _rel(path):
        """Bundle-internal path -> path relative to the output folder (where we unpacked it)."""
        return f"violations/{path}" if path else None

    for rec in (manifest or {}).get("violations", []):
        stem = rec.get("violation_id")
        details = rec.get("details") or {}
        evidence = rec.get("evidence") or {}
        key_frame = int(rec.get("key_frame") or 0)
        lat, lon = fix_at(track, frames, key_frame)
        speed = details.get("est_speed_kmh", details.get("max_speed_kmh", 0)) or 0
        vtype = rec.get("violation") or rec.get("violation_type")
        payloads.append({
            # --- the POST /api/internal/violation body -------------------------------
            "driveId": drive_id,
            "videoClipPath": clips.get(stem),
            "carId": rec.get("plate") or f"vehicle-{rec.get('vehicle_id')}",
            "calculatedSpeed": round(float(speed), 2),
            "lat": lat,
            "lon": lon,
            "violationType": BACKEND_TYPE.get(vtype, "lane_crossing"),
            # --- what the backend adds when it stores the row ------------------------
            "detectedAt": detected_at,
            "status": "pending",
            "reviewedBy": None,
            # --- everything the pipeline knows (kept for the reviewer / the dashboard) -
            "violationId": stem,
            "sessionId": session_id,
            "vehicleId": rec.get("vehicle_id"),
            "pipelineViolationType": vtype,
            "tier": rec.get("tier"),
            "keyFrame": key_frame,
            "detectorConfidence": rec.get("detector_confidence"),
            "crossingSolidLineConfidence": rec.get("crossing_solid_line_confidence"),
            "plate": rec.get("plate"),
            "plateScore": rec.get("plate_score"),
            "plateReads": rec.get("n_reads"),
            "manualReview": rec.get("manual_review"),
            "description": rec.get("description"),
            "details": details,
            # false = the pipeline had no ego pose / no GPS, so the field above is a placeholder
            "speedKnown": "est_speed_kmh" in details or "max_speed_kmh" in details,
            "locationKnown": bool(frames),
            "evidence": {
                "clip": clips.get(stem),
                "images": [_rel(c.get("file")) for c in (evidence.get("crops") or [])],
                "plateImages": [_rel(c["plate"]["file"]) for c in (evidence.get("crops") or [])
                                if c.get("plate")],
                "report": _rel((evidence.get("report") or {}).get("file")),
                "record": _rel(rec.get("record_file")),
                "summary": _rel(rec.get("summary_file")),
            },
        })
    return payloads
