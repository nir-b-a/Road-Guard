import sys
import os
import time
import json
import torch
import cv2
import numpy as np
import speed_estimation.botsort_patch
import cloud_env                          # headless/Colab detection + smart Drive I/O
from video_handler import VideoHandler
from ultralytics import YOLO
from ultralytics.engine.results import Results
from Constants import LPR
import Constants
#from lpr.reader import PaddleOCRDetectorReader, try_read_plate
from Objects.World import World
from speed_estimation.speed_estimator import (estimateDistance, smooth_distances,
                             estimate_world_speeds, focal_length_from_fov)
from speed_estimation import distanceLogger
from speed_estimation.bbox_interpolator import interpolate_bbox_gaps
from speed_estimation import ego_yaw
from speed_estimation import draw_vehicle_plots
from speed_estimation import annotated_video
from speed_estimation import overspeed
from speed_estimation import relevance_flags
# Clean-room ground-plane distance estimator (separate package). Replaces the old
# estimateDistance/smooth_distances for the distance step; main.py stays the runner
# and the YOLO/ByteTrack loop + Vehicle/World population are unchanged.
#from ground_distance import estimate_ground_distances

# ── Violation -> evidence (LPR + best pictures) ──────────────────────────────
# ViolationEvent is the generic record every detector emits; EvidenceCollector reads
# the plate + best crops from an in-memory buffer (no video re-read). FastALPR is the
# only OCR backend installed in roadguard-dl. These imports are wrapped so the core
# tracking/speed pipeline still runs if fast_alpr is missing.
from violations.event import ViolationEvent, ViolationType
try:
    from lpr.reader import FastALPRReader
    from lpr.evidence import EvidenceCollector
    _EVIDENCE_IMPORT_OK = True
except Exception as _e:                       # pragma: no cover
    print(f"[evidence] import unavailable ({_e}); evidence stage will be skipped")
    _EVIDENCE_IMPORT_OK = False

# ── Yellow-line (shoulder) violation detector ────────────────────────────────
# It lives under tools/ and consumes a per-frame lane-seg cache; we build that cache
# live in the frame loop (a 2nd, segmentation model + optical-flow ego shift). Wrapped
# so a missing tools chain just disables the yellow-line rule, not the whole run.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)
try:
    from ghost_mask import (estimate_ego_shift,        # optical-flow ego shift (same as the harness)
                            compute_verdict_timeline,  # solid-line crossing verdict timeline
                            events_from_timeline)
    from motion_filter import merge_events             # per-vehicle cooldown collapse
    import shoulder_violation                          # yellow-line evaluator (emits ViolationEvent)
    _YELLOW_IMPORT_OK = True
except Exception as _e:                                # pragma: no cover
    print(f"[yellow] import unavailable ({_e}); yellow-line rule will be skipped")
    _YELLOW_IMPORT_OK = False

# ── Backend export (per-violation clip + 3 pics + .docx -> .tar.gz -> backend) ─
try:
    from violations import export as vx, docx_report
    from violations.clip_extract import clip_window
    from violations.annotate_clip import annotate_clip
    _EXPORT_IMPORT_OK = True
except Exception as _e:                                # pragma: no cover
    print(f"[export] import unavailable ({_e}); --push-url/--job-payload export will be skipped")
    _EXPORT_IMPORT_OK = False


DEVICE = 0 if torch.cuda.is_available() else 'cpu'
USE_HALF = torch.cuda.is_available()   # FP16 inference - faster on modern NVIDIA GPUs

YOLO_MODEL = None
CLASSES = Constants.DETECTION_CLASSES
CONFIDENCE_LVL = Constants.CONFIDENCE_LVL

SMOOTHER = Constants.DEFAULT_SMOOTHER

# Per-vehicle rolling top-K sharpest-crop buffer + violation-time plate read.
# Created in main(); referenced as a module global from processFrame's live hook.
EVIDENCE_COLLECTOR = None
# YOLOv8-seg lane model (yellow line). Created in main() if the yellow rule is enabled.
LANE_MODEL = None
# Stage-2 tire detector (solid-line crossing confirmation). Created in main() if weights exist.
TIRE_MODEL = None


def print_time(start_time, read_times, yolo_times, postprocess_times, frame_id):
    total_time = time.time() - start_time
    avg_read      = sum(read_times)      / len(read_times)      * 1000
    avg_yolo      = sum(yolo_times)      / len(yolo_times)      * 1000
    avg_postproc  = sum(postprocess_times) / len(postprocess_times) * 1000

    print(f"\n--- timing report ({frame_id} frames) ---")
    print(f"  avg frame read  : {avg_read:.2f} ms")
    print(f"  avg yolo        : {avg_yolo:.2f} ms")
    print(f"  avg postprocess : {avg_postproc:.2f} ms")
    print(f"  total           : {total_time:.2f} s")


def _cli_value(flag: str, default):
    """Return the argv token after `flag`, or `default` if the flag is absent.

    Lets a run override a Constant without editing Constants.py -- used for the
    handedness signs (--lat-sign / --heading-sign) that must be re-validated on
    real footage, for --smoother, and for the A/B knobs (--model / --imgsz / --out-dir).
    """
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


# load yolo vehicle model. --model overrides Constants.YOLO_VERSION (enables the v8m/11x A/B).
# task="detect" is required for exported engines/onnx -- they don't carry task metadata.
# .to() only works on .pt models; an exported .engine is already bound to its build device.
def loadYoloModel():
    weights = _cli_value("--model", Constants.YOLO_VERSION)
    is_pt = str(weights).endswith(".pt")
    print(f"Loading {weights} on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    try:
        model = YOLO(weights, task="detect")
        if is_pt:
            model.to(DEVICE)
        return model
    except Exception as e:
        print(f"Error occurred: {e}")


def loadLaneModel(weights: str):
    """YOLOv8-seg lane model for the yellow-line rule, or None if it can't be loaded."""
    try:
        m = YOLO(weights, task="segment")
        print(f"[yellow] lane-seg model {os.path.basename(weights)} classes={m.names}")
        return m
    except Exception as e:
        print(f"[yellow] could not load lane model {weights}: {e}; yellow-line disabled")
        return None


def seg_lanes(lane_model, frame, conf: float) -> list:
    """Run lane segmentation on one frame -> the SAME lane-record format the offline harness
    (crossing_violation_test.build_cache) produces, so shoulder_violation can consume it."""
    lanes = []
    lres = lane_model.predict(frame, conf=conf, verbose=False)[0]
    if lres.masks is not None and lres.boxes is not None and len(lres.boxes):
        for poly, c, cf, b in zip(lres.masks.xy, lres.boxes.cls.tolist(),
                                  lres.boxes.conf.tolist(), lres.boxes.xyxy.tolist()):
            poly = np.asarray(poly, dtype=np.float32)
            if len(poly) < 3:
                continue
            cnt = poly.round().astype(np.int32).reshape(-1, 1, 2)
            approx = cv2.approxPolyDP(cnt, epsilon=1.5, closed=True).reshape(-1, 2)
            lanes.append({"cls": lres.names[int(c)], "conf": round(float(cf), 3),
                          "bbox": [int(round(x)) for x in b], "contour": approx.tolist()})
    return lanes


def _shift_frame_dict(d, shift: int):
    """Re-key a frame->value dict by `shift` frames (sensor<->camera time-sync test).

    Returns a NEW dict where camera frame `f` carries the value that was originally
    at frame `f + shift`. So `shift=+30` pairs each video frame with the ego sample
    30 frames LATER (the ego trajectory is pulled earlier/"back" relative to the
    video); `shift=-30` pairs it with the sample 30 frames earlier. The original
    frame keys are preserved (full coverage), and out-of-range lookups CLAMP to the
    nearest real sample instead of falling back to 0 -- a (0,0) ego pose at the
    boundary would inject a fake jump and spike the differentiated speed.
    Pure relabel: bbox dict, distance path, and intrinsics are untouched.
    """
    if not d or shift == 0:
        return d
    keys = sorted(d.keys())
    lo, hi = keys[0], keys[-1]
    out = {}
    for f in keys:
        src = f + shift
        if src < lo:
            src = lo
        elif src > hi:
            src = hi
        out[f] = d[src]
    return out


def run_speed_estimation(world: World, fps: float,
                         fx: float, fy: float, cx: float, cy: float,
                         *,
                         is_simulation: bool, is_android: bool,
                         telemetry_csv: str, frames_csv: str,
                         gyro_csv: str, gravity_csv: str, gps_csv: str,
                         linacc_csv: str = "",
                         image_width: int = 0, image_height: int = 0,
                         valid_roi: tuple[int, int, int, int] | None = None):
    smoother     = _cli_value("--smoother", SMOOTHER)
    lat_sign     = int(_cli_value("--lat-sign", Constants.LAT_SIGN))
    heading_sign = int(_cli_value("--heading-sign", Constants.HEADING_SIGN))
    lateral_ref  = _cli_value("--lateral-ref", Constants.LATERAL_REF)
    wide_angle_reweight = str(_cli_value(
        "--wide-angle-reweight", Constants.WIDE_ANGLE_REWEIGHT)).lower() in ("1", "true", "yes", "on")
    edge_gate = str(_cli_value(
        "--edge-gate", Constants.EDGE_GATE)).lower() in ("1", "true", "yes", "on")
    # Frame-edge DROP: IGNORE (for speed only) frames whose bbox touches the border.
    # Stronger than --edge-gate. Off by default; pass `--edge-drop 1` to enable.
    edge_drop = str(_cli_value(
        "--edge-drop", Constants.EDGE_DROP)).lower() in ("1", "true", "yes", "on")
    # Shrink-rate DROP: IGNORE (for speed only) frames whose bbox collapses fast.
    # Off by default; pass `--shrink-drop 1` to enable.
    shrink_drop = str(_cli_value(
        "--shrink-drop", Constants.SHRINK_DROP)).lower() in ("1", "true", "yes", "on")
    # Occlusion down-weighting: trust frames LESS where the vehicle is hidden behind a
    # nearer one. ON by default; pass `--occlusion-gate 0` to disable.
    occlusion_gate = str(_cli_value(
        "--occlusion-gate", Constants.OCCLUSION_GATE)).lower() in ("1", "true", "yes", "on")
    # Aspect-ratio gate: trust frames LESS where the bbox width/height ratio jumps abruptly.
    # ON by default; pass `--aspect-gate 0` to disable.
    aspect_gate = str(_cli_value(
        "--aspect-gate", Constants.ASPECT_GATE)).lower() in ("1", "true", "yes", "on")
    # Relevance rejection, two INDEPENDENT criteria. Each ON by default.
    reject_lateral = str(_cli_value(
        "--reject-lateral", Constants.REJECT_BIG_LATERAL)).lower() in ("1", "true", "yes", "on")
    reject_direction = str(_cli_value(
        "--reject-direction", Constants.REJECT_DIRECTION)).lower() in ("1", "true", "yes", "on")
    # Far-distance DROP: ignore frames whose estimated depth is beyond max_speed_distance.
    # ON by default; pass `--distance-drop 0` to disable.
    distance_drop = str(_cli_value(
        "--distance-drop", Constants.DISTANCE_DROP)).lower() in ("1", "true", "yes", "on")
    max_speed_distance = float(_cli_value("--max-speed-distance", Constants.MAX_SPEED_DISTANCE_M))
    # Ego<->camera time-sync test: shift the ego pose by N frames vs the camera bboxes.
    ego_shift = int(_cli_value("--ego-shift", 0))

    ego_pos = ego_heading = None
    if is_simulation and os.path.exists(telemetry_csv):
        ego_pos     = ego_yaw.ego_position_from_telemetry(telemetry_csv)
        ego_heading = ego_yaw.ego_heading_from_telemetry(telemetry_csv)
    elif is_android:
        ego_heading = ego_yaw.ego_heading_from_android(
            frames_csv, gyro_csv, gravity_csv, sign=heading_sign)
        ego_pos     = ego_yaw.ego_position_from_android(
            frames_csv, gps_csv, ego_heading,
            linacc_csv=linacc_csv, gravity_csv=gravity_csv)

    if not (ego_pos and ego_heading):
        print("[main] no ego pose available -> skipping world-frame speed path "
              "(needs --simulation + telemetry.csv, or an Android clip with "
              "frames.csv + gyro.csv + gravity.csv + gps.csv)")
        return

    # Ego<->camera time-sync test (--ego-shift N): slide the sensor-clock pose by N frames.
    if ego_shift:
        ego_pos     = _shift_frame_dict(ego_pos, ego_shift)
        ego_heading = _shift_frame_dict(ego_heading, ego_shift)

    print(f"[main] world-frame speed: smoother={smoother}, "
          f"lat_sign={lat_sign}, heading_sign={heading_sign}, lateral_ref={lateral_ref}, "
          f"wide_angle_reweight={wide_angle_reweight}, edge_gate={edge_gate}, "
          f"edge_drop={edge_drop}, shrink_drop={shrink_drop}, "
          f"occlusion_gate={occlusion_gate}, aspect_gate={aspect_gate}, "
          f"reject_lateral={reject_lateral}, reject_direction={reject_direction}, "
          f"distance_drop={distance_drop} (>= {max_speed_distance:.0f}m), "
          f"valid_roi={valid_roi}, ego_shift={ego_shift}")
    frame_ts = ego_yaw.load_frame_timestamps(frames_csv) if is_android else None
    # ── RELEVANCE REJECTION: decide which vehicles are probably NOT on our road
    # (persistently big lateral offset, or oncoming) and SKIP them in estimate_world_speeds.
    reject_ids = set(relevance_flags.compute_relevance_flags(
        world, ego_pos, ego_heading,
        fx=fx, fy=fy, cx=cx, cy=cy, fps=fps,
        method=Constants.DISTANCE_CALCULATION_METHOD,
        lateral_ref=lateral_ref, lat_sign=lat_sign,
        camera_height_m=Constants.DEFAULT_CAMERA_HEIGHT_M,
        reject_lateral=reject_lateral, reject_direction=reject_direction))

    estimate_world_speeds(
        world, ego_pos, ego_heading, fps,
        fx=fx, fy=fy, cx=cx, cy=cy,
        frame_ts=frame_ts,
        method=Constants.DISTANCE_CALCULATION_METHOD,
        smoother=smoother,
        lat_sign=lat_sign,
        lateral_ref=lateral_ref,
        wide_angle_reweight=wide_angle_reweight,
        edge_gate=edge_gate,
        edge_drop=edge_drop,
        shrink_drop=shrink_drop,
        occlusion_gate=occlusion_gate,
        aspect_gate=aspect_gate,
        distance_drop=distance_drop,
        max_speed_distance=max_speed_distance,
        image_width=image_width, image_height=image_height,
        valid_roi=valid_roi,
        camera_height_m=Constants.DEFAULT_CAMERA_HEIGHT_M,
        min_track_seconds=Constants.MIN_TRACK_SECONDS,
        reject_ids=reject_ids,
    )

    # ── WIDE-ANGLE DIAG (removable): split each vehicle's speed into along/cross-track
    # + dump the bbox-slide traces, to localize the wide-angle over-estimation. Delete
    # this block + wide_angle_diag.py to remove. ──
    if "--wa-diag" in sys.argv:
        try:
            import wide_angle_diag
            _base = os.path.dirname(os.path.abspath(frames_csv if is_android else telemetry_csv))
            _wa_id = _cli_value("--wa-diag-id", None)
            _only_ids = ([int(x) for x in str(_wa_id).split(",") if x.strip()]
                         if _wa_id else None)
            wide_angle_diag.dump(
                world, ego_pos, ego_heading,
                fx=fx, fy=fy, cx=cx, cy=cy, fps=fps, lat_sign=lat_sign,
                out_dir=os.path.join(_base, "wa_diag"), frame_ts=frame_ts,
                only_ids=_only_ids)
        except Exception as e:
            print(f"[wa-diag] failed: {e}")
    # ── end WIDE-ANGLE DIAG block ──

    # ── EGO-CANCEL DIAG (removable) ──
    if "--cancel-diag" in sys.argv:
        try:
            import ego_cancel_diag
            _base = os.path.dirname(os.path.abspath(frames_csv if is_android else telemetry_csv))
            _cd_id = _cli_value("--cancel-diag-id", None)
            _cd_only = ([int(x) for x in str(_cd_id).split(",") if x.strip()]
                        if _cd_id else None)
            _cd_sta = _cli_value("--cancel-diag-stationary", None)
            _cd_sta_ids = (set(int(x) for x in str(_cd_sta).split(",") if x.strip())
                           if _cd_sta else None)
            _cd_method = _cli_value("--cancel-diag-method", Constants.DISTANCE_CALCULATION_METHOD)
            _cd_gps = ego_yaw.ego_speed_from_android(frames_csv, gps_csv) if is_android else None
            ego_cancel_diag.dump(
                world, ego_pos, ego_heading,
                fx=fx, fy=fy, cx=cx, cy=cy, fps=fps, lat_sign=lat_sign,
                method=_cd_method, lateral_ref=lateral_ref, frame_ts=frame_ts,
                gps_speed=_cd_gps, only_ids=_cd_only, stationary_ids=_cd_sta_ids,
                camera_height_m=Constants.DEFAULT_CAMERA_HEIGHT_M,
                out_dir=os.path.join(_base, "cancel_diag"))
        except Exception as e:
            print(f"[cancel-diag] failed: {e}")
    # ── end EGO-CANCEL DIAG block ──


def processFrame(yolo_model, world: World, frame, frame_id):
    """Track vehicles + traffic lights into the World, feed the evidence buffer, and return
    this frame's vehicle records (track_id + bbox) for the yellow-line cache."""
    t_yolo = time.time()
    results = yolo_model.track(
        frame,
        persist=True,
        tracker=Constants.YOLO_TRACKER,
        verbose=False,
        classes=CLASSES,
        conf=CONFIDENCE_LVL,
        device=DEVICE,
        imgsz=int(_cli_value("--imgsz", Constants.YOLO_IMGSZ)),
        half=USE_HALF,
    )
    yolo_time = time.time() - t_yolo
    t_post = time.time()

    detections = results[0]
    vehicle_ids_in_frame = []
    traffic_light_ids_in_frame = []
    frame_vehicles = []                       # [{track_id, bbox}] for the yellow-line cache

    for box in detections.boxes:

        if box.id is None:
            continue

        object_id = int(box.id.item())
        object_type = int(box.cls.item())
        object_conf = float(box.conf.item()) if box.conf is not None else 1.0
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bounding_box = (x1, y1, x2, y2)

        # if the object is a vehicle
        if Constants.is_vehicle(object_type):
            vehicle_ids_in_frame.append(object_id)
            frame_vehicles.append({"track_id": object_id, "bbox": bounding_box})

            if world.getVehicle(object_id) is None:
                world.addVehicle(object_id, object_type, frame_id)

            v = world.getVehicle(object_id)
            v.updateBoxAndEndFrame(frame_id, bounding_box)
            v.record_classification(object_type, object_conf)

            # Evidence hook: offer this vehicle's crop to its rolling sharpest-crop buffer,
            # so the best pictures are already in memory if it later commits a violation.
            if EVIDENCE_COLLECTOR is not None:
                EVIDENCE_COLLECTOR.observe_vehicle(frame_id, frame, object_id, bounding_box)

        # if the object is a traffic light
        if Constants.is_traffic_light(object_type):
            traffic_light_ids_in_frame.append(object_id)

            if world.getTrafficLight(object_id) is None:
                world.addTrafficLight(object_id, frame_id)

            trl = world.getTrafficLight(object_id)
            trl.updateBoxAndEnsFrame(frame_id, bounding_box)

    world.registerFrame(frame_id, vehicle_ids_in_frame, traffic_light_ids_in_frame)
    postprocess_time = time.time() - t_post

    print(f"frame: {frame_id}")
    return yolo_time, postprocess_time, frame_vehicles


# --------------------------------------------------------------------------- #
# Yellow-line evaluation + combined-violation evidence stage (post-loop)
# --------------------------------------------------------------------------- #
def make_speed_lookup(world: World):
    """(track_id, frame) -> km/h from World speeds. Returns None when no world speed exists
    (raw dashcam clip, no ego pose) so shoulder_violation falls back to its motion proxy."""
    has_speed = any(v.speed_per_frame for v in world.vehicles.values())
    if not has_speed:
        print("[yellow] no world-frame speed available -> shoulder rule uses motion PROXY")
        return None
    def lookup(tid, frame):
        v = world.getVehicle(tid)
        if v is None:
            return 0.0
        s = v.speed_per_frame.get(frame)
        return (float(s) * 3.6) if s else 0.0      # m/s -> km/h (the gate's unit)
    return lookup


def evaluate_yellow_line(world: World, seg_frames: list, video_path: str, video_name: str,
                         frame_width: int, frame_height: int, total_frames: int, fps: float,
                         once_per_vehicle: bool = False):
    """Run the yellow-line (shoulder) detector over the live-built lane cache.
    Returns (events, recs, incidents). Empty if disabled or no frames.

    once_per_vehicle: report shoulder-driving STRICTLY ONCE per vehicle (Tal's baseline rule)."""
    if not (_YELLOW_IMPORT_OK and seg_frames):
        return [], {}, []
    cache = {"prefix": video_name, "path": video_path, "fps": fps,
             "w": frame_width, "h": frame_height, "total": total_frames, "frames": seg_frames}
    speed_lookup = make_speed_lookup(world)
    incidents, recs, events, yellow_frames = shoulder_violation.evaluate(
        cache, speed_lookup, once_per_vehicle=once_per_vehicle)
    print(f"[yellow] {yellow_frames} frames with a yellow line, "
          f"{len(incidents)} incident(s) -> {len(events)} ViolationEvent(s)")
    return events, recs, incidents


def overspeed_to_events(overspeed_events: list) -> list:
    """Convert OverspeedEvent -> the generic ViolationEvent the evidence stage consumes."""
    out = []
    for e in overspeed_events:
        out.append(ViolationEvent(
            vehicle_id=e.vehicle_id,
            violation_type=ViolationType.SPEEDING,
            key_frame=e.frame,
            confidence=1.0,
            details={"est_speed_kmh": e.est_speed_kmh, "speed_limit_kmh": e.speed_limit_kmh,
                     "over_by_kmh": e.over_by_kmh, "n_frames_over": e.n_frames_over},
        ))
    return out


def speeding_to_events(speeding_events: list) -> list:
    """Convert SpeedingEvent (fixed-limit, simulated) -> generic ViolationEvent.

    key_frame is the ONSET frame (the red-box "report once" moment); details carries the episode
    bounds + the peak speed reached inside the limit zone (max_speed_kmh)."""
    out = []
    for e in speeding_events:
        out.append(ViolationEvent(
            vehicle_id=e.vehicle_id,
            violation_type=ViolationType.SPEEDING,
            key_frame=e.onset_frame,
            confidence=1.0,
            details={"est_speed_kmh": e.est_speed_kmh, "max_speed_kmh": e.max_speed_kmh,
                     "speed_limit_kmh": e.speed_limit_kmh, "over_by_kmh": e.over_by_kmh,
                     "start_frame": e.start_frame, "end_frame": e.end_frame,
                     "n_frames_over": e.n_frames_over},
        ))
    return out


def _stage2_confirm(incidents: list, timeline: dict, seg_frames: list,
                     frame_height: int, video_path: str, tire_model) -> list:
    """Second-pass tire-model filter over Stage-1 crossing incidents.

    Re-reads the video once sequentially (grab() on non-candidate frames for speed), runs
    Stage2Cascade with live TireModel inference for each Stage-1 candidate frame, and returns
    only the incidents confirmed by Stage-2.  Incidents where Stage-2 never fires are dropped
    (FP killer).  Falls back to the full Stage-1 list if the video can't be opened."""
    import cv2 as _cv2
    from collections import defaultdict as _dd
    from violations.cascade.stage2 import CascadeParams, Stage2Cascade

    _SOLID_CLS = {"solid_white_lane", "traffic_island"}

    rec_by_f   = {fr["frame"]: fr for fr in seg_frames}
    inc_tids   = {inc[0] for inc in incidents}

    # Build per-frame candidate list (only Stage-1-hit frames for incident tracks).
    candidates_by_frame: dict = _dd(list)
    for tid, frame_hits in timeline.items():
        if tid not in inc_tids:
            continue
        for fi, hit in frame_hits.items():
            if not hit:
                continue
            rec = rec_by_f.get(fi)
            if rec is None:
                continue
            for v in rec.get("vehicles", []):
                if v.get("track_id") == tid:
                    candidates_by_frame[fi].append((tid, v["bbox"]))
                    break

    if not candidates_by_frame:
        return incidents  # no candidate frames found; keep all

    cap = _cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[crossing/stage2] cannot open video for second pass; keeping Stage-1 results")
        return incidents

    params   = CascadeParams()
    cascades = {tid: Stage2Cascade(params,
                                   tire_detect=lambda f, b, _tm=tire_model: _tm.detect_in_crop(f, b))
                for tid in inc_tids}
    stage2_fired: set = set()
    fi = 0
    max_fi = max(candidates_by_frame.keys())

    while fi <= max_fi:
        if fi in candidates_by_frame:
            ok, frame = cap.read()
            if not ok:
                break
            rec = rec_by_f.get(fi, {})
            solid_contours = [ln["contour"] for ln in rec.get("lanes", [])
                               if ln.get("cls") in _SOLID_CLS]
            for tid, bbox in candidates_by_frame[fi]:
                fired, _ = cascades[tid].step(frame, tid, bbox, solid_contours, frame_height)
                if fired:
                    stage2_fired.add(tid)
        else:
            if not cap.grab():
                break
        fi += 1

    cap.release()

    confirmed = [(tid, s, e) for tid, s, e in incidents if tid in stage2_fired]
    dropped   = len(incidents) - len(confirmed)
    print(f"[crossing/stage2] confirmed {len(confirmed)}/{len(incidents)} "
          f"(dropped {dropped} FP candidates)")
    return confirmed


def evaluate_crossing(seg_frames: list, fps: float, frame_height: int, frame_width: int,
                       video_path: str | None = None, tire_model=None) -> list:
    """Solid-line CROSSING detector (Stage-1 + optional Stage-2 tire confirmation).

    Stage-1: ghost/verdict timeline off the lane-seg cache -> K-consecutive hit frames ->
    per-vehicle 3 s cooldown.
    Stage-2 (when video_path + tire_model provided): re-reads the video for candidate frames,
    runs Stage2Cascade (TireModel + geometry) to confirm each Stage-1 incident, and drops
    incidents where no tire evidence is found.  Falls back to Stage-1 only if either is absent."""
    if not (_YELLOW_IMPORT_OK and seg_frames):
        return []

    shifts    = [fr.get("shift", [0.0, 0.0]) for fr in seg_frames]
    timeline  = compute_verdict_timeline(seg_frames, shifts, frame_height, frame_width, fps,
                                         ttl_sec=0.5, phantom_min_sec=0.0)
    k_consec  = max(1, round(0.05 * fps))
    incidents = merge_events(events_from_timeline(timeline, k_consec), fps, cooldown_sec=3.0)

    print(f"[crossing] Stage-1: {len(incidents)} incident(s)")
    if not incidents:
        return []

    if tire_model is not None and video_path is not None:
        incidents = _stage2_confirm(incidents, timeline, seg_frames,
                                    frame_height, video_path, tire_model)
    else:
        print("[crossing] Stage-2 skipped (no tire model or no video path)")

    events = []
    for tid, s, e in incidents:
        n_frames = e - s + 1
        conf     = min(1.0, max(0.30, n_frames / max(1.0, 0.3 * fps)))
        events.append(ViolationEvent(
            vehicle_id=tid,
            violation_type=ViolationType.SOLID_LINE_CROSSING,
            key_frame=s,
            confidence=round(conf, 4),
            details={"start_frame": s, "end_frame": e, "n_frames_over": n_frames},
        ))
    print(f"[crossing] {len(events)} crossing event(s) after Stage-2")
    return events


def run_evidence_and_report(world: World, all_events: list, out_dir: str, video_name: str):
    """For every ViolationEvent: read the plate ONCE per vehicle from the in-memory buffer,
    save the best-evidence crops, and write a consolidated violations CSV.
    RECALL-FIRST: an unreadable plate is written as 'UNKNOWN - manual review', never dropped."""
    import csv
    evidence_dir = os.path.join(out_dir, f"{video_name}_evidence")
    results_by_event = []
    for e in all_events:
        v = world.getVehicle(e.vehicle_id)
        known = v.license_plate if v is not None else None
        result = None
        if EVIDENCE_COLLECTOR is not None:
            result = EVIDENCE_COLLECTOR.collect_evidence(e, known_plate=known)
            if v is not None and result.plate:
                v.license_plate = result.plate            # read once; reused by later violations
            out = os.path.join(evidence_dir, f"v{e.vehicle_id}_{e.violation_type.lower()}")
            EVIDENCE_COLLECTOR.save_evidence(result, out, plate_reader=EVIDENCE_COLLECTOR._ocr.__self__)
        results_by_event.append((e, result))

    csv_path = os.path.join(out_dir, f"{video_name}_violations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["vehicle_id", "violation_type", "key_frame", "confidence", "est_speed_kmh",
                     "plate", "plate_score", "n_reads", "manual_review"])
        for e, result in results_by_event:
            plate = result.plate_label if result is not None else ""
            pscore = f"{result.plate_score:.3f}" if result is not None else ""
            nreads = result.n_reads if result is not None else ""
            manual = result.manual_review if result is not None else ""
            wr.writerow([e.vehicle_id, e.violation_type, e.key_frame, f"{e.confidence:.3f}",
                         e.details.get("est_speed_kmh", ""), plate, pscore, nreads, manual])
    print(f"[violations] {len(all_events)} event(s) -> {csv_path}")
    if EVIDENCE_COLLECTOR is not None and all_events:
        print(f"[violations] evidence crops -> {evidence_dir}")
    return results_by_event


def write_perframe_and_tracks(world: World, all_frame_vehicles: dict, recs: dict, incidents: list,
                              out_dir: str, video_name: str, fps: float):
    """Per-frame metrics (speed + yellow conf/violation) and a per-frame boxes table."""
    import csv
    firing = set()
    for tid, s, e in incidents:
        firing.update(range(s, e + 1))

    pf_path = os.path.join(out_dir, f"{video_name}_perframe.csv")
    with open(pf_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "time_s", "n_vehicles", "max_speed_kmh", "yellow_conf", "yellow_violation"])
        for f in sorted(all_frame_vehicles):
            vehs = all_frame_vehicles[f]
            speeds = []
            yconf = 0.0
            for vd in vehs:
                r = recs.get(vd["track_id"], {}).get(f)
                if r:
                    yconf = max(yconf, float(r.get("conf", 0.0)))
                    if r.get("speed"):
                        speeds.append(float(r["speed"]))
                else:
                    v = world.getVehicle(vd["track_id"])
                    s = v.speed_per_frame.get(f) if v is not None else None
                    if s:
                        speeds.append(float(s) * 3.6)
            wr.writerow([f, round(f / fps, 3), len(vehs),
                         round(max(speeds), 1) if speeds else "",
                         round(yconf * 100.0, 1), f in firing])

    tr_path = os.path.join(out_dir, f"{video_name}_tracks.csv")
    with open(tr_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "track_id", "x1", "y1", "x2", "y2"])
        for f in sorted(all_frame_vehicles):
            for vd in all_frame_vehicles[f]:
                x1, y1, x2, y2 = vd["bbox"]
                wr.writerow([f, vd["track_id"], x1, y1, x2, y2])
    print(f"[perframe] -> {pf_path}\n[tracks]   -> {tr_path}")


# --------------------------------------------------------------------------- #
# Backend export -- bundle each violation (annotated clip + 3 pics + plate crops + speeding .docx)
# and push it to the backend. Behind --push-url / --job-payload; a normal run is a no-op here.
# --------------------------------------------------------------------------- #
def _load_job_payload():
    """--job-payload <json|@file.json>: the backend job (Cloudflare presigned upload_url / notify_url,
    job_id, and optional reference video_meta). Returns the parsed dict, or None if absent/invalid."""

    raw = _cli_value("--job-payload", None)
    if not raw:
        return None
    try:
        if raw.startswith("@"):
            with open(raw[1:], encoding="utf-8") as fh:
                return json.load(fh)
        return json.loads(raw)
    except Exception as e:
        print(f"[export] could not parse --job-payload ({e}); export disabled")
        return None


def _video_meta_dict(job, video_path, frame_w, frame_h, fps, total_frames):
    """Reference VideoMeta for the manifest. Prefer the fingerprint the backend put in the job
    payload (from its DB / Cloudflare); otherwise synthesise one from what the live run already
    knows (no ffprobe dependency on the hot path -- sha256/codec left null)."""

    if job and job.get("video_meta"):
        return dict(job["video_meta"])
    return {"filename": os.path.basename(video_path), "width": int(frame_w),
            "height": int(frame_h), "fps": round(float(fps), 3),
            "frame_count": int(total_frames), "sha256": None,
            "duration_sec": round(total_frames / fps, 3) if fps else None, "codec": None}


def pull_job_video(job: dict):
    """Worker-node PULL+VERIFY step: download the source clip from the job's presigned Cloudflare
    GET URL and fail-fast check it against the reference fingerprint carried in the job payload.

    Returns ``(local_video_path, reference_meta_dict, work_dir)``. Raises (IntegrityError / HTTP /
    ValueError) on a corrupt download or a job missing video_url/video_meta -- the worker SHOULD die
    here, before burning GPU on a bad clip. The download needs ``requests`` (lazy) + ``ffprobe`` for
    the verify; both live in violations.ingest_client / video_integrity."""

    import tempfile
    from violations import ingest_client

    job_id = str(job.get("job_id") or "job")
    # Name the local file from the explicit job field, else the URL path (sans query string).

    name = job.get("video_name") \
        or os.path.basename((job.get("video_url") or "video.mp4").split("?")[0]) \
        or "video.mp4"
    work_dir = tempfile.mkdtemp(prefix=f"roadguard_{job_id}_")
    dest = os.path.join(work_dir, name)
    print(f"[job {job_id}] pulling '{name}' from presigned URL -> {dest}")
    local_path, ref_meta, local_meta = ingest_client.download_and_verify_from_job(job, dest)
    print(f"[job {job_id}] integrity OK: {local_meta.width}x{local_meta.height} "
          f"@ {local_meta.fps:g}fps, {local_meta.frame_count} frames")
    return local_path, ref_meta.to_dict(), work_dir


def export_and_push_violations(world: World, results_by_event: list, *, job: dict | None,
                               video_path: str, video_name: str, fps: float, frame_w: int,
                               frame_h: int, total_frames: int, out_dir: str):
    """Assemble the prioritised evidence bundle for every violation and upload it to the backend.

    Triggered ONLY by --push-url (legacy multipart POST through the Node server) or --job-payload
    (Cloudflare: presigned PUT of the .tar.gz + a lightweight webhook notify). No flag -> no-op.
    ``job`` is the already-parsed --job-payload dict (or None). Reuses the same EvidenceResult (plate
    + crops) the evidence stage already collected, so no video re-read for the pictures. RECALL-FIRST:
    a clip/.docx that fails to render is skipped and the violation is still shipped."""
    push_url = _cli_value("--push-url", None)
    if not push_url and not job:
        return                                       # standard run -- no backend export

    if not _EXPORT_IMPORT_OK:
        print("[export] --push-url/--job-payload set but violations.export is unavailable; skipped")
        return
    if not results_by_event:
        print("[export] no violations to export; nothing pushed")
        return

    video_meta = _video_meta_dict(job, video_path, frame_w, frame_h, fps, total_frames)
    records = []
    for event, result in results_by_event:
        v = world.getVehicle(event.vehicle_id)
        boxes = v.bounding_box if v is not None else {}
        win = clip_window(event.key_frame, fps, total_frames=total_frames)
        stem = vx.violation_id(event)
        clip = None
        try:                                         # annotated clip: red box on the offending car + caption

            clip = annotate_clip(video_path, os.path.join(out_dir, f"{stem}.mp4"), win,
                                 box_for_frame=boxes.get,
                                 caption=vx.describe_violation(event), fps=fps,
                                 tag=f"VEH {event.vehicle_id} - {event.violation_type}")
        except Exception as e:
            print(f"[export] clip render failed for {stem} ({e}); shipping without a clip")
        report = None
        if event.violation_type == ViolationType.SPEEDING:
            try:
                report = docx_report.speeding_report(event, video_meta=video_meta,
                                                     window=win.to_dict())
            except Exception as e:
                print(f"[export] .docx render failed for {stem} ({e})")
        records.append((event, result, clip, report))

    bundle = vx.build_export(records, video_meta=video_meta,
                             client_info={"app": "roadguard", "component": "main",
                                          "source_video": video_name})
    targz = vx.bundle_targz(bundle)
    bundle_path = os.path.join(out_dir, f"{video_name}_violations_bundle.tar.gz")
    with open(bundle_path, "wb") as fh:
        fh.write(targz)
    queue = [(x["violation"], x["vehicle_id"], x["detector_confidence"]) for x in bundle.violations]
    print(f"[export] {len(bundle.violations)} violation(s), {len(bundle.files)} file(s), "
          f"{len(targz):,} bytes -> {bundle_path}\n[export] queue order: {queue}")

    job_id = (job or {}).get("job_id")
    if job and job.get("upload_url"):                # Cloudflare presigned PUT + webhook notify

        res = vx.upload_bundle_presigned(
            job["upload_url"], bundle, notify_url=job.get("notify_url"), job_id=job_id,
            source_video=video_name, object_key=job.get("object_key"),
            object_url=job.get("object_url"))
        print(f"[export] presigned PUT -> {res.put.status_code} (ok={res.put.ok}); "
              f"notify -> {(res.notify.status_code if res.notify else 'skipped')}")
    elif push_url:                                   # legacy multipart POST through the Node server

        extra = {"source_video": video_name}
        if job_id:
            extra["job_id"] = job_id
        resp = vx.post_bundle(push_url, targz, extra_fields=extra)
        print(f"[export] POST {push_url} -> {resp.status_code} (ok={resp.ok})")
    else:
        print(f"[export] job payload had no upload_url and no --push-url; bundle saved locally only")


def process_video_with_models(
    video_path: str,
    yolo_model,
    lane_model,
    tire_model,
    *,
    out_dir: str | None = None,
    max_frames: int = 0,
    is_simulation: bool = True,
    lane_conf: float = 0.25,
    benchmark: bool = False,
    sim_speed: bool = False,
    job: dict | None = None,
    undistorter=None,
) -> dict:
    """Process a single video with pre-loaded models.

    Called by main() (CLI) and by the in-process validation harness so models load
    once for N videos instead of once per subprocess.  The BoT-SORT tracker is reset
    at the start of each call so track IDs are always fresh.

    Returns {"annotated_video", "vehicles_csv", "violations"} matching main()."""
    # Reset BoT-SORT tracker state so each video starts with fresh track IDs.
    if getattr(yolo_model, "predictor", None) is not None:
        yolo_model.predictor = None

    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if out_dir is None:
        out_dir = video_dir
    os.makedirs(out_dir, exist_ok=True)

    yellow_enabled = (lane_model is not None) and _YELLOW_IMPORT_OK

    vh = VideoHandler(video_path)
    frame_height, frame_width = vh.get_frame().shape[:2]
    world = World(vh.get_frame_count())
    print(f"number of frames in the video is {vh.get_frame_count()}")

    frame_id = 0
    read_times, yolo_times, postprocess_times = [], [], []
    all_frame_vehicles: dict[int, list] = {}
    seg_frames: list = []
    prev_gray = None

    start_time = time.time()
    while True:
        t_read = time.time()
        frame = vh.get_frame()
        if undistorter is not None:
            frame = undistorter(frame)
        read_times.append(time.time() - t_read)
        if frame is None:
            read_times.pop()
            break

        yolo_time, postprocess_time, frame_vehicles = processFrame(yolo_model, world, frame, frame_id)
        yolo_times.append(yolo_time)
        postprocess_times.append(postprocess_time)
        all_frame_vehicles[frame_id] = frame_vehicles

        if yellow_enabled:
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shift = estimate_ego_shift(prev_gray, cur_gray) if prev_gray is not None else (0.0, 0.0)
            prev_gray = cur_gray
            seg_frames.append({
                "frame": frame_id, "vehicles": frame_vehicles,
                "lanes": seg_lanes(lane_model, frame, lane_conf),
                "shift": [round(shift[0], 2), round(shift[1], 2)],
            })

        frame_id += 1
        if max_frames and frame_id >= max_frames:
            break
        t_read = time.time()
        vh.read_next()
        read_times[-1] += time.time() - t_read

    vh.release()
    print_time(start_time, read_times, yolo_times, postprocess_times, frame_id)

    # ── 1. Input mode ────────────────────────────────────────────────────────
    telemetry_csv   = os.path.join(video_dir, "telemetry.csv")
    frames_csv      = os.path.join(video_dir, "frames.csv")
    gyro_csv        = os.path.join(video_dir, "gyro.csv")
    gravity_csv     = os.path.join(video_dir, "gravity.csv")
    gps_csv         = os.path.join(video_dir, "gps.csv")
    linacc_csv      = os.path.join(video_dir, "linacc.csv")
    intrinsics_json = os.path.join(video_dir, "intrinsics.json")
    is_android = (not is_simulation) and all(
        os.path.exists(p) for p in (frames_csv, gyro_csv, gravity_csv, gps_csv))

    # ── 3. Finalize tracks ───────────────────────────────────────────────────
    for vehicle in world.vehicles.values():
        vehicle.resolve_vehicle_type()

    # ── 4. Timebase & bbox gap-fill ──────────────────────────────────────────
    fps = vh.get_fps()
    if is_android:
        fps = ego_yaw.fps_from_frame_timestamps(frames_csv) or fps
    interpolate_bbox_gaps(world, fps, max_gap_seconds=Constants.MAX_GAP_SECONDS)

    # ── 5. Camera intrinsics ─────────────────────────────────────────────────
    if is_android:
        fx, fy, cx, cy = ego_yaw.load_intrinsics(intrinsics_json, frame_width, frame_height)
    else:
        fl = focal_length_from_fov(frame_width, fov_horizontal_deg=Constants.FOV_HORIZONTAL_DEG)
        fx = fy = fl
        cx = frame_width / 2
        cy = frame_height / 2

    if undistorter is not None and undistorter.K_used is not None:
        K = undistorter.K_used
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        print(f"[undistort] intrinsics overridden to rectified K: "
              f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # ── 6. Distance ──────────────────────────────────────────────────────────
    estimateDistance(world, fx, fy, cx, cy, Constants.DEFAULT_CAMERA_HEIGHT_M)
    smooth_distances(world, Constants.SMOOTH_WINDOW, Constants.POLYORDER)

    # ── 7. Speed ─────────────────────────────────────────────────────────────
    valid_roi = undistorter.valid_roi if undistorter is not None else None
    run_speed_estimation(
        world, fps, fx, fy, cx, cy,
        is_simulation=is_simulation, is_android=is_android,
        telemetry_csv=telemetry_csv, frames_csv=frames_csv,
        gyro_csv=gyro_csv, gravity_csv=gravity_csv, gps_csv=gps_csv,
        linacc_csv=linacc_csv,
        image_width=frame_width, image_height=frame_height,
        valid_roi=valid_roi)

    # ── 7.0 Simulated speeds (--sim-speed mode) ───────────────────────────────
    overspeed_events = []
    speeding_events = []
    if sim_speed:
        from speed_estimation import simulated_speed as _sim_spd
        _sim_spd.assign_simulated_speeds(world, fps)
        speeding_events = overspeed.flag_speeding_fixed_limit(
            world,
            limit_kmh=_sim_spd.SIM_LIMIT_KMH,
            threshold_kmh=_sim_spd.SIM_SPEEDING_THRESHOLD_KMH,
            fps=fps)
        print(f"[speeding] {len(speeding_events)} speeding episode(s) "
              f"(limit={_sim_spd.SIM_LIMIT_KMH:.0f}, "
              f">={_sim_spd.SIM_SPEEDING_THRESHOLD_KMH:.0f} km/h)")
    elif is_android and "--no-overspeed" not in sys.argv:
        try:
            margin = float(_cli_value("--overspeed-margin", overspeed.OVERSPEED_MARGIN_KMH))
            ego_track = overspeed.build_ego_track(frames_csv, gps_csv)
            overspeed_events = overspeed.flag_overspeed_vehicles(world, ego_track, margin_kmh=margin)
            report = overspeed.format_overspeed_report(overspeed_events)
            print("\n[overspeed] " + report)
            with open(os.path.join(out_dir, f"{video_name}_overspeed.txt"), "w", encoding="utf-8") as fh:
                fh.write(report + "\n")
            overspeed.write_overspeed_csv(overspeed_events,
                                          os.path.join(out_dir, f"{video_name}_overspeed.csv"))
        except Exception as e:
            print(f"[overspeed] skipped (lookup/parse failed): {e}")

    # ── 7.6 Yellow-line (shoulder) violation ──────────────────────────────────
    yellow_events, recs, incidents = evaluate_yellow_line(
        world, seg_frames, video_path, video_name,
        frame_width, frame_height, frame_id, fps, once_per_vehicle=sim_speed)

    # ── 7.65 Solid-line crossing violation (Stage-1 + Stage-2 tire confirmation) ──
    crossing_events = evaluate_crossing(seg_frames, fps, frame_height, frame_width,
                                        video_path=video_path, tire_model=tire_model)

    # ── 7.7 Evidence stage ────────────────────────────────────────────────────
    all_events = (crossing_events
                  + speeding_to_events(speeding_events) + overspeed_to_events(overspeed_events)
                  + yellow_events)
    results_by_event = run_evidence_and_report(world, all_events, out_dir, video_name)
    write_perframe_and_tracks(world, all_frame_vehicles, recs, incidents, out_dir, video_name, fps)

    # ── 7.75 Backend export (opt-in) ──────────────────────────────────────────
    export_and_push_violations(
        world, results_by_event, job=job, video_path=video_path, video_name=video_name, fps=fps,
        frame_w=frame_width, frame_h=frame_height, total_frames=frame_id, out_dir=out_dir)

    # ── 8. Outputs ────────────────────────────────────────────────────────────
    distanceLogger.export_vehicle_summary(
        world, os.path.join(out_dir, f"{video_name}_vehicles.csv"))

    if benchmark:
        print("[benchmark] done -- CSV/text data written; skipped plots + annotated video render")
        return {
            "annotated_video": None,
            "vehicles_csv":    os.path.join(out_dir, f"{video_name}_vehicles.csv"),
            "violations":      [],
        }

    draw_vehicle_plots.plot_all_vehicles(
        world, out_dir, video_name,
        telemetry_csv=telemetry_csv if is_simulation else None)

    ego_speed: dict[int, float] = {}
    ego_source = ""
    ego_speed_raw: dict[int, float] | None = None
    raw_source = ""
    if is_simulation and os.path.exists(telemetry_csv):
        ego_speed, ego_source = ego_yaw.ego_speed_from_telemetry(telemetry_csv), "telemetry"
    elif is_android:
        gps_speed = ego_yaw.ego_speed_from_android(frames_csv, gps_csv)
        fused_speed = ego_yaw.ego_speed_fused(frames_csv, gps_csv, linacc_csv, gravity_csv)
        if fused_speed:
            ego_speed, ego_source = fused_speed, "accelerometer+GPS fused"
            ego_speed_raw, raw_source = gps_speed, "GPS only"
        else:
            ego_speed, ego_source = gps_speed, "GPS"
    draw_vehicle_plots.plot_ego_speed(ego_speed, out_dir, video_name, source=ego_source,
                                      ego_speed_raw=ego_speed_raw, raw_source=raw_source)

    annotated_final = os.path.join(out_dir, f"{video_name}_annotated.mp4")
    with cloud_env.staged_output(annotated_final) as render_path:
        annotated_video.render_annotated_video(
            world, video_path, render_path, fps=fps,
            violation_events=all_events)

    violations = []
    for e, result in results_by_event:
        vehicle = world.getVehicle(e.vehicle_id)
        plate = ((result.plate if result is not None else None)
                 or (vehicle.license_plate if vehicle else None))
        violations.append({
            "vehicle_id":      e.vehicle_id,
            "carId":           plate or f"vehicle-{e.vehicle_id}",
            "calculatedSpeed": e.details.get("est_speed_kmh", 0),
            "lat":             0.0,
            "lon":             0.0,
            "violation_type":  e.violation_type,
        })

    return {
        "annotated_video": annotated_final,
        "vehicles_csv":    os.path.join(out_dir, f"{video_name}_vehicles.csv"),
        "violations":      violations,
    }


def main():
    global EVIDENCE_COLLECTOR, LANE_MODEL, TIRE_MODEL

    # Cloud awareness FIRST: detect Colab/headless (auto, or via --colab) and neutralize the
    # cv2 GUI calls so nothing crashes without a display. Everything below stays env-agnostic.
    cloud_env.init()

    yolo_model = loadYoloModel()

    # ── Worker-node mode (--job-payload) ──────────────────────────────────────
    job = _load_job_payload()
    job_work_dir = None
    if job and job.get("video_url"):
        video_path, _job_ref_meta, job_work_dir = pull_job_video(job)
    else:
        video_path = sys.argv[1]
    is_simulation = "--simulation" in sys.argv
    sim_speed     = "--sim-speed"  in sys.argv
    benchmark     = "--benchmark"  in sys.argv
    if benchmark:
        print("[benchmark] fast-data mode: CSV/text outputs only (no plots, no annotated video)")

    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = _cli_value("--out-dir", video_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Evidence collector (one OCR reader for the whole run).
    if _EVIDENCE_IMPORT_OK and "--no-evidence" not in sys.argv:
        try:
            EVIDENCE_COLLECTOR = EvidenceCollector(FastALPRReader().read_plate_with_conf,
                                                   min_area=LPR.MIN_VEHICLE_AREA)
            print("[evidence] EvidenceCollector ready (FastALPR)")
        except Exception as e:
            print(f"[evidence] disabled (reader init failed: {e})")
            EVIDENCE_COLLECTOR = None

    # Yellow-line lane-seg model (a SECOND model run per frame). --no-yellow disables it.
    yellow_enabled = _YELLOW_IMPORT_OK and "--no-yellow" not in sys.argv
    if yellow_enabled:
        lane_weights = _cli_value("--lane-weights",
                                  os.path.join(REPO_ROOT, "weights", "phase3_v3_yellowprotect.pt"))
        if os.path.isfile(lane_weights):
            LANE_MODEL = loadLaneModel(lane_weights)
        else:
            print(f"[yellow] lane weights not found ({lane_weights}); yellow-line disabled")
        yellow_enabled = LANE_MODEL is not None
    lane_conf = float(_cli_value("--lane-conf", 0.25))
    max_frames = int(_cli_value("--max-frames", 0))

    # Stage-2 tire model (solid-line crossing FP killer). Skipped with --no-stage2 or missing weights.
    if "--no-stage2" not in sys.argv:
        _tire_weights = _cli_value("--tire-weights",
                                   os.path.join(REPO_ROOT, "models", "tire_yolo11n.pt"))
        if os.path.isfile(_tire_weights):
            try:
                from violations.cascade.tire_model import TireModel as _TireModel
                TIRE_MODEL = _TireModel(_tire_weights)
                print(f"[stage2] tire model loaded: {_tire_weights}")
            except Exception as _te:
                print(f"[stage2] tire model failed to load ({_te}); Stage-2 disabled")
        else:
            print(f"[stage2] tire weights not found ({_tire_weights}); Stage-2 disabled")

    # ── UNDISTORT (removable) ─────────────────────────────────────────────────
    undistorter = None
    if "--undistort" in sys.argv:
        import undistort as _undistort
        _report = _undistort.find_report(os.path.dirname(os.path.abspath(video_path)))
        if _report:
            undistorter = _undistort.FrameUndistorter.from_report(_report)
            print(f"[undistort] using calibration report: {_report}")
        else:
            print("[undistort] --undistort set but no calibration report found; running raw")

    return process_video_with_models(
        video_path, yolo_model, LANE_MODEL, TIRE_MODEL,
        out_dir=out_dir, max_frames=max_frames, is_simulation=is_simulation,
        lane_conf=lane_conf, benchmark=benchmark, sim_speed=sim_speed,
        job=job, undistorter=undistorter,
    )


# the main function of the program
if __name__ == "__main__":
    main()
