import sys
import os
import time
import torch
import cv2
import numpy as np
import speed_estimation.botsort_patch
import cloud_env                          # headless/Colab detection + smart Drive I/O
from collections import Counter
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
    from ghost_mask import estimate_ego_shift          # optical-flow ego shift (same as the harness)
    import shoulder_violation                          # yellow-line evaluator (emits ViolationEvent)
    _YELLOW_IMPORT_OK = True
except Exception as _e:                                # pragma: no cover
    print(f"[yellow] import unavailable ({_e}); yellow-line rule will be skipped")
    _YELLOW_IMPORT_OK = False


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
def loadYoloModel():
    weights = _cli_value("--model", Constants.YOLO_VERSION)
    print(f"Loading {weights} on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    try:
        model = YOLO(weights)
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


def run_speed_estimation(world: World, fps: float,
                         fx: float, fy: float, cx: float, cy: float,
                         *,
                         is_simulation: bool, is_android: bool,
                         telemetry_csv: str, frames_csv: str,
                         gyro_csv: str, gravity_csv: str, gps_csv: str,
                         linacc_csv: str = "",
                         image_width: int = 0, image_height: int = 0):
    smoother     = _cli_value("--smoother", SMOOTHER)
    lat_sign     = int(_cli_value("--lat-sign", Constants.LAT_SIGN))
    heading_sign = int(_cli_value("--heading-sign", Constants.HEADING_SIGN))
    lateral_ref  = _cli_value("--lateral-ref", Constants.LATERAL_REF)
    wide_angle_reweight = str(_cli_value(
        "--wide-angle-reweight", Constants.WIDE_ANGLE_REWEIGHT)).lower() in ("1", "true", "yes", "on")
    edge_gate = str(_cli_value(
        "--edge-gate", Constants.EDGE_GATE)).lower() in ("1", "true", "yes", "on")

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

    print(f"[main] world-frame speed: smoother={smoother}, "
          f"lat_sign={lat_sign}, heading_sign={heading_sign}, lateral_ref={lateral_ref}, "
          f"wide_angle_reweight={wide_angle_reweight}, edge_gate={edge_gate}")
    frame_ts = ego_yaw.load_frame_timestamps(frames_csv) if is_android else None
    estimate_world_speeds(
        world, ego_pos, ego_heading, fps,
        fx=fx, fy=fy, cx=cx, cy=cy,
        frame_ts=frame_ts,
        method="height",
        smoother=smoother,
        lat_sign=lat_sign,
        lateral_ref=lateral_ref,
        wide_angle_reweight=wide_angle_reweight,
        edge_gate=edge_gate,
        image_width=image_width, image_height=image_height,
        camera_height_m=Constants.DEFAULT_CAMERA_HEIGHT_M,
        min_track_seconds=Constants.MIN_TRACK_SECONDS,
    )


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
                         frame_width: int, frame_height: int, total_frames: int, fps: float):
    """Run the yellow-line (shoulder) detector over the live-built lane cache.
    Returns (events, recs, incidents). Empty if disabled or no frames."""
    if not (_YELLOW_IMPORT_OK and seg_frames):
        return [], {}, []
    cache = {"prefix": video_name, "path": video_path, "fps": fps,
             "w": frame_width, "h": frame_height, "total": total_frames, "frames": seg_frames}
    speed_lookup = make_speed_lookup(world)
    incidents, recs, events, yellow_frames = shoulder_violation.evaluate(cache, speed_lookup)
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


def run_evidence_and_report(world: World, all_events: list, out_dir: str, video_name: str):
    """For every ViolationEvent (speeding + yellow-line): read the plate ONCE per vehicle from
    the in-memory buffer, save the best-evidence crops, and write a consolidated violations CSV.
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


def write_perframe_and_tracks(world: World, all_frame_vehicles: dict, recs: dict, incidents: list,
                              out_dir: str, video_name: str, fps: float):
    """Per-frame metrics (speed + yellow conf/violation) and a per-frame boxes table -- the join
    data the A/B runner merges across detectors (frame-by-frame compare + IoU recall tally)."""
    import csv
    firing = set()                              # (frame) -> any vehicle firing a yellow incident
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
                    if r.get("speed"):          # yellow's per-frame speed (world km/h, else proxy)
                        speeds.append(float(r["speed"]))
                else:                            # yellow disabled -> fall back to world speed
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


def main():
    global EVIDENCE_COLLECTOR, LANE_MODEL

    # Cloud awareness FIRST: detect Colab/headless (auto, or via --colab) and neutralize the
    # cv2 GUI calls so nothing crashes without a display. Everything below stays env-agnostic.
    cloud_env.init()

    yolo_model = loadYoloModel()

    video_path = sys.argv[1]
    is_simulation = "--simulation" in sys.argv
    # --benchmark: fast-data mode. Skip every rendered output (per-vehicle plots + the
    # annotated video) and emit only the CSV/text data.
    benchmark = "--benchmark" in sys.argv
    if benchmark:
        print("[benchmark] fast-data mode: CSV/text outputs only (no plots, no annotated video)")

    # Output routing: --out-dir keeps the two A/B detectors from clobbering each other.
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = _cli_value("--out-dir", video_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Evidence collector (one OCR reader for the whole run). Skipped on --no-evidence or if
    # fast_alpr is unavailable -- the rest of the pipeline is unaffected.
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
    max_frames = int(_cli_value("--max-frames", 0))     # 0 = whole video (quick-test knob)

    vh = VideoHandler(video_path)
    frame_height, frame_width = vh.get_frame().shape[:2]
    world = World(vh.get_frame_count())
    print(f"number of frames in the video is {vh.get_frame_count()}")

    frame_id = 0
    read_times, yolo_times, postprocess_times = [], [], []
    all_frame_vehicles: dict[int, list] = {}            # frame -> [{track_id, bbox}]
    seg_frames: list = []                               # per-frame lane-seg cache (yellow rule)
    prev_gray = None

    start_time = time.time()
    while True:
        t_read = time.time()
        frame = vh.get_frame()
        read_times.append(time.time() - t_read)
        if frame is None:
            read_times.pop()
            break

        yolo_time, postprocess_time, frame_vehicles = processFrame(yolo_model, world, frame, frame_id)
        yolo_times.append(yolo_time)
        postprocess_times.append(postprocess_time)
        all_frame_vehicles[frame_id] = frame_vehicles

        # Build the yellow-line cache live: lane segmentation + optical-flow ego shift.
        if yellow_enabled:
            cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shift = estimate_ego_shift(prev_gray, cur_gray) if prev_gray is not None else (0.0, 0.0)
            prev_gray = cur_gray
            seg_frames.append({"frame": frame_id, "vehicles": frame_vehicles,
                               "lanes": seg_lanes(LANE_MODEL, frame, lane_conf),
                               "shift": [round(shift[0], 2), round(shift[1], 2)]})



        frame_id += 1
        if max_frames and frame_id >= max_frames:
            break
        t_read = time.time()
        vh.read_next()
        read_times[-1] += time.time() - t_read

    vh.release()
    print_time(start_time, read_times, yolo_times, postprocess_times, frame_id)

    for v in world.vehicles.values():
        if v._plate_candidates:
            counts = Counter(v._plate_candidates)
            max_count = max(counts.values())
            v.license_plate = next(p for p in reversed(v._plate_candidates) if counts[p] == max_count)

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

    # ── 6. Distance ──────────────────────────────────────────────────────────
    estimateDistance(world, fx, fy, cx, cy, camera_height_m=Constants.DEFAULT_CAMERA_HEIGHT_M)
    smooth_distances(world, smooth_window=Constants.SMOOTH_WINDOW)

    # ── 7. Speed (world-frame reconstruction + Kalman) ───────────────────────
    run_speed_estimation(
        world, fps, fx, fy, cx, cy,
        is_simulation=is_simulation, is_android=is_android,
        telemetry_csv=telemetry_csv, frames_csv=frames_csv,
        gyro_csv=gyro_csv, gravity_csv=gravity_csv, gps_csv=gps_csv,
        linacc_csv=linacc_csv,
        image_width=frame_width, image_height=frame_height)

    # ── 7.5 Overspeed (android-only; needs the ego GPS track as the limit proxy) ──
    overspeed_events = []
    if is_android and "--no-overspeed" not in sys.argv:
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

    # ── 7.6 Yellow-line (shoulder) violation ─────────────────────────────────
    yellow_events, recs, incidents = evaluate_yellow_line(
        world, seg_frames, video_path, video_name,
        frame_width, frame_height, frame_id, fps)

    # ── 7.7 Evidence stage over ALL violations (speeding + yellow), one record shape ──
    all_events = overspeed_to_events(overspeed_events) + yellow_events
    run_evidence_and_report(world, all_events, out_dir, video_name)
    write_perframe_and_tracks(world, all_frame_vehicles, recs, incidents, out_dir, video_name, fps)

    # ── 8. Outputs ───────────────────────────────────────────────────────────
    # CSV data first -- this is the only output --benchmark keeps.
    distanceLogger.export_vehicle_summary(
        world, os.path.join(out_dir, f"{video_name}_vehicles.csv"))

    if benchmark:
        print("[benchmark] done -- CSV/text data written; skipped plots + annotated video render")
        return

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

    # Smart cloud I/O: on Colab the annotated video is rendered to local NVMe and copied to
    # Drive once at the end (staged_output), instead of writing every frame to the slow Drive
    # FUSE mount. Off Colab this is a transparent no-op writing straight to out_dir.
    annotated_final = os.path.join(out_dir, f"{video_name}_annotated.mp4")
    with cloud_env.staged_output(annotated_final) as render_path:
        annotated_video.render_annotated_video(world, video_path, render_path, fps=fps)


# the main function of the program
if __name__ == "__main__":
    main()
