import sys
import os
import time
import torch
import cv2
import speed_estimation.botsort_patch
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
from speed_estimation import relevance_flags
# Clean-room ground-plane distance estimator (separate package). Replaces the old
# estimateDistance/smooth_distances for the distance step; main.py stays the runner
# and the YOLO/ByteTrack loop + Vehicle/World population are unchanged.
#from ground_distance import estimate_ground_distances


DEVICE = 0 if torch.cuda.is_available() else 'cpu'
USE_HALF = torch.cuda.is_available()   # FP16 inference - faster on modern NVIDIA GPUs

YOLO_MODEL = None
CLASSES = Constants.DETECTION_CLASSES
CONFIDENCE_LVL = Constants.CONFIDENCE_LVL

SMOOTHER = Constants.DEFAULT_SMOOTHER



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

# load yolov8 model and set it to YOLO_MODEL (for now we'll use model x)
def loadYoloModel():
    model_file = str(Constants.YOLO_VERSION)
    is_pt = model_file.endswith(".pt")
    print(f"Loading {model_file} on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    # task= is required for exported engines/onnx -- they don't carry task metadata,
    # so Ultralytics can't auto-guess it (the "Unable to guess model task" warning).
    model = YOLO(model_file, task="detect")
    # .to() only works on .pt models. An exported .engine is already bound to the
    # device it was built for, and processFrame's track(..., device=DEVICE) sets the
    # device anyway -- so calling .to() on an engine just raises. Skip it for engines.
    if is_pt:
        model.to(DEVICE)
    return model


def _cli_value(flag: str, default):
    """Return the argv token after `flag`, or `default` if the flag is absent.

    Lets a run override a Constant without editing Constants.py -- used for the
    handedness signs (--lat-sign / --heading-sign) that must be re-validated on
    real footage, and for --smoother.
    """
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


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
    # smoother + handedness-sign selection (CLI overrides the Constants defaults).
    # lat_sign / heading_sign are mount/handedness dependent and unknown on a new
    # real clip -- sweep them with these flags instead of editing Constants.py.
    smoother     = _cli_value("--smoother", SMOOTHER)
    lat_sign     = int(_cli_value("--lat-sign", Constants.LAT_SIGN))
    heading_sign = int(_cli_value("--heading-sign", Constants.HEADING_SIGN))
    # Cross-track reference point: "center" (legacy) vs "near_edge" (suppresses the
    # wide-angle lateral blow-up). Swept here so it can be A/B'd without editing Constants.
    lateral_ref  = _cli_value("--lateral-ref", Constants.LATERAL_REF)
    # Wide-angle down-weighting: trust wide-bearing frames less in the smoother.
    # Off by default; pass `--wide-angle-reweight 1` to enable for an A/B test.
    wide_angle_reweight = str(_cli_value(
        "--wide-angle-reweight", Constants.WIDE_ANGLE_REWEIGHT)).lower() in ("1", "true", "yes", "on")
    # Frame-edge gate: heavily down-weight frames whose bbox touches the border.
    # Off by default; pass `--edge-gate 1` to enable.
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
    # Occlusion down-weighting (note #2): trust frames LESS where the vehicle is hidden
    # behind a nearer one. ON by default (Constants.OCCLUSION_GATE); pass `--occlusion-gate 0`
    # to disable.
    occlusion_gate = str(_cli_value(
        "--occlusion-gate", Constants.OCCLUSION_GATE)).lower() in ("1", "true", "yes", "on")
    # Aspect-ratio gate: trust frames LESS where the bbox width/height ratio jumps abruptly
    # (clip/collapse/flicker; a smooth turn is not penalised). ON by default
    # (Constants.ASPECT_GATE); pass `--aspect-gate 0` to disable.
    aspect_gate = str(_cli_value(
        "--aspect-gate", Constants.ASPECT_GATE)).lower() in ("1", "true", "yes", "on")
    # Relevance rejection (note #1), two INDEPENDENT criteria. Each ON by default
    # (Constants.REJECT_BIG_LATERAL / REJECT_DIRECTION); pass `--reject-lateral 0`
    # and/or `--reject-direction 0` to disable. Both off -> no vehicle is rejected.
    reject_lateral = str(_cli_value(
        "--reject-lateral", Constants.REJECT_BIG_LATERAL)).lower() in ("1", "true", "yes", "on")
    reject_direction = str(_cli_value(
        "--reject-direction", Constants.REJECT_DIRECTION)).lower() in ("1", "true", "yes", "on")
    # Far-distance DROP: ignore (for speed only) frames whose estimated depth is
    # beyond max_speed_distance -- height-based depth is too noisy that far out.
    # ON by default (Constants.DISTANCE_DROP); pass `--distance-drop 0` to disable,
    # or `--max-speed-distance <m>` to change the cutoff (default MAX_SPEED_DISTANCE_M).
    distance_drop = str(_cli_value(
        "--distance-drop", Constants.DISTANCE_DROP)).lower() in ("1", "true", "yes", "on")
    max_speed_distance = float(_cli_value("--max-speed-distance", Constants.MAX_SPEED_DISTANCE_M))
    # Ego<->camera time-sync test: shift the ego pose (position AND heading, both on
    # the sensor clock) by N frames vs the camera bboxes. 0 = no shift. Sweep e.g.
    # `--ego-shift 30` / `--ego-shift -30` to check for a sensor lag that misaligns
    # ego-motion cancellation. Speed-path only; distance export is unaffected.
    ego_shift = int(_cli_value("--ego-shift", 0))

    # Ego POSE feeds the same world-frame core from either source: telemetry in
    # sim, reconstructed from gyro + GPS for an Android clip.
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

    # Ego<->camera time-sync test (--ego-shift N): slide the sensor-clock pose
    # (position + heading TOGETHER) by N frames vs the camera bboxes. See _shift_frame_dict.
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
    # Real per-frame timestamps so the Kalman differentiates on true dt (not 1/fps).
    frame_ts = ego_yaw.load_frame_timestamps(frames_csv) if is_android else None
    # ── RELEVANCE REJECTION (note #1): decide which vehicles are probably NOT on our
    # road (persistently big lateral offset, or oncoming) from the SAME world
    # reconstruction the speed path uses, then SKIP them in estimate_world_speeds
    # (no speed -> no plot, no overspeed). HARD filter -- thresholds in Constants.REJECT_*.
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
        method=Constants.DISTANCE_CALCULATION_METHOD,        # changed to "height" instead of "combined" (in every reference)
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

    # ── EGO-CANCEL DIAG (removable): per-frame breakdown of the ego-motion
    # cancellation for chosen vehicles, to classify 1a (depth under-responds) vs
    # 1b (ego term not moving) vs 1c (frame/sign mismatch). Uses the SAME ego_pos
    # /ego_heading the speed run used (any --ego-shift already applied). Flags:
    #   --cancel-diag                       enable
    #   --cancel-diag-id 281,353            restrict to these vehicle ids
    #   --cancel-diag-stationary 281        ids you KNOW are parked -> get a verdict
    #   --cancel-diag-method ground|height  override (defaults to the run's method)
    # Delete this block + ego_cancel_diag.py to remove. ──
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



# def processFrame(yolo_model, world: World, frame, frame_id, lpr_reader: PaddleOCRDetectorReader):
def processFrame(yolo_model, world: World, frame, frame_id):
    t_yolo = time.time()
    # run object tracking on the frame using YOLO
    results = yolo_model.track(
        frame,
        persist=True,
        tracker=Constants.YOLO_TRACKER,
        verbose=False,
        classes=CLASSES,
        conf=CONFIDENCE_LVL,
        device=DEVICE,
        imgsz=Constants.YOLO_IMGSZ,    # for better results use 1984 without half=USE_HALF!!! (but it's slow...)
        half=USE_HALF,
    )
    yolo_time = time.time() - t_yolo
    t_post = time.time()

    detections = results[0]
    vehicle_ids_in_frame = []
    traffic_light_ids_in_frame = []

    
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

            if world.getVehicle(object_id) is None:
                world.addVehicle(object_id, object_type, frame_id)

            v = world.getVehicle(object_id)
            v.updateBoxAndEndFrame(frame_id, bounding_box)
            # Vote on this vehicle's class every frame; resolved after tracking
            # so a one-frame mis-classification can't lock the type (issue #1).
            v.record_classification(object_type, object_conf)

        # if the object is a traffic light
        if Constants.is_traffic_light(object_type):
            traffic_light_ids_in_frame.append(object_id)

            if world.getTrafficLight(object_id) is None:
                world.addTrafficLight(object_id, frame_id)

            trl = world.getTrafficLight(object_id)
            trl.updateBoxAndEnsFrame(frame_id, bounding_box)

    world.registerFrame(frame_id, vehicle_ids_in_frame, traffic_light_ids_in_frame)
    postprocess_time = time.time() - t_post


    """rendered = results[0].plot()
    for vid in vehicle_ids_in_frame:
        v = world.getVehicle(vid)
        if v and v.license_plate:
            x1, y1, x2, y2 = v.bounding_box[v.end_frame]
            cv2.rectangle(rendered, (x1, y1 - 24), (x1 + 160, y1), (0, 200, 0), -1)
            cv2.putText(rendered, v.license_plate, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("frame", rendered)"""

    print(f"frame: {frame_id}")

    return yolo_time, postprocess_time



def run_pipeline(video_path, *, is_simulation=False, yolo_model=None):
    """Run the full detection -> distance -> speed -> overspeed pipeline on one clip.

    This is the importable entry point the backend worker calls; ``main()`` (CLI) is a
    thin wrapper around it, so terminal usage is unchanged. Returns:

        { "annotated_video": <path|None>,    # evidence clip the worker uploads to R2
          "vehicles_csv":    <path>,         # per-vehicle summary
          "violations":      [ { "carId", "calculatedSpeed", "lat", "lon", ... }, ... ] }

    so the worker can upload the evidence clip and POST each violation to the backend.
    Pass ``yolo_model`` to reuse a model already loaded by the caller (the worker loads
    it once and processes many clips, instead of reloading the 100 MB weights per job).
    Tuning flags are still read from ``sys.argv`` (CLI use); when called from the worker
    sys.argv carries none, so the Constants defaults apply.
    """
    # load yolo model (reused across clips when the caller passes one in)
    if yolo_model is None:
        yolo_model = loadYoloModel()
    #lpr_reader = PaddleOCRDetectorReader("israeli_plates.pt")

    # create video handler object
    vh = VideoHandler(video_path)
    frame_height, frame_width = vh.get_frame().shape[:2]

    # ── UNDISTORT (removable): rectify frames so the pinhole pipeline is valid off-axis ──
    # Only for clips recorded WITHOUT on-device distortion correction. Delete this block
    # + the two below (and undistort.py) to remove the experiment entirely.
    undistorter = None
    if "--undistort" in sys.argv:
        import undistort as _undistort
        _report = _undistort.find_report(os.path.dirname(os.path.abspath(video_path)))
        if _report:
            undistorter = _undistort.FrameUndistorter.from_report(_report)
            print(f"[undistort] using calibration report: {_report}")
        else:
            print("[undistort] --undistort set but no calibration report found; running raw")
    # ── end UNDISTORT block ──

    world = World(vh.get_frame_count())

    print(f"number of frames in the video is {vh.get_frame_count()}")

    frame_id = 0
    read_times = []
    yolo_times = []
    postprocess_times = []
    FRAME_SKIP = 1  # process every Nth frame; raise to 3 if still too slow

    # iterates on the video's frames and sends them to process
    start_time = time.time()
    while True:
        # get the current frame
        t_read = time.time()
        frame = vh.get_frame()
        if undistorter is not None:        # UNDISTORT (removable)
            frame = undistorter(frame)
        read_times.append(time.time() - t_read)

        # stop if there are no more frames
        if frame is None:
            read_times.pop()
            break

        yolo_time, postprocess_time = processFrame(yolo_model, world, frame, frame_id)
        yolo_times.append(yolo_time)
        postprocess_times.append(postprocess_time)

        #if frame_id % FRAME_SKIP == 0:
            #processFrame(yolo_model, world, frame, frame_id, lpr_reader)

        # exit loop if 'q' is pressed
        """if cv2.waitKey(1) & 0xFF == ord('q'):
            break"""

        frame_id += 1
        t_read = time.time()
        vh.read_next()
        read_times[-1] += time.time() - t_read
    
    # release video resources
    vh.release()

    print_time(start_time, read_times, yolo_times, postprocess_times, frame_id)

    for v in world.vehicles.values():
        if v._plate_candidates:
            counts = Counter(v._plate_candidates)
            max_count = max(counts.values())
            v.license_plate = next(p for p in reversed(v._plate_candidates) if counts[p] == max_count)

    # ── 1. Output paths & input mode ─────────────────────────────────────────
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    telemetry_csv   = os.path.join(video_dir, "telemetry.csv")
    frames_csv      = os.path.join(video_dir, "frames.csv")
    gyro_csv        = os.path.join(video_dir, "gyro.csv")
    gravity_csv     = os.path.join(video_dir, "gravity.csv")
    gps_csv         = os.path.join(video_dir, "gps.csv")
    linacc_csv      = os.path.join(video_dir, "linacc.csv")   # optional (accel+GPS fusion)
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

    # ── UNDISTORT (removable): match pipeline geometry to the rectified frames ──
    # The frames were warped into K_used (the calibrated K, no distortion), so use it
    # for every downstream projection regardless of what intrinsics.json/FOV gave.
    if undistorter is not None and undistorter.K_used is not None:
        K = undistorter.K_used
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        print(f"[undistort] intrinsics overridden to rectified K: "
              f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    # ── end UNDISTORT block ──

    # ── 6. Distance (clean-room ground-plane estimator) ──────────────────────
    # Ground-plane distance from: gravity-derived plane normal + a GPS-anchored
    # dense-optical-flow factor graph that self-calibrates the camera height (no
    # manual calibration, no known camera height, no known vehicle size). Writes
    # horizontal ground range into vehicle.dist_per_frame, same sink as before.
    #estimate_ground_distances(
    #    world, video_path,
    #    fx=fx, fy=fy, cx=cx, cy=cy,
    #    image_width=frame_width, image_height=frame_height,
    #    sensor_dir=video_dir, is_android=is_android)
    
    estimateDistance(
        world,
        fx, fy, cx, cy,
        Constants.DEFAULT_CAMERA_HEIGHT_M,
    )

    smooth_distances(world, Constants.SMOOTH_WINDOW, Constants.POLYORDER)

    # ── 7. Speed (world-frame reconstruction + Kalman) ───────────────────────
    # When --undistort rectified the frames, the warp leaves an invalid band inside
    # the frame; pass the undistorter's usable rectangle so the edge-drop gate uses
    # that inner border instead of the raw frame edge.
    valid_roi = undistorter.valid_roi if undistorter is not None else None
    run_speed_estimation(
        world, fps, fx, fy, cx, cy,
        is_simulation=is_simulation, is_android=is_android,
        telemetry_csv=telemetry_csv, frames_csv=frames_csv,
        gyro_csv=gyro_csv, gravity_csv=gravity_csv, gps_csv=gps_csv,
        linacc_csv=linacc_csv,
        image_width=frame_width, image_height=frame_height,
        valid_roi=valid_roi)

    # ── 7.5 Overspeed ────────────────────────────────────────────────────────
    # Flag tracked vehicles whose ESTIMATED speed (step 7, m/s) exceeds the legal
    # limit. Android-only: it uses the ego GPS track as the limit-location proxy
    # (we have no GPS for the other cars on the road). The speed-limit lookups hit
    # the live Overpass API, so this is best-effort -- a network/parse failure must
    # not throw away everything steps 1-7 just computed. Tunables:
    #   --no-overspeed                skip this step entirely
    #   --overspeed-margin <km/h>     how far over the limit before flagging
    #                                 (default OVERSPEED_MARGIN_KMH; absorbs the
    #                                 estimator's tendency to over-estimate)
    events = []   # OverspeedEvent list -> turned into the returned violations below
    if is_android and "--no-overspeed" not in sys.argv:
        try:
            margin = float(_cli_value("--overspeed-margin", overspeed.OVERSPEED_MARGIN_KMH))
            ego_track = overspeed.build_ego_track(frames_csv, gps_csv)
            events = overspeed.flag_overspeed_vehicles(world, ego_track, margin_kmh=margin)
            report = overspeed.format_overspeed_report(events)
            print("\n[overspeed] " + report)
            txt_path = os.path.join(video_dir, f"{video_name}_overspeed.txt")
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(report + "\n")
            csv_path = os.path.join(video_dir, f"{video_name}_overspeed.csv")
            overspeed.write_overspeed_csv(events, csv_path)
            print(f"[overspeed] saved -> {txt_path}\n[overspeed] saved -> {csv_path}")
        except Exception as e:
            print(f"[overspeed] skipped (lookup/parse failed): {e}")

    # ── 8. Outputs ───────────────────────────────────────────────────────────
    draw_vehicle_plots.plot_all_vehicles(
        world, video_dir, video_name,
        telemetry_csv=telemetry_csv if is_simulation else None)

    # Ego speed plot: the series the world-frame estimator ACTUALLY uses for ego
    # motion. In sim that's telemetry. On Android the estimator dead-reckons
    # position from the accelerometer+GPS FUSED speed (ego_speed_fused), so plot
    # that as the primary trace -- with the raw GPS-only speed overlaid so the
    # graph shows how much fusion changed the ego speed (and whether it engaged).
    ego_speed: dict[int, float] = {}
    ego_source = ""
    ego_speed_raw: dict[int, float] | None = None
    raw_source = ""
    if is_simulation and os.path.exists(telemetry_csv):
        ego_speed, ego_source = ego_yaw.ego_speed_from_telemetry(telemetry_csv), "telemetry"
    elif is_android:
        gps_speed = ego_yaw.ego_speed_from_android(frames_csv, gps_csv)
        fused_speed = ego_yaw.ego_speed_fused(frames_csv, gps_csv, linacc_csv, gravity_csv)
        if fused_speed:  # fusion engaged -> it's what the estimator uses; GPS is the overlay
            ego_speed, ego_source = fused_speed, "accelerometer+GPS fused"
            ego_speed_raw, raw_source = gps_speed, "GPS only"
        else:            # no linacc / fusion unavailable -> estimator falls back to GPS
            ego_speed, ego_source = gps_speed, "GPS"
    draw_vehicle_plots.plot_ego_speed(ego_speed, video_dir, video_name, source=ego_source,
                                      ego_speed_raw=ego_speed_raw, raw_source=raw_source)

    vehicles_csv = os.path.join(video_dir, f"{video_name}_vehicles.csv")
    distanceLogger.export_vehicle_summary(world, vehicles_csv)

    # Annotated video: source footage + per-vehicle bbox/id/speed/distance overlay,
    # plus the ego speed (same series plotted above) as a banner at the top. This is the
    # evidence clip the worker uploads to R2 and references from each violation.
    annotated_path = os.path.join(video_dir, f"{video_name}_annotated.mp4")
    annotated_video.render_annotated_video(
        world, video_path, annotated_path, fps=fps, ego_speed=ego_speed)

    # ── 9. Build the violations result (one per flagged vehicle) ─────────────
    # carId is the recognized plate when LPR is enabled; otherwise a stable per-track
    # placeholder so the record is still linkable to the annotated clip.
    violations = []
    for e in events:
        vehicle = world.getVehicle(e.vehicle_id)
        plate = getattr(vehicle, "license_plate", None) if vehicle else None
        violations.append({
            "vehicle_id":      e.vehicle_id,
            "carId":           plate or f"vehicle-{e.vehicle_id}",
            "calculatedSpeed": e.est_speed_kmh,
            "lat":             e.lat,
            "lon":             e.lon,
            "speed_limit_kmh": e.speed_limit_kmh,
            "over_by_kmh":     e.over_by_kmh,
        })

    return {
        "annotated_video": annotated_path,
        "vehicles_csv":    vehicles_csv,
        "violations":      violations,
    }



def main():
    """CLI entry point: ``python main.py <video_path> [flags]`` -- unchanged behaviour."""
    video_path = sys.argv[1]
    is_simulation = "--simulation" in sys.argv
    result = run_pipeline(video_path, is_simulation=is_simulation)
    n = len(result.get("violations", []))
    print(f"\n[main] pipeline done -- {n} violation(s) detected; "
          f"annotated video: {result.get('annotated_video')}")
    return result


# the main function of the program
if __name__ == "__main__":
    main()