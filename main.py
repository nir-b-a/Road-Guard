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

# load yolov8 model and set it to YOLO_MODEL
def loadYoloModel():
    print(f"Loading YOLO11x on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    try:
        model = YOLO(Constants.YOLO_VERSION)
        model.to(DEVICE)
        return model
    except Exception as e:
        print(f"Error occurred: {e}")


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


def run_speed_estimation(world: World, fps: float,
                         fx: float, fy: float, cx: float, cy: float,
                         *,
                         is_simulation: bool, is_android: bool,
                         telemetry_csv: str, frames_csv: str,
                         gyro_csv: str, gravity_csv: str, gps_csv: str,
                         linacc_csv: str = "",
                         image_width: int = 0, image_height: int = 0):
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

    print(f"[main] world-frame speed: smoother={smoother}, "
          f"lat_sign={lat_sign}, heading_sign={heading_sign}, lateral_ref={lateral_ref}, "
          f"wide_angle_reweight={wide_angle_reweight}, edge_gate={edge_gate}")
    # Real per-frame timestamps so the Kalman differentiates on true dt (not 1/fps).
    frame_ts = ego_yaw.load_frame_timestamps(frames_csv) if is_android else None
    estimate_world_speeds(
        world, ego_pos, ego_heading, fps,
        fx=fx, fy=fy, cx=cx, cy=cy,
        frame_ts=frame_ts,
        method="height",        # changed to "height" instead of "combined" (in every reference)
        smoother=smoother,
        lat_sign=lat_sign,
        lateral_ref=lateral_ref,
        wide_angle_reweight=wide_angle_reweight,
        edge_gate=edge_gate,
        image_width=image_width, image_height=image_height,
        camera_height_m=Constants.DEFAULT_CAMERA_HEIGHT_M,
        min_track_seconds=Constants.MIN_TRACK_SECONDS,
    )



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



def main():
    
    # load yolo model
    yolo_model = loadYoloModel()
    #lpr_reader = PaddleOCRDetectorReader("israeli_plates.pt")

    # get the video path from command line argumants
    video_path = sys.argv[1]
    is_simulation = "--simulation" in sys.argv

    # create video handler object
    vh = VideoHandler(video_path)
    frame_height, frame_width = vh.get_frame().shape[:2]

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

    distanceLogger.export_vehicle_summary(
        world, os.path.join(video_dir, f"{video_name}_vehicles.csv"))

    # Annotated video: source footage + per-vehicle bbox/id/speed/distance overlay.
    annotated_video.render_annotated_video(
        world, video_path,
        os.path.join(video_dir, f"{video_name}_annotated.mp4"),
        fps=fps)




# the main function of the program
if __name__ == "__main__":
    main()