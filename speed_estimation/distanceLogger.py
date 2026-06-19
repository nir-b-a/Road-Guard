import csv
from Objects.World import World
from Constants import class_name

def export_distances(world: World, max_frame, output_file="distances.csv", vehicle_id: int | None = None):
    v = world.getVehicle(vehicle_id) if vehicle_id is not None else world.getVehicle(1)
    if v is None:
        print(f"[distanceLogger] no vehicle found for export_distances (id={vehicle_id})")
        return

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "estimated_distance"])

        for frame in range(0, max_frame):
            dist = v.dist_per_frame.get(frame, 0.0)
            writer.writerow([frame, dist])


def export_speeds(speeds_by_vehicle: dict[int, dict[int, float]],
                  max_frame: int,
                  output_file: str = "speeds.csv",
                  vehicle_id: int | None = None):
    """
    Export per-frame speed (m/s) to a CSV.

    speeds_by_vehicle: output of speed_estimator.estimate_speeds(...).

    If vehicle_id is given, export only that vehicle.
    Otherwise export the vehicle with the most speed samples.
    """
    if not speeds_by_vehicle:
        print(f"[distanceLogger] no speeds to export -> {output_file}")
        return

    if vehicle_id is not None:
        if vehicle_id not in speeds_by_vehicle:
            print(f"[distanceLogger] vehicle id={vehicle_id} has no speeds")
            return
        speeds = speeds_by_vehicle[vehicle_id]
    else:
        vehicle_id = max(speeds_by_vehicle, key=lambda v: len(speeds_by_vehicle[v]))
        speeds = speeds_by_vehicle[vehicle_id]

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "estimated_speed_mps"])
        for frame in range(max_frame):
            writer.writerow([frame, speeds.get(frame, 0.0)])


def export_persisted_speeds(world: World, max_frame: int,
                            output_file: str = "speeds.csv",
                            vehicle_id: int | None = None):
    """
    Export the speed series PERSISTED on the Vehicle objects (issue #2):
    speed_per_frame + speed_std_per_frame, written by speed_estimator.estimate_world_speeds.

    Columns: frame, estimated_speed_mps, speed_std_mps.

    If vehicle_id is given, export that vehicle; otherwise the vehicle with the
    most persisted speed samples (mirrors export_speeds' selection rule). Frames
    with no estimate are written as 0.0 so the CSV is dense over [0, max_frame),
    matching export_speeds' convention.
    """
    candidates = {vid: v for vid, v in world.vehicles.items() if v.speed_per_frame}
    if not candidates:
        print(f"[distanceLogger] no persisted speeds to export -> {output_file}")
        return

    if vehicle_id is not None:
        v = candidates.get(vehicle_id)
        if v is None:
            print(f"[distanceLogger] vehicle id={vehicle_id} has no persisted speeds")
            return
    else:
        vehicle_id = max(candidates, key=lambda i: len(candidates[i].speed_per_frame))
        v = candidates[vehicle_id]

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "estimated_speed_mps", "speed_std_mps"])
        for frame in range(max_frame):
            writer.writerow([frame,
                             v.speed_per_frame.get(frame, 0.0),
                             v.speed_std_per_frame.get(frame, 0.0)])


def export_all_bboxes(world: World, max_frame, output_file="all_bboxes.csv", vehicle_id: int | None = None):
    """
    Export detections across every frame.
    If vehicle_id is given, exports only that vehicle (simulation mode).
    Otherwise exports all vehicles.
    Used by calibrate_distance.py.
    """
    vehicles_to_export = (
        {vehicle_id: world.vehicles[vehicle_id]}
        if vehicle_id is not None and vehicle_id in world.vehicles
        else world.vehicles
    )

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vehicle_id", "x1", "y1", "x2", "y2"])

        for frame in range(max_frame):
            for vid, vehicle in vehicles_to_export.items():
                bbox = vehicle.bounding_box.get(frame)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    continue
                writer.writerow([frame, vid, x1, y1, x2, y2])


def export_vehicle_summary(world: World, output_file: str = "vehicles.csv"):
    """
    One row per DETECTED vehicle: id, resolved class, first/last detected frame,
    number of real detections, whether it received a speed estimate, and the bbox
    at its first detected frame.

    CSV (not JSON/plain text): the data is flat and tabular, so CSV is both
    human-readable in a spreadsheet and trivially machine-parsable downstream.
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "vehicle_id", "resolved_class", "class_id",
            "first_frame", "last_frame", "num_detections", "has_speed_estimate",
            "first_bbox_x1", "first_bbox_y1", "first_bbox_x2", "first_bbox_y2",
        ])
        for vid, vehicle in sorted(world.vehicles.items()):
            real = sorted(fr for fr, b in vehicle.bounding_box.items() if b != (0, 0, 0, 0))
            first_frame = real[0] if real else vehicle.start_frame
            last_frame  = real[-1] if real else vehicle.end_frame
            x1, y1, x2, y2 = vehicle.bounding_box.get(first_frame, (0, 0, 0, 0))
            writer.writerow([
                vid, class_name(vehicle.vehicle_type), int(vehicle.vehicle_type),
                first_frame, last_frame, len(real), bool(vehicle.speed_per_frame),
                x1, y1, x2, y2,
            ])