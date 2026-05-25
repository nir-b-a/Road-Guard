import csv
from Objects.World import World

def export_distances(world: World, max_frame, output_file="distances.csv"):
    v = world.getVehicle(1)
    if v is None:
        return

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "estimated_distance"])

        for frame in range(0, max_frame):
            dist = v.dist_per_frame.get(frame, 0.0)
            writer.writerow([frame, dist])


def export_all_bboxes(world: World, max_frame, output_file="all_bboxes.csv"):
    """
    Export every detection from every vehicle across every frame.
    Used by calibrate_distance.py, which matches detections to the known
    CARLA target using 3D projection (no reliance on stable YOLO IDs).
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "vehicle_id", "x1", "y1", "x2", "y2"])

        for frame in range(max_frame):
            for vid, vehicle in world.vehicles.items():
                bbox = vehicle.bounding_box.get(frame)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                    continue
                writer.writerow([frame, vid, x1, y1, x2, y2])