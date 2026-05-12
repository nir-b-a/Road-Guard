import json


class FrameLogger:
    def __init__(self, path):
        path += ".jsonl"
        self.file = open(path, "w")

    def log_frame(self, frame_id, vehicle_ids, traffic_light_ids, number_of_bounding_boxes):
        entry = {
            "frame": frame_id,
            "vehicles": vehicle_ids,
            "traffic_lights": traffic_light_ids,
            "bounding boxes count": number_of_bounding_boxes
        }

        self.file.write(json.dumps(entry) + "\n")

    def close(self):
        self.file.close()