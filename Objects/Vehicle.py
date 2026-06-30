from dataclasses import dataclass, field

@dataclass
class Vehicle:

    id: int
    vehicle_type: int
    start_frame: int
    end_frame: int

    bounding_box: dict[int, tuple[int, int, int, int]] = field(default_factory=dict, repr=False)  # (frame_id) -> (x1, y1, x2, y2)

    speed_per_frame: dict[int, float] = field(default_factory=dict, repr=False)
    speed_std_per_frame: dict[int, float] = field(default_factory=dict, repr=False)
    dist_per_frame: dict[int, float] = field(default_factory=dict, repr=False)
    class_score: dict[int, float] = field(default_factory=dict, repr=False)

    license_plate: str | None = field(default=None, repr=False)


    def updateBoxAndEndFrame(self, frame_id: int, bbox: tuple[int, int, int, int]):
        for missing in range(self.end_frame + 1, frame_id):
            self.bounding_box.setdefault(missing, (0, 0, 0, 0))

        self.bounding_box[frame_id] = bbox
        self.end_frame = frame_id

    def record_classification(self, class_id: int, confidence: float) -> None:
        if confidence is None:
            confidence = 1.0
        self.class_score[class_id] = self.class_score.get(class_id, 0.0) + confidence

    def resolve_vehicle_type(self) -> int:
        if self.class_score:
            best = max(self.class_score.values())
            top = [c for c, s in self.class_score.items() if s == best]
            if self.vehicle_type not in top:
                self.vehicle_type = min(top)
        return self.vehicle_type

    def getRelativeFrameBbox(self, relative_frame_id: int):
        return self.bounding_box[relative_frame_id]
    
    def centerInFrame(self, frame_id: int) -> tuple[int, int] | None:
        bbox = self.bounding_box.get(frame_id)
        if bbox is None:
            return None
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    def bottomCenterInFrame(self, frame_id: int) -> tuple[int, int] | None:
        bbox = self.bounding_box.get(frame_id)
        if bbox is None:
            return None
        return ((bbox[0] + bbox[2]) // 2, bbox[3])
    
