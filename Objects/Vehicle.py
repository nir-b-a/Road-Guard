from dataclasses import dataclass, field

@dataclass
class Vehicle:

    # THE ID MUST BE UNIQUE!!!
    # The constructor will ask for (id, vehicle type, start frame, end frame (which at initiation is the same as start frame))
    id: int
    vehicle_type: int       
    start_frame: int
    # Last frame the vehicle was detected in. THE CONSTRACTOR DOES NOT SET A VALUE TO IT, A VALUE WILL BE SET AUTOMATICALLY BY SOME FUNCTION (in this class).
    end_frame: int

    # A dictionary, the key represent a frame the car was detected in and the value is a tuple of the bounding box. If the car was tracked but not need all value will be set to 0.
    bounding_box: dict[int, tuple[int, int, int, int]] = field(default_factory=dict, repr=False)        # (frame_id) -> (x1, y1, x2, y2)

    # a speed value for each frame. May not be needed so as for now, it's in comment.
    speed_per_frame: dict[int, int] = field(default_factory=dict, repr=False)               # The constructor won't ask for a speed_per_frame dict.
    #license_plate: int = field(repr=False)


    def updateBoxAndEndFrame(self, frame_id: int, bbox: tuple[int, int, int, int]):
        self.bounding_box[frame_id] = bbox
        # Fill gaps between last seen frame and current frame (vehicle temporarily left detection)
        for missing_frame in range(self.end_frame + 1, frame_id):
            self.bounding_box.setdefault(missing_frame, (0, 0, 0, 0))
        self.end_frame = frame_id

    # To easily get the bounding box of a vehicle by his relative frame (for example, (1) -> the bounding box from the first frame he appears in)
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
    
