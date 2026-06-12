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
    license_plate: str | None = field(default=None, repr=False)
    _plate_candidates: list = field(default_factory=list, repr=False)
    _lpr_last_read_frame: int = field(default=-1, repr=False)


    def updateBoxAndEndFrame(self, frame_id: int, bbox: tuple[int, int, int, int]):
        self.bounding_box[frame_id] = bbox
        self.end_frame = frame_id               # IMPORTANT! If frame_id is relative (for example we process the 300th frame but it's the first time this object appears,
                                                #   so the frame_id the function gets is 1) then we need to change the logic here and set the right frame_id to end_frame.

        # if a vehicle is missing in some frames and reappears later, we set its bounding box to (0,0,0,0)
        index = self.start_frame
        for index in range(self.end_frame):
            self.bounding_box.setdefault(index, (0, 0, 0, 0))

    # To easily get the bounding box of a vehicle by his relative frame (for example, (1) -> the bounding box from the first frame he appears in)
    def getRelativeFrameBbox(self, relative_frame_id: int):
        return self.bounding_box[relative_frame_id]
    
    def needs_lpr(self, min_area: int, frame_id: int, read_interval: int = 40) -> bool:
        if not self.bounding_box:
            return False
        bbox = self.bounding_box.get(self.end_frame, (0, 0, 0, 0))
        if bbox == (0, 0, 0, 0):
            return False
        x1, y1, x2, y2 = bbox
        if (x2 - x1) * (y2 - y1) < min_area:
            return False
        return frame_id - self._lpr_last_read_frame >= read_interval

    def centerInFrame(self, frame_id: int) -> tuple[int, int] | None:
        bbox = self.bounding_box[frame_id]

        if bbox is None:
            return None
        
        return((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
