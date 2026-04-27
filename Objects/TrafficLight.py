from dataclasses import dataclass, field

@dataclass
class TrafficLight:

    # The constructor needs to get (id, start_frame, end frame)
    id: int
    start_frame: int
    end_frame: int

    bounding_box: dict[int, tuple[int, int, int, int]] = field(default_factory=dict, repr=False)

    color_state: dict[int, int] = field(default_factory=dict, repr=False)


    def updateBoxAndEnsFrame(self, frame_id: int, bbox: tuple[int, int, int, int]):
        self.bounding_box[frame_id] = bbox      
        self.end_frame = frame_id

    def getRelativeFrameBbox(self, relative_frame_id: int):
        return self.bounding_box[relative_frame_id]
    
    def centerInFrame(self, frame_id: int) -> tuple[int, int] | None:
        bbox = self.bounding_box[frame_id]

        if bbox is None:
            return None
        
        return((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    

    # we need a way to set the color/state of the traffic light in each frame.
    #def setColorStateInFrame(self, color):