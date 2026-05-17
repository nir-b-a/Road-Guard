import sys
import cv2
from video_handler import VideoHandler
from ultralytics import YOLO
from ultralytics.engine.results import Results
from Constants import DetectClass
from Objects.World import World

from frameLogger import FrameLogger
from line_crossing.line_detector import detect_solid_lines
from line_crossing.crossing_detector import check_crossing


YOLO_MODEL = None
CLASSES = DetectClass.Detection_Classes
CONFIDENCE_LVL = 0.5


# load yolov8 model and set it to YOLO_MODEL (for now we'll use model x)
def loadYoloModel():
    #global YOLO_MODEL
    try:
        return YOLO("yolov8m.pt")
    except Exception as e:
        print(f"Error occurred: {e}")


"""Is the yolov8 model uses some tracking algorithm?"""
def processFrame(yolo_model, world: World, frame, frame_id, frame_logger: FrameLogger):
    # run object tracking on the frame using YOLO
    results = yolo_model.track(
        frame,
        persist=True,       #tracker="bytetrack.yaml",  # or "botsort.yaml" - claude
        # change to False later
        verbose=True,
        classes=CLASSES,
        conf=CONFIDENCE_LVL
        #iou=0.5,               # IoU threshold for NMS (non-max suppression)
    )

    detections = results[0]
    vehicle_ids_in_frame = []
    traffic_light_ids_in_frame = []

    """ Remove when not testing"""
    logger_bounding_boxes = 0
    
    for box in detections.boxes:

        object_id = int(box.id.item())
        object_type = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bounding_box = (x1, y1, x2, y2)

        logger_bounding_boxes += 1

        # if the object is a vehicle
        if object_type in DetectClass.Vehicle_Classes:
            vehicle_ids_in_frame.append(object_id)

            # create a new Vehicle object
            if world.getVehicle(object_id) is None:
                world.addVehicle(object_id, object_type, frame_id)

            v = world.getVehicle(object_id)
            v.updateBoxAndEndFrame(frame_id, bounding_box)

        # if the object is a traffic light
        if object_type in DetectClass.Traffic_Light_Class:
            traffic_light_ids_in_frame.append(object_id)

            # create new traffic light object
            if world.getTrafficLight(object_id) is None:
                world.addTrafficLight(object_id, frame_id)

            trl = world.getTrafficLight(object_id)
            trl.updateBoxAndEnsFrame(frame_id, bounding_box)

    world.registerFrame(frame_id, vehicle_ids_in_frame, traffic_light_ids_in_frame)
    world.detected_lines[frame_id] = detect_solid_lines(frame)

    frame_logger.log_frame(frame_id, vehicle_ids_in_frame, traffic_light_ids_in_frame, logger_bounding_boxes, world.detected_lines.get(frame_id, {}))


    """rendered = results[0].plot()

    # display the processed frame
    cv2.imshow("frame", rendered)"""



def main():

    """ Just for testing"""
    yolo_logger = FrameLogger('test_1_yolo')
    world_logger = FrameLogger('test_1_world')
    
    # load yolov8 model
    yolo_model = loadYoloModel()

    # get the video path from command line argumants
    video_path = sys.argv[1]

    # create video handler object
    vh = VideoHandler(video_path)

    world = World(vh.get_frame_count())

    print(f"number of frames in the video is {vh.get_frame_count()}")

    frame_id = 0

    # iterates on the video's frames and sends them to process
    while True:
        # get the current frame
        frame = vh.get_frame()

        # stop if there are no more frames
        if frame is None:
            break

        """ Remove frame_logger when not testing"""
        processFrame(yolo_model, world, frame, frame_id, yolo_logger)

        # exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1
        vh.read_next()
    
    # release video resources
    vh.release()

    for vehicle in world.vehicles.values():
        violation_frame = check_crossing(vehicle, world.detected_lines)
        if violation_frame is not None:
            print(f"[VIOLATION] Vehicle {vehicle.id} crossed a solid line at frame {violation_frame}")
            # TODO: POST /api/internal/violation to report to backend

    """ Part of the testing remove later"""
    for frame_counter in range(world.frame_count):
        bbox_counter = len(world.objects_in_frame[frame_counter].vehicle_ids) + len(world.objects_in_frame[frame_counter].traffic_light_ids)
        world_logger.log_frame(frame_counter, world.objects_in_frame[frame_counter].vehicle_ids, world.objects_in_frame[frame_counter].traffic_light_ids, bbox_counter)




# the main function of the program
if __name__ == "__main__":
    main()