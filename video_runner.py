# video_runner.py
from video_handler import VideoHandler
import cv2

CLASSES = [2, 3, 5, 7, 9]
CONFIDENCE_LVL = 0.5


 # this function reads video frames and applies the YOLO model on each frame
def process_video(video_path, model):

    # initialize video handler for reading frames
    vh = VideoHandler(video_path)

    while True:
        # get the current frame
        frame = vh.get_frame()

        # stop if there are no more frames
        if frame is None:
            break

        # run object tracking on the frame using YOLO
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=CLASSES,
            conf=CONFIDENCE_LVL
        )

        rendered = results[0].plot()

        # display the processed frame
        cv2.imshow("frame", rendered)

        # exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        vh.read_next()

    # release video resources
    vh.release()
