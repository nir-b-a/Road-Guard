import cv2
# defining the classes we want to identify
# we track car, motorcycle, bus, truck and traffic light. the rest of the built-in classes are ignored
CLASSES = [2, 3, 5, 7, 9]
FRAME_SIZE = (640, 640)

# extracts object center points from detection results.
def get_centers(results, annotated=None, draw=False):
    boxes = results[0].boxes
    centers = []

    if boxes is None:
        return centers, annotated

    for box in boxes.xyxy:
        x1, y1, x2, y2 = box

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        centers.append((cx, cy))

        if draw and annotated is not None:
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    return centers, annotated


def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    model.trackers = None

    # reading all the frames of the video
    while cap.isOpened():
        ret, frame = cap.read()

        # if frame not valid
        if not ret:
            break
            # editing the size of the frame
        frame = cv2.resize(frame, FRAME_SIZE)

        # analyzing each frame
        results = model.track(frame, persist=True, verbose=False, classes=CLASSES, conf=0.5)

        # comment in the next line for detections log to appear in terminal
        # results = model(frame, persist=True)

        rendered_frame = results[0].plot()

        centers, rendered_frame = get_centers(results, rendered_frame, draw=True)

        # showing the results in a video player
        cv2.imshow("frame", rendered_frame)

        # can quit at any time if press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
