import cv2
# defining the classes we want to identify
# we track car, motorcycle, bus, truck and traffic light. the rest of the built-in classes are ignored
CLASSES = [2, 3, 5, 7, 9]

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    model.trackers = None

    # reading all the frames of the video
    while cap.isOpened():
        ret, frame = cap.read()

        # if frame not valid
        if not ret:
            break

        # analyzing each frame
        results = model.track(frame, persist=True, verbose=False, classes=CLASSES)

        # comment in the next line for detections log to appear in terminal
        # results = model(frame, persist=True)

        annotated = results[0].plot()
        boxes = results[0].boxes

        # iterate through all boxes
        if boxes is not None:
            for box in boxes.xyxy:
                # get 4 edges of box
                x1, y1, x2, y2 = box

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # mark center of box with a dot
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

        # showing the results in a video player
        cv2.imshow("frame", annotated)

        # can quit at any time if press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
