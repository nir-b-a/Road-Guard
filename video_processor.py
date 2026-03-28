import cv2

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    # reading all the frames of the video
    while cap.isOpened():
        ret, frame = cap.read()

        # if frame not valid
        if not ret:
            break

        # analyzing each frame
        results = model.track(frame, persist=True, verbose=False)

        # comment in the next line for detections log to appear in terminal
       # results = model(frame, persist=True)

        # showing the results in a video player
        annotated = results[0].plot()
        cv2.imshow("frame", annotated)

        # can quit at any time if press q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()