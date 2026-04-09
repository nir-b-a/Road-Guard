import cv2

DEFAULT_FRAME_SIZE = (640, 640)


class VideoHandler:
    def __init__(self, video_path, frame_size=DEFAULT_FRAME_SIZE):
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError("Failed to open video")

        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read first frame")

        self.frame_size = frame_size


    def get_frame(self):
        return self.frame

        # reads the next frame
    def read_next(self):
        self.ret, self.frame = self.cap.read()

        # releasing memory
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
