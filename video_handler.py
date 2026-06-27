import cv2

DEFAULT_FRAME_SIZE = (640, 640)


class VideoHandler:
    def __init__(self, video_path, frame_size=DEFAULT_FRAME_SIZE):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_size = frame_size

        if not self.cap.isOpened():
            raise ValueError("Failed to open video")

        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read first frame")
        
        # Frame rate. Some codecs (variable frame rate MP4s, AV1) report 0 or
        # garbage values here; fall back to 30 and warn so the caller knows.
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 1 or self.fps > 240:
            print(f"[VideoHandler] WARNING: unreliable FPS={self.fps}, defaulting to 30")
            self.fps = 30.0

        

    # return current frame
    def get_frame(self):
        return self.frame

    # reads the next frame, returns true if successful, false when video ends
    def read_next(self):
        self.ret, self.frame = self.cap.read()
        return self.ret
    
    # return the total number of frame in the video
    def get_frame_count(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # return the frame rate of the video (frames per second)
    def get_fps(self):
        return self.fps
    
    # return the frame in a given index
    def get_frame_at(self, index):
        total = self.get_frame_count()
        if index < 0 or index >= total:
            raise IndexError(f"Frame index {index} out of range (0–{total - 1})")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame at index {index}")

        return frame


    # releasing memory (and destroy any OpenCV windows)
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
