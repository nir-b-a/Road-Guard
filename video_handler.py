import json
import shutil
import subprocess

import cv2
import numpy as np

import cloud_env   # ensures cv2 GUI calls are neutralized on a headless/Colab runtime

DEFAULT_FRAME_SIZE = (640, 640)


class VideoHandler:
    """Sequential frame reader with an automatic fallback decoder.

    Fast path: cv2.VideoCapture (handles ordinary H.264/H.265 mp4s with no overhead).
    Fallback: when cv2 cannot OPEN or DECODE a file (exotic dashcam codecs, some
    VP9/AV1 streams, broken headers), we stream raw frames from the system `ffmpeg`
    instead of pre-transcoding the whole clip to disk first. ffmpeg decodes more
    formats than cv2's bundled build, the decode is decode-ONLY (no re-encode), and
    it overlaps with downstream inference -- so there's no slow upfront conversion
    pass. Reads are sequential only (the pipeline never seeks).
    """

    def __init__(self, video_path, frame_size=DEFAULT_FRAME_SIZE):
        self.video_path = video_path
        self.frame_size = frame_size
        self._backend = None          # "cv2" | "ffmpeg"
        self._proc = None             # ffmpeg subprocess (ffmpeg backend only)
        self._w = self._h = None      # frame dims (ffmpeg backend only)

        # ── Fast path: let OpenCV try first ───────────────────────────────────
        self.cap = cv2.VideoCapture(video_path)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._backend = "cv2"
                self.ret, self.frame = ret, frame
                self.fps = self._sane_fps(self.cap.get(cv2.CAP_PROP_FPS))
                self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                return

        # OpenCV failed to open or decode -> fall back to streaming via ffmpeg.
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._open_ffmpeg_pipe(video_path)

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sane_fps(fps) -> float:
        """Some codecs (VFR mp4s, AV1) report 0 or garbage FPS; fall back to 30."""
        if not fps or fps <= 1 or fps > 240:
            print(f"[VideoHandler] WARNING: unreliable FPS={fps}, defaulting to 30")
            return 30.0
        return float(fps)

    @staticmethod
    def _probe(video_path) -> tuple[int, int, float, int]:
        """ffprobe -> (width, height, fps, frame_count). frame_count is best-effort."""
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
             "-of", "json", video_path],
            capture_output=True, text=True)
        streams = json.loads(out.stdout or "{}").get("streams", [])
        if not streams:
            raise ValueError(f"ffprobe found no video stream in {video_path!r}")
        info = streams[0]
        w, h = int(info["width"]), int(info["height"])
        num, _, den = (info.get("avg_frame_rate") or "0/1").partition("/")
        fps = (float(num) / float(den)) if den and float(den) else 0.0
        nframes = int(info["nb_frames"]) if str(info.get("nb_frames", "")).isdigit() else 0
        if not nframes and info.get("duration") not in (None, "N/A") and fps:
            nframes = int(float(info["duration"]) * fps)     # estimate from duration
        return w, h, fps, nframes

    def _open_ffmpeg_pipe(self, video_path) -> None:
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise ValueError(
                f"OpenCV could not open {video_path!r} and ffmpeg/ffprobe are not installed "
                f"to fall back on.")
        w, h, fps, nframes = self._probe(video_path)
        self._w, self._h = w, h
        self.fps = self._sane_fps(fps)
        self._frame_count = nframes
        self._frame_bytes = w * h * 3
        self._backend = "ffmpeg"
        # Decode-only raw BGR stream straight to stdout (no re-encode, no temp file).
        self._proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-i", video_path,
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 8)
        print(f"[VideoHandler] OpenCV could not decode this file; streaming via ffmpeg "
              f"({w}x{h} @ {self.fps:.0f}fps, ~{nframes or '?'} frames)")
        self.ret, self.frame = self._read_pipe_frame()
        if not self.ret:
            raise ValueError(f"Failed to read first frame from {video_path!r} via ffmpeg")

    def _read_pipe_frame(self):
        """Read exactly one raw BGR frame from the ffmpeg pipe -> (ret, frame)."""
        raw = self._proc.stdout.read(self._frame_bytes)
        if not raw or len(raw) < self._frame_bytes:
            return False, None
        # bytearray makes the underlying buffer writable (ultralytics/cv2 may draw in place).
        frame = np.frombuffer(bytearray(raw), np.uint8).reshape(self._h, self._w, 3)
        return True, frame

    # ──────────────────────────────────────────────────────────────────────────
    # return current frame
    def get_frame(self):
        return self.frame

    # reads the next frame, returns true if successful, false when video ends
    def read_next(self):
        if self._backend == "cv2":
            self.ret, self.frame = self.cap.read()
        else:
            self.ret, self.frame = self._read_pipe_frame()
        return self.ret

    # return the total number of frames in the video (best-effort on the ffmpeg backend)
    def get_frame_count(self):
        if self._backend == "cv2":
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self._frame_count

    # return the frame rate of the video (frames per second)
    def get_fps(self):
        return self.fps

    # return the frame at a given index (random access; cv2 backend only)
    def get_frame_at(self, index):
        if self._backend != "cv2":
            raise NotImplementedError(
                "get_frame_at requires random access, unavailable on the ffmpeg streaming "
                "backend (the pipeline only reads sequentially).")
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
        if self.cap is not None:
            self.cap.release()
        if self._proc is not None:
            try:
                self._proc.stdout.close()
            except OSError:
                pass
            self._proc.terminate()
            self._proc = None
        # Safe everywhere now: on a headless/Colab runtime cloud_env has turned
        # destroyAllWindows into a no-op, so this never crashes without a display.
        if not cloud_env.HEADLESS:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
