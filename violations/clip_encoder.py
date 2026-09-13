"""
clip_encoder -- ONE H.264 encode for an evidence clip, instead of mp4v-then-transcode.

WHAT THIS REPLACES
    Evidence clips used to be written by ``cv2.VideoWriter(..., "mp4v")`` -- MPEG-4 Part 2, a
    1998 codec -- and then re-encoded to H.264 by ffmpeg on the way to R2 (worker.transcode_h264).
    That cost two encodes plus a decode, AND handed x264 an image mp4v had already damaged, so
    bits were spent faithfully reproducing mp4v's blocking artefacts.

    ``ClipWriter`` takes the same BGR frames and pipes them straight into ONE libx264 encode.
    The frames are byte-identical to what cv2.VideoWriter was given -- every box, banner and
    footer is drawn exactly as before -- so only the compression changes, never the picture
    that gets drawn. Resolution and frame rate are untouched for the same reason: the overlay
    text is thin (thickness-1 strokes at scale 0.55), and downscaling would smear the very
    annotations the clip exists to show.

QUALITY
    Defaults target "heavily compressed but still clearly readable", the way a dashcam stores
    footage: CRF 30 at ``-preset slow``. These clips are only ever watched by a human reviewer,
    never re-processed by a model, so the quality bar is legibility rather than fidelity.
    Roughly: CRF 23 (the old default) is near-transparent and large, 30 is comfortably
    watchable, 32+ starts showing blocking on motion.

ROBUSTNESS
    A missing ffmpeg must never lose a real detection, so if ffmpeg cannot be started the
    writer silently falls back to the old ``cv2.VideoWriter`` mp4v path -- degraded, but a clip
    still exists, and worker.transcode_h264 still fixes it up on the way out. ffmpeg is spawned
    in the constructor, BEFORE any frame is written, so that fallback is decided while falling
    back is still free.

Env knobs (shared with worker.py so both paths agree):
    FFMPEG_BIN    ffmpeg binary                                  (default "ffmpeg" on PATH)
    H264_CRF      x264 quality; lower = better/bigger            (default 30)
    H264_PRESET   x264 speed/efficiency preset                   (default slow)
    H264_GOP      keyframe interval in frames                    (default 30, ~1 s at 30 fps)
"""
from __future__ import annotations

import os
import subprocess
import tempfile

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
# Defaults live here and worker.py imports them, so the single-pass path and the legacy
# transcode fallback can never drift apart.
DEFAULT_CRF = os.environ.get("H264_CRF", "30")
DEFAULT_PRESET = os.environ.get("H264_PRESET", "slow")
DEFAULT_GOP = os.environ.get("H264_GOP", "30")


def h264_output_args(*, crf: str | int | None = None, preset: str | None = None,
                     gop: str | int | None = None) -> list:
    """The x264 output flags shared by every evidence-clip encode (no I/O -> unit-testable).

    ``-pix_fmt yuv420p`` because Safari and older Android refuse 4:4:4; ``+faststart`` puts the
    moov atom first so the dashboard can start playing before the clip has fully arrived; ``-an``
    because evidence clips carry no audio.
    """
    return [
        "-c:v", "libx264",
        "-preset", str(preset if preset is not None else DEFAULT_PRESET),
        "-crf", str(crf if crf is not None else DEFAULT_CRF),
        "-g", str(gop if gop is not None else DEFAULT_GOP),
        # No explicit -profile:v: libx264 already picks High for 8-bit yuv420p, and pinning it
        # only breaks otherwise-valid settings (High cannot do CRF 0 lossless, for one).
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
    ]


class ClipWriter:
    """Drop-in replacement for ``cv2.VideoWriter``: ``.write(frame)``, ``.release()``,
    ``.isOpened()``.

    Frames must be contiguous HxWx3 uint8 BGR at the declared size -- exactly what
    cv2.VideoWriter accepts. A wrong-sized frame RAISES rather than being silently dropped:
    cv2 ignores mismatches, but a raw pipe would desync and produce a garbled clip, which is
    far worse than a loud failure the caller already handles.
    """

    def __init__(self, dest: str, fps: float, size, *, fallback_fourcc: str = "mp4v",
                 crf=None, preset=None, gop=None):
        self.dest = dest
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.width, self.height = int(size[0]), int(size[1])
        self._proc = None
        self._cv_writer = None
        self._errlog = None
        self._broken = False
        self._frames = 0

        cmd = [
            FFMPEG_BIN, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", f"{self.fps:g}",
            "-i", "-",
            *h264_output_args(crf=crf, preset=preset, gop=gop),
            dest,
        ]
        try:
            # stderr goes to a FILE, never a pipe: nothing reads it until release(), and a full
            # pipe buffer would deadlock ffmpeg mid-clip.
            self._errlog = tempfile.TemporaryFile()
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                          stdout=subprocess.DEVNULL, stderr=self._errlog)
        except (OSError, ValueError) as e:
            # Overwhelmingly: ffmpeg is not on PATH. Decided before frame 1, so falling back
            # to the old writer costs nothing.
            if self._errlog is not None:
                self._errlog.close()
                self._errlog = None
            self._proc = None
            print(f"[clip] ffmpeg unavailable ({e}) -- writing {fallback_fourcc} instead; "
                  f"the clip will be re-encoded on upload")
            self._open_fallback(fallback_fourcc)

    def _open_fallback(self, fourcc: str) -> None:
        import cv2
        self._cv_writer = cv2.VideoWriter(
            self.dest, cv2.VideoWriter_fourcc(*fourcc), self.fps, (self.width, self.height))

    # -- cv2.VideoWriter surface --------------------------------------------- #
    def isOpened(self) -> bool:  # noqa: N802 -- mirrors cv2's camelCase on purpose
        if self._cv_writer is not None:
            return self._cv_writer.isOpened()
        return self._proc is not None and self._proc.poll() is None

    def write(self, frame) -> None:
        if self._cv_writer is not None:
            self._cv_writer.write(frame)
            self._frames += 1
            return
        if self._broken:
            return
        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            raise ValueError(f"frame {frame.shape[1]}x{frame.shape[0]} does not match the "
                             f"declared {self.width}x{self.height}")
        try:
            self._proc.stdin.write(frame.tobytes() if frame.flags["C_CONTIGUOUS"]
                                   else frame.copy().tobytes())
            self._frames += 1
        except (BrokenPipeError, OSError):
            self._broken = True          # ffmpeg died; release() surfaces why

    def release(self) -> None:
        """Close the encoder. Raises if ffmpeg failed -- callers already treat a clip-render
        exception as 'ship without a clip', which is the correct outcome for a corrupt file."""
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None
            return
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self._read_err()
            raise RuntimeError(f"ffmpeg timed out encoding {self.dest}")
        err = self._read_err()
        if proc.returncode != 0 or not os.path.exists(self.dest) \
                or os.path.getsize(self.dest) == 0:
            raise RuntimeError(f"ffmpeg failed encoding {self.dest} "
                               f"(exit {proc.returncode}): {err[:300]}")

    def _read_err(self) -> str:
        if self._errlog is None:
            return ""
        try:
            self._errlog.seek(0)
            return self._errlog.read().decode("utf-8", "replace").strip()
        except (OSError, ValueError):
            return ""
        finally:
            try:
                self._errlog.close()
            except OSError:
                pass
            self._errlog = None

    # -- context manager, for callers that want it ---------------------------- #
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # Already failing: close the encoder but let the ORIGINAL exception propagate.
            try:
                self.release()
            except Exception:
                pass
            return False
        self.release()
        return False


def open_clip_writer(dest: str, fps: float, size, *, fourcc: str = "mp4v", **kw) -> ClipWriter:
    """Factory mirroring the old ``cv2.VideoWriter(dest, fourcc(*fourcc), fps, size)`` call.

    ``fourcc`` is now only the FALLBACK container used when ffmpeg is missing; the normal path
    is always H.264. Kept in the signature so existing callers that pass it still work.
    """
    return ClipWriter(dest, fps, size, fallback_fourcc=fourcc, **kw)
