"""
clip_extract -- cut the short evidence VIDEO around each violation (the thing we ship back).

The contract with the backend is: analyse one full clip, then return ONLY the moments where a
violation occurred -- a window of ``PRE_ROLL_SEC`` before the incident to ``POST_ROLL_SEC`` after it
-- one sub-clip per violation, alongside the manifest fields (priority, plate, etc.). This module is
that cut.

Two layers, kept apart so the math tests without ffmpeg:
  * :func:`clip_window` -- PURE: key_frame + fps (+ optional total_frames) -> the [start, end]
    window in both frames and seconds, clamped to the real video bounds.
  * :func:`extract_clip` -- runs ffmpeg to write the sub-clip. The ffmpeg invocation is built by
    :func:`ffmpeg_extract_args` (testable as a plain list) and executed through an injectable
    ``runner``, so unit tests assert the command without a video or the ffmpeg binary.

Lossless by default (``-c copy``): the returned clip keeps the source resolution and per-frame bytes
intact, which is exactly what the :mod:`violations.video_integrity` hop checks downstream. A
stream-copy cut snaps the START to the nearest prior keyframe (so the real start can sit a fraction
earlier than requested) -- harmless here because the 10 s pre-roll is slack around the incident. When
a frame-accurate cut matters, pass ``recompress=True`` (a deliberate re-encode hop: same resolution,
different bytes -> validate that hop with ``allow_recompress=True``).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# How much context to keep around each incident, per the backend contract.
PRE_ROLL_SEC = 10.0
POST_ROLL_SEC = 5.0

# runner(args: list[str]) -> None : run ffmpeg with these args (raise on failure). Injectable.
FfmpegRunner = Callable[[list], Any]


@dataclass
class ClipWindow:
    """The trim window for one violation, in both frame and second units."""
    start_frame: int
    end_frame: int               # inclusive last frame of the window
    start_sec: float
    end_sec: float               # exclusive cut point (start of the frame AFTER end_frame)
    key_frame: int

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def n_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)

    def to_dict(self) -> dict:
        return {"start_frame": self.start_frame, "end_frame": self.end_frame,
                "start_sec": round(self.start_sec, 3), "end_sec": round(self.end_sec, 3),
                "key_frame": self.key_frame, "duration_sec": round(self.duration_sec, 3)}


def clip_window(key_frame: int, fps: float, *, total_frames: Optional[int] = None,
                pre_sec: float = PRE_ROLL_SEC, post_sec: float = POST_ROLL_SEC) -> ClipWindow:
    """The [key_frame - pre_sec, key_frame + post_sec] window, clamped to [0, total_frames-1].

    Returns frame indices AND the second offsets ffmpeg cuts on. ``end_sec`` is the start of the
    frame *after* ``end_frame`` so the cut INCLUDES the last frame. Clamps at the video start (a
    violation in the first 10 s still yields a clip, just with a shorter pre-roll) and, when
    ``total_frames`` is known, at the video end.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    start_frame = max(0, round(key_frame - pre_sec * fps))
    end_frame = round(key_frame + post_sec * fps)
    if total_frames is not None:
        last = max(0, total_frames - 1)
        end_frame = min(end_frame, last)
        start_frame = min(start_frame, last)
    start_frame = min(start_frame, end_frame)
    return ClipWindow(
        start_frame=start_frame, end_frame=end_frame,
        start_sec=start_frame / fps, end_sec=(end_frame + 1) / fps,
        key_frame=key_frame)


@dataclass
class ClipAsset:
    """A produced evidence clip, ready to fold into the export bundle. Carry ``data`` (the encoded
    bytes) OR ``path`` (read lazily at bundle time), plus the ``window`` it was cut on."""
    data: Optional[bytes] = None
    path: Optional[str] = None
    window: Optional[ClipWindow] = None
    container: str = "mp4"
    recompressed: bool = False

    def read_bytes(self) -> bytes:
        if self.data is not None:
            return self.data
        if self.path is not None:
            with open(self.path, "rb") as f:
                return f.read()
        raise ValueError("ClipAsset has neither data nor path")


def ffmpeg_extract_args(src: str, dest: str, window: ClipWindow, *,
                        recompress: bool = False, crf: int = 18) -> list:
    """Build the ffmpeg argument list for the cut (no execution -> trivially testable).

    ``-ss``/``-to`` are placed BEFORE ``-i`` for a fast seek. ``-c copy`` is a lossless stream copy
    (keeps resolution + bytes); ``recompress=True`` re-encodes (libx264) for a frame-accurate cut at
    the same resolution.
    """
    args = ["-y", "-ss", f"{window.start_sec:.3f}", "-to", f"{window.end_sec:.3f}", "-i", src]
    if recompress:
        args += ["-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-an"]
    else:
        args += ["-c", "copy"]
    args += [dest]
    return args


def _default_ffmpeg_runner(args: list) -> None:
    subprocess.run(["ffmpeg", *args], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_clip(src: str, dest: str, window: ClipWindow, *,
                 runner: FfmpegRunner = _default_ffmpeg_runner,
                 recompress: bool = False) -> ClipAsset:
    """Cut ``src`` to ``dest`` over ``window`` via ffmpeg; return a ClipAsset pointing at ``dest``."""
    runner(ffmpeg_extract_args(src, dest, window, recompress=recompress))
    return ClipAsset(path=dest, window=window,
                     container=os.path.splitext(dest)[1].lstrip(".").lower() or "mp4",
                     recompressed=recompress)


def extract_violation_clips(src: str, events, fps: float, out_dir: str, *,
                            total_frames: Optional[int] = None,
                            runner: FfmpegRunner = _default_ffmpeg_runner,
                            recompress: bool = False,
                            pre_sec: float = PRE_ROLL_SEC,
                            post_sec: float = POST_ROLL_SEC,
                            namer: Optional[Callable] = None) -> dict:
    """Cut one evidence sub-clip per ViolationEvent -> ``{vehicle/event key -> ClipAsset}``.

    ``namer(event) -> stem`` controls the filename (defaults to the export ``violation_id`` shape).
    This is the bridge the live pipeline calls after the violation list is final: hand it the source
    video + the events, get back the clips to fold into the export bundle.
    """
    os.makedirs(out_dir, exist_ok=True)
    if namer is None:
        from violations.export import violation_id as namer            # one naming source of truth
    out: dict = {}
    for ev in events:
        win = clip_window(ev.key_frame, fps, total_frames=total_frames,
                          pre_sec=pre_sec, post_sec=post_sec)
        stem = namer(ev)
        dest = os.path.join(out_dir, f"{stem}.{'mp4'}")
        out[stem] = extract_clip(src, dest, win, runner=runner, recompress=recompress)
    return out
