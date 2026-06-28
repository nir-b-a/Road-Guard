"""
video_integrity -- prove a video has not silently lost fidelity between processing hops.

The Road Guard pipeline passes a clip through several stages (ingest -> detection -> evidence
crop -> backend upload). A clip that gets re-encoded, downscaled, or frame-dropped anywhere along
the way quietly destroys evidence quality: a plate that was readable at 1080p is gone at 720p, and
a re-compressed frame can fabricate or smear the very pixels a violation rests on. This module is
the GUARD RAIL: it fingerprints the raw video once at ingest, then asserts at every later hop that
the metadata still matches and the resolution has not been lost.

  * :func:`probe_video`  -- run ``ffprobe`` once on a raw clip -> ``VideoMeta`` with
    ``{w, h, fps, frame_count, sha256, duration_sec, codec}``.
  * :func:`validate_hop` / :func:`assert_hop` -- compare the current clip's meta against the
    reference captured at ingest; report (or raise on) any resolution loss, frame loss, or
    re-compression.

``ffprobe`` is invoked through an injectable ``runner`` so the parsing + validation logic
unit-tests with simulated ffprobe JSON -- no ffmpeg binary and no real video required.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Callable, Optional

# runner(args: list[str]) -> str : run ffprobe with these args, return its stdout (JSON) text.
FfprobeRunner = Callable[[list], str]

# How close two frame rates must be to count as "unchanged" (absorbs 29.97 vs 30000/1001 rounding).
FPS_TOLERANCE = 0.01


class IntegrityError(Exception):
    """Raised by :func:`assert_hop` when a clip failed an integrity check at a pipeline hop."""


@dataclass
class VideoMeta:
    """Integrity fingerprint of one video file."""
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    sha256: Optional[str] = None      # None when hashing was skipped (e.g. metadata-only probe)
    duration_sec: Optional[float] = None
    codec: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VideoMeta":
        return cls(**{k: d.get(k) for k in (
            "filename", "width", "height", "fps", "frame_count",
            "sha256", "duration_sec", "codec")})

    @property
    def resolution(self) -> tuple:
        return (self.width, self.height)


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file's bytes (1 MiB chunks -> flat memory on multi-GB clips)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _default_ffprobe_runner(args: list) -> str:
    """Invoke the real ffprobe binary and return its stdout. Separated so tests inject a fake."""
    out = subprocess.run(["ffprobe", *args], check=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out.stdout.decode("utf-8", "replace")


def _parse_fps(stream: dict) -> float:
    """ffprobe gives frame rate as a fraction string ("30000/1001"). Parse to float, preferring
    ``avg_frame_rate`` then ``r_frame_rate``; 0/0 (unknown) -> 0.0."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = stream.get(key)
        if not val or val == "0/0":
            continue
        try:
            return float(Fraction(val))
        except (ValueError, ZeroDivisionError):
            try:
                return float(val)
            except ValueError:
                continue
    return 0.0


def _first_video_stream(probe: dict) -> dict:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    raise ValueError("ffprobe found no video stream")


def probe_video(path: str, *, compute_sha256: bool = True,
                runner: FfprobeRunner = _default_ffprobe_runner) -> VideoMeta:
    """Fingerprint ``path`` with ffprobe -> VideoMeta.

    Reads the first video stream's geometry + frame rate from ffprobe JSON and (unless
    ``compute_sha256=False``) the byte-exact SHA-256 of the file. ``frame_count`` falls back to
    ``round(duration * fps)`` when the container omits ``nb_frames`` (common for streamed MP4).
    """
    args = ["-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", path]
    probe = json.loads(runner(args))
    stream = _first_video_stream(probe)
    fmt = probe.get("format", {})

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_fps(stream)

    duration = stream.get("duration") or fmt.get("duration")
    duration_sec = float(duration) if duration not in (None, "N/A") else None

    nb = stream.get("nb_frames")
    if nb not in (None, "N/A", "0"):
        frame_count = int(nb)
    elif duration_sec is not None and fps > 0:
        frame_count = int(round(duration_sec * fps))     # container omitted the count -> derive it
    else:
        frame_count = 0

    sha = file_sha256(path) if compute_sha256 else None
    return VideoMeta(
        filename=os.path.basename(path),
        width=width, height=height, fps=fps,
        frame_count=frame_count, sha256=sha,
        duration_sec=duration_sec, codec=stream.get("codec_name"))


def validate_hop(reference: VideoMeta, current: VideoMeta, *,
                 allow_recompress: bool = False,
                 fps_tolerance: float = FPS_TOLERANCE,
                 frame_tolerance: int = 0) -> list:
    """Compare a clip at the current hop against the reference fingerprinted at ingest.

    Returns a list of human-readable problem strings (EMPTY list == passed). Checks:
      * RESOLUTION never shrinks/changes  (the headline "did we lose pixels" check),
      * FRAME RATE unchanged within ``fps_tolerance``,
      * FRAME COUNT unchanged within ``frame_tolerance`` (catches dropped frames / re-mux trims),
      * BYTES IDENTICAL via sha256 -- UNLESS ``allow_recompress=True``. A sha mismatch is exactly
        how a silent re-encode (lossy recompression) is caught on a hop that was meant to be a
        lossless passthrough. Set ``allow_recompress=True`` for hops where a re-encode is expected
        and intentional, and the geometry checks still guarantee no resolution/frame loss.
    """
    problems: list = []

    if (reference.width, reference.height) != (current.width, current.height):
        problems.append(
            f"resolution changed: {reference.width}x{reference.height} -> "
            f"{current.width}x{current.height} (evidence pixels lost)")

    if reference.fps and abs(reference.fps - current.fps) > fps_tolerance:
        problems.append(f"fps changed: {reference.fps:.4f} -> {current.fps:.4f}")

    if reference.frame_count and abs(reference.frame_count - current.frame_count) > frame_tolerance:
        problems.append(
            f"frame_count changed: {reference.frame_count} -> {current.frame_count} "
            f"(frames dropped or trimmed)")

    if not allow_recompress and reference.sha256 and current.sha256:
        if reference.sha256 != current.sha256:
            problems.append(
                "sha256 mismatch: file bytes changed (silent re-encode / recompression) "
                f"{reference.sha256[:12]}... != {current.sha256[:12]}...")

    return problems


def assert_hop(reference: VideoMeta, current: VideoMeta, *, hop: str = "", **kwargs) -> None:
    """Raise :class:`IntegrityError` (with every problem) if the clip failed validation at ``hop``;
    return silently on success. Use this to FAIL-CLOSED at each pipeline boundary."""
    problems = validate_hop(reference, current, **kwargs)
    if problems:
        where = f" at hop '{hop}'" if hop else ""
        raise IntegrityError(
            f"video integrity check failed{where} for {current.filename}:\n  - "
            + "\n  - ".join(problems))
