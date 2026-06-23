"""
Tests for violations.video_integrity -- ffprobe parsing + the per-hop validation that catches
simulated recompression and resolution loss. ffprobe is replaced by a fake runner returning canned
JSON, so no ffmpeg binary or real video is needed.
"""
import hashlib
import json

import pytest

from violations import video_integrity as vi
from violations.video_integrity import VideoMeta, IntegrityError


def _probe_json(width=1920, height=1080, r_fps="30/1", avg_fps=None,
                nb_frames="900", duration="30.0", codec="h264"):
    stream = {"codec_type": "video", "width": width, "height": height,
              "r_frame_rate": r_fps, "codec_name": codec}
    if avg_fps is not None:
        stream["avg_frame_rate"] = avg_fps
    if nb_frames is not None:
        stream["nb_frames"] = nb_frames
    if duration is not None:
        stream["duration"] = duration
    return json.dumps({"streams": [{"codec_type": "audio"}, stream],
                       "format": {"duration": duration}})


def _runner(payload):
    return lambda args: payload


# --------------------------------------------------------------------------- #
# probe parsing
# --------------------------------------------------------------------------- #
def test_probe_parses_geometry_fps_and_count():
    meta = vi.probe_video("clip.mp4", compute_sha256=False, runner=_runner(_probe_json()))
    assert meta.width == 1920 and meta.height == 1080
    assert meta.fps == pytest.approx(30.0)
    assert meta.frame_count == 900
    assert meta.codec == "h264"
    assert meta.duration_sec == pytest.approx(30.0)
    assert meta.sha256 is None       # hashing skipped
    assert meta.resolution == (1920, 1080)


def test_probe_parses_fractional_ntsc_fps():
    meta = vi.probe_video("clip.mp4", compute_sha256=False,
                          runner=_runner(_probe_json(r_fps="30000/1001")))
    assert meta.fps == pytest.approx(29.97, abs=1e-2)


def test_probe_derives_frame_count_when_container_omits_it():
    # nb_frames absent -> frame_count = round(duration * fps) = round(30 * 30)
    meta = vi.probe_video("clip.mp4", compute_sha256=False,
                          runner=_runner(_probe_json(nb_frames=None, duration="30.0")))
    assert meta.frame_count == 900


def test_probe_raises_without_a_video_stream():
    payload = json.dumps({"streams": [{"codec_type": "audio"}], "format": {}})
    with pytest.raises(ValueError):
        vi.probe_video("clip.mp4", compute_sha256=False, runner=_runner(payload))


def test_file_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    data = b"road-guard evidence bytes" * 1000
    p.write_bytes(data)
    assert vi.file_sha256(str(p)) == hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# per-hop validation
# --------------------------------------------------------------------------- #
def _meta(**kw):
    base = dict(filename="clip.mp4", width=1920, height=1080, fps=30.0,
                frame_count=900, sha256="a" * 64, duration_sec=30.0, codec="h264")
    base.update(kw)
    return VideoMeta(**base)


def test_identical_meta_passes():
    ref = _meta()
    assert vi.validate_hop(ref, _meta()) == []


def test_resolution_loss_is_caught():
    ref = _meta()
    down = _meta(width=1280, height=720, sha256="a" * 64)
    problems = vi.validate_hop(ref, down)
    assert any("resolution changed" in p for p in problems)


def test_recompression_caught_by_sha_in_strict_mode():
    ref = _meta()
    recompressed = _meta(sha256="b" * 64)   # same geometry, different bytes == re-encode
    strict = vi.validate_hop(ref, recompressed)
    assert any("sha256 mismatch" in p for p in strict)
    # ...but a hop that legitimately re-encodes can opt out of the byte check
    assert vi.validate_hop(ref, recompressed, allow_recompress=True) == []


def test_recompression_with_resolution_loss_still_caught_when_recompress_allowed():
    ref = _meta()
    bad = _meta(width=1280, height=720, sha256="b" * 64)
    problems = vi.validate_hop(ref, bad, allow_recompress=True)
    # bytes allowed to differ, but losing pixels is never allowed
    assert any("resolution changed" in p for p in problems)
    assert not any("sha256" in p for p in problems)


def test_frame_drop_and_fps_change_are_caught():
    ref = _meta()
    dropped = _meta(frame_count=880, sha256="a" * 64)
    assert any("frame_count changed" in p for p in vi.validate_hop(ref, dropped))
    slowed = _meta(fps=25.0, sha256="a" * 64)
    assert any("fps changed" in p for p in vi.validate_hop(ref, slowed))


def test_fps_tolerance_absorbs_ntsc_rounding():
    ref = _meta(fps=30.0)
    cur = _meta(fps=29.997, sha256="a" * 64)
    assert vi.validate_hop(ref, cur, fps_tolerance=0.01) == []


def test_assert_hop_raises_on_failure_and_is_silent_on_success():
    ref = _meta()
    vi.assert_hop(ref, _meta())   # no raise
    with pytest.raises(IntegrityError) as ei:
        vi.assert_hop(ref, _meta(width=640, height=480), hop="evidence-crop")
    msg = str(ei.value)
    assert "evidence-crop" in msg and "resolution changed" in msg


def test_videometa_dict_roundtrip():
    ref = _meta()
    assert VideoMeta.from_dict(ref.to_dict()) == ref
