"""
Tests for violations.clip_extract -- the 10s-pre / 5s-post evidence-clip window math and the ffmpeg
command construction. ffmpeg is replaced by a recording fake, so no binary or video is touched.
"""
import pytest

from violations import clip_extract as ce
from violations.clip_extract import ClipWindow, ClipAsset
from violations.event import ViolationEvent, ViolationType


# --------------------------------------------------------------------------- #
# window math (pure)
# --------------------------------------------------------------------------- #
def test_window_is_10s_before_and_5s_after_the_key_frame():
    w = ce.clip_window(key_frame=600, fps=30.0)            # 30 fps -> 300 frames pre, 150 post
    assert w.start_frame == 600 - 300
    assert w.end_frame == 600 + 150
    assert w.start_sec == pytest.approx(300 / 30.0)
    # ~15s, plus one frame because the end frame is inclusive
    assert w.duration_sec == pytest.approx(ce.PRE_ROLL_SEC + ce.POST_ROLL_SEC, abs=0.1)
    assert w.key_frame == 600


def test_window_clamps_at_video_start():
    # incident in the first few seconds -> pre-roll is shortened, never negative
    w = ce.clip_window(key_frame=30, fps=30.0)
    assert w.start_frame == 0
    assert w.start_sec == 0.0
    assert w.end_frame == 30 + 150


def test_window_clamps_at_video_end():
    w = ce.clip_window(key_frame=990, fps=30.0, total_frames=1000)
    assert w.end_frame == 999                              # last frame, not 990+150
    assert w.start_frame == 990 - 300


def test_window_rejects_nonpositive_fps():
    with pytest.raises(ValueError):
        ce.clip_window(100, 0.0)


def test_custom_pre_post_roll():
    w = ce.clip_window(1000, 25.0, pre_sec=4.0, post_sec=2.0)
    assert w.start_frame == 1000 - 100
    assert w.end_frame == 1000 + 50


# --------------------------------------------------------------------------- #
# ffmpeg command construction
# --------------------------------------------------------------------------- #
def test_lossless_args_use_stream_copy_and_seek_window():
    w = ce.clip_window(600, 30.0)
    args = ce.ffmpeg_extract_args("in.mp4", "out.mp4", w)
    assert "-c" in args and args[args.index("-c") + 1] == "copy"   # lossless: keeps resolution+bytes
    # -ss / -to come BEFORE -i (fast seek)
    assert args.index("-ss") < args.index("-i")
    assert args.index("-to") < args.index("-i")
    assert args[-1] == "out.mp4"


def test_recompress_args_reencode_for_frame_accuracy():
    w = ce.clip_window(600, 30.0)
    args = ce.ffmpeg_extract_args("in.mp4", "out.mp4", w, recompress=True)
    assert "libx264" in args
    assert "copy" not in args


def test_extract_clip_invokes_runner_and_returns_asset():
    calls = []
    w = ce.clip_window(600, 30.0)
    asset = ce.extract_clip("in.mp4", "out.mp4", w, runner=lambda a: calls.append(a))
    assert len(calls) == 1 and calls[0][-1] == "out.mp4"
    assert isinstance(asset, ClipAsset) and asset.path == "out.mp4" and asset.window is w


def test_extract_violation_clips_one_per_event(tmp_path):
    events = [
        ViolationEvent(vehicle_id=1, violation_type=ViolationType.SPEEDING, key_frame=300),
        ViolationEvent(vehicle_id=2, violation_type=ViolationType.SOLID_LINE_CROSSING, key_frame=900),
    ]
    cmds = []
    clips = ce.extract_violation_clips("src.mp4", events, fps=30.0, out_dir=str(tmp_path),
                                       total_frames=2000, runner=lambda a: cmds.append(a))
    assert len(clips) == 2 and len(cmds) == 2
    # filenames follow the export violation_id naming
    assert any("v1_SPEEDING_f300" in k for k in clips)
    assert any("v2_SOLID_LINE_CROSSING_f900" in k for k in clips)


def test_clip_asset_reads_bytes_from_data_or_path(tmp_path):
    assert ClipAsset(data=b"abc").read_bytes() == b"abc"
    p = tmp_path / "c.mp4"
    p.write_bytes(b"video-bytes")
    assert ClipAsset(path=str(p)).read_bytes() == b"video-bytes"
    with pytest.raises(ValueError):
        ClipAsset().read_bytes()
