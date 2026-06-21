"""
Phase-1 pure-logic test suite for the Road Guard offline pipeline.

Runs locally with NO GPU and NO PyTorch: YOLO/ByteTrack never load. OCR, the blur metric
and the raw video frames are all mocked. We test the data-engineering logic only:

  Test 1  LPR voting  -> size gate, blur rejection, majority vote, tie-break, score formula
  Test 2  Handshake   -> raw IoU fails across an ego-motion gap, ego-compensated IoU inherits
  (plus) hard-negative trigger classification + on-disk layout
"""
import os
import sys

import numpy as np
import pytest

# import the sibling modules under tools/pipeline (mirrors the repo's sys.path convention)
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPELINE_DIR)

import lpr_consumer            # noqa: E402
import joiner                  # noqa: E402
import hard_negatives          # noqa: E402


# --------------------------------------------------------------------------- #
# Test helpers: a "frame" is a constant array whose pixel value == its frame id,
# so the mocked blur/OCR callables can map a crop back to per-frame fixtures.
# --------------------------------------------------------------------------- #
W, H = 1000, 500


def make_frame_provider():
    def provider(frame_id: int):
        return np.full((H, W, 3), frame_id, dtype=np.uint8)
    return provider


def _crop_tag(crop) -> int:
    return int(np.asarray(crop).reshape(-1)[0])


def make_blur_fn(blur_by_frame):
    return lambda crop: blur_by_frame[_crop_tag(crop)]


def make_ocr_reader(ocr_by_frame):
    return lambda crop: ocr_by_frame[_crop_tag(crop)]


def cache_from_frames(frames):
    return {"w": W, "h": H, "frames": frames}


# =========================================================================== #
# Test 1 -- LPR voting logic
# =========================================================================== #
def test_lpr_voting_tiebreak_blur_and_score():
    # one track (id=1); each frame designed to exercise one rule.
    frames = [
        {"frame": 0, "shift": [0, 0], "vehicles": [{"track_id": 1, "bbox": [0, 0, 200, 100]}]},  # area 20000 OK
        {"frame": 1, "shift": [0, 0], "vehicles": [{"track_id": 1, "bbox": [0, 0, 200, 100]}]},  # area 20000 OK
        {"frame": 2, "shift": [0, 0], "vehicles": [{"track_id": 1, "bbox": [0, 0, 400, 100]}]},  # area 40000 OK (largest)
        {"frame": 3, "shift": [0, 0], "vehicles": [{"track_id": 1, "bbox": [0, 0, 180, 100]}]},  # area 18000 OK
        {"frame": 4, "shift": [0, 0], "vehicles": [{"track_id": 1, "bbox": [0, 0, 300, 100]}]},  # area 30000 but BLURRY
        {"frame": 5, "shift": [0, 0], "vehicles": [{"track_id": 1, "bbox": [0, 0, 100, 100]}]},  # area 10000 -> too small
    ]
    cache = cache_from_frames(frames)

    blur_by_frame = {0: 500.0, 1: 500.0, 2: 500.0, 3: 500.0, 4: 10.0, 5: 500.0}   # 4 below default 100
    ocr_by_frame = {
        0: ("11-111-11", 0.8),
        1: ("11-111-11", 0.6),
        2: ("22-222-22", 0.9),
        3: ("22-222-22", 0.7),
        4: ("33-333-33", 0.99),   # rejected by blur, must never appear
        5: ("44-444-44", 0.95),   # rejected by size, must never appear
    }

    result = lpr_consumer.run_lpr(
        cache, make_frame_provider(),
        make_ocr_reader(ocr_by_frame), make_blur_fn(blur_by_frame),
    )[1]

    # size + blur gates leave exactly 4 valid reads (frames 0,1,2,3)
    assert result["n_reads"] == 4
    # votes tie 2-2; tie-break by largest bbox area -> "22-222-22" (its read has area 40000)
    assert result["plate_candidate"] == "22-222-22"
    # score = vote_fraction * mean_conf = (2/4) * ((0.9 + 0.7)/2) = 0.5 * 0.8 = 0.4
    assert result["plate_confidence_score"] == pytest.approx(0.4)


def test_lpr_voting_center_distance_secondary_tiebreak():
    # two plates, one vote each, identical area -> secondary key (smaller centre offset) wins.
    far = lpr_consumer.PlateRead(plate="AA", ocr_conf=0.9, area=20000, center_dist=300, frame=0)
    near = lpr_consumer.PlateRead(plate="BB", ocr_conf=0.9, area=20000, center_dist=50, frame=1)
    plate, _ = lpr_consumer.vote_plate([far, near])
    assert plate == "BB"


def test_lpr_no_reads_returns_none():
    plate, score = lpr_consumer.vote_plate([])
    assert plate is None and score == 0.0


def test_run_lpr_for_tracks_targets_only_requested_tracks():
    # track 1 appears in 2 frames (frame 1 bigger), track 2 in one frame. Ask only for track 1.
    frames = [
        {"frame": 0, "shift": [0, 0], "vehicles": [{"track_id": 1, "bbox": [0, 0, 200, 100]},
                                                    {"track_id": 2, "bbox": [0, 0, 200, 100]}]},
        {"frame": 1, "shift": [0, 0], "vehicles": [{"track_id": 1, "bbox": [0, 0, 400, 100]}]},
    ]
    cache = cache_from_frames(frames)
    blur = {0: 500.0, 1: 500.0}
    ocr = {0: ("AA-AAA-AA", 0.9), 1: ("BB-BBB-BB", 0.9)}
    out = lpr_consumer.run_lpr_for_tracks(cache, [1], make_frame_provider(),
                                          make_ocr_reader(ocr), make_blur_fn(blur))
    assert set(out.keys()) == {1}                       # track 2 never OCR'd
    assert out[1]["plate_candidate"] == "BB-BBB-BB"     # larger frame wins the area tie-break


# =========================================================================== #
# Test 2 -- Ego-compensated handshake
# =========================================================================== #
def _handshake_cache(new_track_start: int):
    """track 7 dies at frame 100; track 8 is (re)born at `new_track_start`.
    A constant ego shift of (15,0)/frame runs through the gap AND track 8's first frame,
    so the cumulative drift from frame 100 to frame 104 is exactly 60 px."""
    frames = []
    for f in (98, 99, 100):                                   # track 7's life
        frames.append({"frame": f, "shift": [0, 0],
                       "vehicles": [{"track_id": 7, "bbox": [100, 100, 200, 200]}]})
    for f in range(101, new_track_start):                     # the gap: pure ego-motion
        frames.append({"frame": f, "shift": [15, 0], "vehicles": []})
    # track 8 reappears displaced by the cumulative shift; its FIRST frame still carries the
    # camera motion from the previous frame (so sum_shift over 101..new_start lands on 60).
    for i, f in enumerate(range(new_track_start, new_track_start + 3)):
        frames.append({"frame": f, "shift": [15, 0] if i == 0 else [0, 0],
                       "vehicles": [{"track_id": 8, "bbox": [160, 100, 260, 200]}]})
    return cache_from_frames(frames)


def test_handshake_inherits_plate_across_id_switch():
    cache = _handshake_cache(new_track_start=104)
    index = lpr_consumer.build_track_index(cache)

    # raw IoU fails: the two boxes are 60px apart -> 0.25
    assert joiner.raw_iou(index, 7, 8) < 0.8
    # ego-compensated IoU succeeds: translate dead box by summed shift (60,0) -> exact overlap
    assert joiner.ego_compensated_iou(cache, index, 7, 8) > 0.8

    plate_map = {
        7: {"plate_candidate": "55-555-55", "plate_confidence_score": 0.7},
        8: {"plate_candidate": None, "plate_confidence_score": 0.0},   # ran LPR, got nothing
    }
    events = [{"violation_id": 1, "track_id": 8,
               "start_frame": 104, "end_frame": 110, "confidence": 0.9}]

    joined = joiner.join_events(events, plate_map, cache)
    assert joined[0]["plate_candidate"] == "55-555-55"
    assert joined[0]["plate_source"] == "inherited:7"
    assert joined[0]["inherited_from"] == 7


def test_handshake_skips_when_gap_out_of_range():
    # track 8 reborn 10 frames later -> outside the 3..5 window -> no inheritance
    cache = _handshake_cache(new_track_start=110)
    plate_map = {
        7: {"plate_candidate": "55-555-55", "plate_confidence_score": 0.7},
        8: {"plate_candidate": None, "plate_confidence_score": 0.0},
    }
    events = [{"violation_id": 1, "track_id": 8,
               "start_frame": 110, "end_frame": 116, "confidence": 0.9}]
    joined = joiner.join_events(events, plate_map, cache)
    assert joined[0]["plate_candidate"] is None
    assert joined[0]["plate_source"] == "none"


def test_join_uses_own_plate_when_present():
    cache = _handshake_cache(new_track_start=104)
    plate_map = {8: {"plate_candidate": "77-777-77", "plate_confidence_score": 0.9}}
    events = [{"violation_id": 1, "track_id": 8,
               "start_frame": 104, "end_frame": 110, "confidence": 0.9}]
    joined = joiner.join_events(events, plate_map, cache)
    assert joined[0]["plate_candidate"] == "77-777-77"
    assert joined[0]["plate_source"] == "own"


# =========================================================================== #
# Hard-negative logging
# =========================================================================== #
def test_hard_negative_classification():
    assert hard_negatives.classify_event(
        {"confidence": 0.9, "plate_candidate": None, "plate_confidence_score": 0.0}
    ) == hard_negatives.REASON_HIGH_CONF_NO_PLATE
    assert hard_negatives.classify_event(
        {"confidence": 0.9, "plate_candidate": "12-345-67", "plate_confidence_score": 0.1}
    ) == hard_negatives.REASON_HIGH_CONF_NO_PLATE      # weak plate counts as "no plate"
    assert hard_negatives.classify_event(
        {"confidence": 0.5, "plate_candidate": "12-345-67", "plate_confidence_score": 0.9}
    ) == hard_negatives.REASON_BORDERLINE
    assert hard_negatives.classify_event(
        {"confidence": 0.9, "plate_candidate": "12-345-67", "plate_confidence_score": 0.9}
    ) is None                                          # confident AND billable -> not logged
    assert hard_negatives.classify_event(
        {"confidence": 0.2, "plate_candidate": None, "plate_confidence_score": 0.0}
    ) is None                                          # clearly-not-a-violation -> not logged


def test_hard_negative_logging_layout(tmp_path):
    events = [
        {"violation_id": 1, "track_id": 8, "confidence": 0.95,
         "plate_candidate": None, "plate_confidence_score": 0.0},      # high_conf_no_plate
        {"violation_id": 2, "track_id": 9, "confidence": 0.5,
         "plate_candidate": "12-345-67", "plate_confidence_score": 0.8},  # borderline
        {"violation_id": 3, "track_id": 10, "confidence": 0.95,
         "plate_candidate": "33-333-33", "plate_confidence_score": 0.9},  # not logged
    ]
    logged = hard_negatives.log_hard_negatives(events, str(tmp_path), clip_prefix="demo")
    assert len(logged) == 2
    assert os.path.isfile(tmp_path / hard_negatives.REASON_HIGH_CONF_NO_PLATE / "demo__viol1.json")
    assert os.path.isfile(tmp_path / hard_negatives.REASON_BORDERLINE / "demo__viol2.json")
    assert os.path.isfile(tmp_path / "manifest.jsonl")
    with open(tmp_path / "manifest.jsonl") as fh:
        assert sum(1 for _ in fh) == 2
