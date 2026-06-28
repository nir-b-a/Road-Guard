"""
Unit tests for the dual-graph fusion pieces -- pure Python, no cv2/torch/models.

  plate_char_voter      : length-first then per-position digit voting for Israeli plates
  plate_vehicle_linker  : link plate tracks to vehicle tracks by spatial containment
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plate_char_voter as pcv          # noqa: E402
import plate_vehicle_linker as pvl       # noqa: E402


# --------------------------------------------------------------------------- #
# per-character temporal voting
# --------------------------------------------------------------------------- #
def test_char_vote_recovers_majority_digit_per_slot():
    # last digit disagrees (3,3,2); confidence-weighted vote -> 3. Format as XX-XXX-XX.
    reads = [("14-281-33", 0.9), ("14-281-33", 0.8), ("14-281-32", 0.5)]
    plate, score = pcv.vote_characters(reads)
    assert plate == "14-281-33"
    assert 0.0 < score <= 1.0


def test_char_vote_is_length_first_then_positional():
    # two 7-digit reads (high conf) vs one 8-digit (low conf) -> length 7 wins, no misalignment
    reads = [("1428132", 0.9), ("1428132", 0.8), ("12345678", 0.3)]
    plate, _ = pcv.vote_characters(reads)
    assert plate == "14-281-32"            # XX-XXX-XX, never blended with the 8-digit read


def test_char_vote_eight_digit_layout():
    reads = [("123-45-678", 0.9), ("123-45-678", 0.7)]
    plate, _ = pcv.vote_characters(reads)
    assert plate == "123-45-678"           # XXX-XX-XXX


def test_char_vote_rejects_invalid_lengths():
    assert pcv.vote_characters([("123", 0.9), ("abcd", 0.5)]) == (None, 0.0)
    assert pcv.vote_characters([]) == (None, 0.0)


# --------------------------------------------------------------------------- #
# plate <-> vehicle containment linking
# --------------------------------------------------------------------------- #
def test_containment_full_partial_none():
    veh = [100, 100, 300, 300]
    assert pvl.containment([150, 150, 200, 200], veh) == 1.0       # plate fully inside
    assert pvl.containment([400, 400, 450, 450], veh) == 0.0       # disjoint
    half = pvl.containment([250, 150, 350, 200], veh)              # half the plate inside
    assert 0.4 < half < 0.6


def test_registry_assigns_plate_to_most_containing_vehicle():
    # plate track 100 sits inside vehicle 7 in 2 frames, briefly overlaps vehicle 8 once (less)
    frames = [
        {"vehicles": [{"track_id": 7, "bbox": [100, 100, 300, 300]},
                      {"track_id": 8, "bbox": [500, 500, 700, 700]}],
         "plates": [{"track_id": 100, "bbox": [180, 200, 230, 230]}]},          # inside 7
        {"vehicles": [{"track_id": 7, "bbox": [110, 100, 310, 300]},
                      {"track_id": 8, "bbox": [400, 400, 600, 600]}],
         "plates": [{"track_id": 100, "bbox": [190, 200, 240, 230]}]},          # inside 7
        {"vehicles": [{"track_id": 8, "bbox": [180, 180, 260, 260]}],
         "plates": [{"track_id": 100, "bbox": [200, 200, 250, 240]}]},          # partial in 8
    ]
    registry = pvl.build_plate_vehicle_registry(frames, min_containment=0.6)
    assert registry[100] == 7
    assert pvl.vehicle_to_plate_tracks(registry) == {7: [100]}


def test_registry_drops_uncontained_plates():
    frames = [{"vehicles": [{"track_id": 7, "bbox": [0, 0, 50, 50]}],
               "plates": [{"track_id": 100, "bbox": [400, 400, 450, 450]}]}]    # plate far away
    assert pvl.build_plate_vehicle_registry(frames) == {}
