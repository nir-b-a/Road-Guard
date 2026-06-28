"""Module F (reid_linker) core tests -- pure logic, no cv2/model. Validates the no-temporal-overlap
hard guard, the gap window, cosine linking, and recall-first plate propagation."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import reid_linker as rl  # noqa: E402


def test_cosine_and_overlap_primitives():
    assert rl.cosine_sim([1, 0], [1, 0]) == 1.0
    assert abs(rl.cosine_sim([1, 0], [0, 1])) < 1e-9
    assert rl.spans_overlap((0, 100), (50, 200))           # share frames -> overlap
    assert not rl.spans_overlap((0, 100), (104, 200))      # disjoint


def test_links_same_car_across_gap():
    spans = {1: (0, 100), 2: (104, 200)}                   # 2 born 4 frames after 1 died
    emb = {1: [1.0, 0.0, 0.0], 2: [0.97, 0.05, 0.0]}       # very similar appearance
    canon = rl.link_tracks(spans, emb, sim_threshold=0.55, max_gap_frames=600)
    assert canon[1] == canon[2]                            # linked into one identity
    assert canon[1] == 1                                   # canonical = oldest id


def test_hard_guard_never_links_time_overlapping_tracks():
    # identical appearance but ALIVE AT THE SAME TIME -> must stay separate (the ID1/ID6 case)
    spans = {1: (0, 445), 6: (256, 445)}
    emb = {1: [1.0, 0.0], 6: [1.0, 0.0]}
    canon = rl.link_tracks(spans, emb, sim_threshold=0.5, max_gap_frames=600)
    assert canon[1] != canon[6]


def test_gap_window_rejects_too_distant_reacquisition():
    spans = {1: (0, 100), 2: (900, 1000)}                  # 800-frame gap
    emb = {1: [1.0, 0.0], 2: [1.0, 0.0]}
    canon = rl.link_tracks(spans, emb, sim_threshold=0.5, max_gap_frames=600)
    assert canon[1] != canon[2]


def test_dissimilar_appearance_not_linked():
    spans = {1: (0, 100), 2: (104, 200)}
    emb = {1: [1.0, 0.0], 2: [0.0, 1.0]}                   # orthogonal -> different cars
    canon = rl.link_tracks(spans, emb, sim_threshold=0.55, max_gap_frames=600)
    assert canon[1] != canon[2]


def test_candidate_predecessors_window_and_overlap():
    spans = {1: (0, 445), 6: (256, 445), 7: (447, 789), 2: (0, 42)}
    # target = the unknown violator tid 7 (born 447); exclude the violating set
    preds = rl.candidate_predecessors(spans, [7], max_gap_frames=600, exclude={6, 7})
    assert 1 in preds                                      # tid1 died f445, gap 2 -> candidate
    assert 2 in preds                                      # tid2 died f42, gap 405 -> within window
    assert 6 not in preds                                  # excluded (also a violator)


def test_candidate_predecessors_rejects_overlap_and_far():
    spans = {1: (0, 445), 9: (300, 800)}                   # 9 overlaps 1 in time
    preds = rl.candidate_predecessors(spans, [9], max_gap_frames=50)
    assert 1 not in preds                                  # overlapping -> never a predecessor


def test_plate_propagates_to_plateless_member():
    canonical = {1: 1, 2: 1}                               # one identity
    plate_map = {1: {"plate_candidate": "47-396-81", "plate_confidence_score": 0.97},
                 2: {"plate_candidate": None, "plate_confidence_score": 0.0}}
    enriched = rl.propagate_plates(canonical, plate_map)
    assert enriched[2]["plate_candidate"] == "47-396-81"
    assert enriched[2]["plate_source"] == "reid_inherited:1"
    assert enriched[2]["reid_inherited_from"] == 1
    assert enriched[1]["plate_source"] == "own"            # original keeps its own read


def test_propagate_does_not_overwrite_existing_own_plate():
    canonical = {1: 1, 2: 1}
    plate_map = {1: {"plate_candidate": "47-396-81", "plate_confidence_score": 0.97},
                 2: {"plate_candidate": "11-111-11", "plate_confidence_score": 0.40}}
    enriched = rl.propagate_plates(canonical, plate_map)
    assert enriched[2]["plate_candidate"] == "11-111-11"   # recall-first ADDS, never overrides own
    assert enriched[2]["plate_source"] == "own"


def test_reid_enrich_end_to_end_uses_fps_for_gap():
    spans = {1: (0, 100), 2: (110, 200)}                   # ~0.33s gap at 30fps
    emb = {1: [1.0, 0.0], 2: [0.95, 0.0]}
    plate_map = {1: {"plate_candidate": "47-396-81", "plate_confidence_score": 0.9},
                 2: {"plate_candidate": None, "plate_confidence_score": 0.0}}
    enriched, canon = rl.reid_enrich(spans, emb, plate_map, fps=30.0,
                                     sim_threshold=0.55, max_gap_sec=20.0)
    assert canon[1] == canon[2]
    assert enriched[2]["plate_candidate"] == "47-396-81"
    assert enriched[2]["plate_source"] == "reid_inherited:1"
