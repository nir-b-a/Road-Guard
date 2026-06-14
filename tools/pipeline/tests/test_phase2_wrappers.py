"""
Phase-2 wrapper WIRING tests -- CPU-only, no models, no video, no GPU.

heavy_pass / violation_consumer / run_pipeline import crossing_violation_test, which imports
cv2, and the heavy pass lazily imports ultralytics/torch. None of that is guaranteed in CI, so
we stub cv2/torch/ultralytics in sys.modules BEFORE importing the wrappers, then patch the
specific legacy functions per test. We assert the *contracts* -- argument routing (esp. the
BoT-SORT override), schema pass-through, and orchestration order -- never a real model run.
"""
import json
import os
import sys
import types
from unittest import mock

import pytest

# --- stub the CI-unavailable heavy deps BEFORE importing anything that pulls them in ---
# setdefault: if a real one is installed (GPU env) we keep it; tests patch call sites anyway.
sys.modules.setdefault("cv2", mock.MagicMock(name="cv2"))
sys.modules.setdefault("torch", mock.MagicMock(name="torch"))
if "ultralytics" not in sys.modules:
    _ultralytics = types.ModuleType("ultralytics")
    _ultralytics.YOLO = mock.MagicMock(name="YOLO")
    sys.modules["ultralytics"] = _ultralytics

# pipeline dir on path (mirrors the repo's sibling-import convention)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import heavy_pass            # noqa: E402
import violation_consumer    # noqa: E402
import run_pipeline          # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_cache():
    """A minimal cache in the EXACT legacy schema (frame/shift/vehicles[track_id/bbox])."""
    return {"prefix": "clipX", "path": "clipX.mp4", "fps": 30.0,
            "w": 1920, "h": 1080, "total": 1,
            "frames": [{"frame": 0, "shift": [0, 0],
                        "vehicles": [{"track_id": 7, "bbox": [0, 0, 10, 10]}]}]}


# =========================================================================== #
# Test 1 -- heavy_pass configuration routing
# =========================================================================== #
def test_heavy_pass_routes_botsort_tracker(fake_cache):
    # patch YOLO so no weights ever load (cross-env safe), and build_cache so no video is read.
    with mock.patch("ultralytics.YOLO", mock.MagicMock(name="YOLO")), \
         mock.patch.object(heavy_pass.cvt, "build_cache", return_value=fake_cache) as m_build:
        result = heavy_pass.run_heavy_pass("clipX", refresh=True)

    assert result is fake_cache
    m_build.assert_called_once()
    args, kwargs = m_build.call_args
    # positional contract preserved: (prefix, lane_model, veh_weights, lane_conf, veh_conf)
    assert args[0] == "clipX"
    assert args[2] == heavy_pass.cvt.VEH_WEIGHTS               # default vehicle weights flow through
    assert args[3] == 0.25 and args[4] == 0.30                 # default lane/veh confidences
    # THE critical override:
    assert kwargs["tracker"] == "botsort.yaml"


def test_heavy_pass_reuses_cache_without_refresh(fake_cache):
    # refresh=False + an existing cache must short-circuit: no model setup, no build_cache.
    with mock.patch.object(heavy_pass.cvt, "load_cache", return_value=fake_cache), \
         mock.patch.object(heavy_pass.cvt, "build_cache") as m_build, \
         mock.patch("ultralytics.YOLO") as m_yolo:
        result = heavy_pass.run_heavy_pass("clipX", refresh=False)
    assert result is fake_cache
    m_build.assert_not_called()
    m_yolo.assert_not_called()


# =========================================================================== #
# Test 2 -- violation_consumer legacy chain (schema + shape pass-through)
# =========================================================================== #
def test_violation_consumer_chain_preserves_schema_and_shape(fake_cache):
    events = [(7, 100, 110), (8, 200, 205)]
    feat7 = {"distance": 12.0, "angle_off": 0.1}              # real LR feature keys, untouched
    features_by_event = {(7, 100, 110): feat7, (8, 200, 205): None}  # 2nd has no feats -> skipped
    model = {"features": ["distance", "angle_off"]}

    with mock.patch.object(violation_consumer.cvt, "ensure_shifts", side_effect=lambda c: c) as m_shift, \
         mock.patch.object(violation_consumer.cvt, "clip_timeline", return_value="TIMELINE") as m_tl, \
         mock.patch.object(violation_consumer, "events_from_timeline", return_value=events) as m_ev, \
         mock.patch.object(violation_consumer.vc, "event_features",
                           side_effect=lambda c, ev: features_by_event[ev]) as m_feat, \
         mock.patch.object(violation_consumer.vc, "confidence_lr", return_value=0.83) as m_conf:
        out = violation_consumer.find_violations(fake_cache, model, k_sec=0.05, prefix="clipX")

    # the exact legacy cache flowed through ensure_shifts -> clip_timeline
    m_shift.assert_called_once_with(fake_cache)
    m_tl.assert_called_once_with(fake_cache)
    # K converted to frames per clip (0.05s * 30fps -> 2) and handed to the grouping rule
    m_ev.assert_called_once_with("TIMELINE", max(1, round(0.05 * 30.0)))
    # features extracted for BOTH events; confidence scored only for the one with features
    assert m_feat.call_count == 2
    m_conf.assert_called_once_with(feat7, model)              # feature dict passed UNMODIFIED
    # payload shape is exactly the joiner contract; the feature-less event is dropped
    assert out == [{"violation_id": 0, "track_id": 7,
                    "start_frame": 100, "end_frame": 110, "confidence": 0.83}]


def test_load_confidence_model_unwraps_inner_model(tmp_path):
    path = tmp_path / "confidence_model.json"
    path.write_text(json.dumps({"model": {"features": ["distance"]}, "auc": 0.78}))
    assert violation_consumer.load_confidence_model(str(path)) == {"features": ["distance"]}


def test_load_confidence_model_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        violation_consumer.load_confidence_model(str(tmp_path / "nope.json"))


# =========================================================================== #
# Test 3 -- orchestrator end-to-end CLI sequence
# =========================================================================== #
def test_orchestrator_runs_components_in_order(tmp_path):
    cache = {"prefix": "mock_video.mp4", "path": "mock_video.mp4",
             "fps": 30.0, "w": 100, "h": 100, "total": 0, "frames": []}
    plate_map = {7: {"plate_candidate": "12-345-67", "plate_confidence_score": 0.9}}
    model = {"features": ["distance", "angle_off"]}
    violations = [{"violation_id": 0, "track_id": 7,
                   "start_frame": 1, "end_frame": 2, "confidence": 0.9}]
    joined = [{**violations[0], "plate_candidate": "12-345-67",
               "plate_confidence_score": 0.9, "plate_source": "own"}]

    manager = mock.MagicMock()                               # records cross-mock call ORDER
    with mock.patch.object(run_pipeline.heavy_pass, "run_heavy_pass", return_value=cache) as m_heavy, \
         mock.patch.object(run_pipeline.lpr_consumer, "make_video_frame_provider", return_value="PROVIDER"), \
         mock.patch.object(run_pipeline, "build_ocr_reader", return_value="OCR"), \
         mock.patch.object(run_pipeline.lpr_consumer, "run_lpr", return_value=plate_map) as m_lpr, \
         mock.patch.object(run_pipeline.violation_consumer, "load_confidence_model", return_value=model), \
         mock.patch.object(run_pipeline.violation_consumer, "find_violations", return_value=violations) as m_find, \
         mock.patch.object(run_pipeline.joiner, "join_events", return_value=joined) as m_join, \
         mock.patch.object(run_pipeline.hard_negatives, "log_hard_negatives", return_value=[]) as m_hard:
        manager.attach_mock(m_heavy, "heavy")
        manager.attach_mock(m_lpr, "lpr")
        manager.attach_mock(m_find, "violations")
        manager.attach_mock(m_join, "join")
        manager.attach_mock(m_hard, "hard")

        argv = ["run_pipeline.py", "mock_video.mp4", "--refresh", "--out", str(tmp_path)]
        with mock.patch.object(sys, "argv", argv):
            run_pipeline.main()

    # argparse routed the prefix + --refresh down to the heavy pass
    m_heavy.assert_called_once()
    assert m_heavy.call_args.args[0] == "mock_video.mp4"
    assert m_heavy.call_args.kwargs.get("refresh") is True

    # deterministic lifecycle: cache -> (LPR + violations) -> join -> hard-negatives
    assert [c[0] for c in manager.mock_calls] == ["heavy", "lpr", "violations", "join", "hard"]

    # data flows down the stream unbroken
    m_join.assert_called_once_with(violations, plate_map, cache)
    assert m_hard.call_args.args[0] == joined
    assert (tmp_path / "mock_video.mp4_violations.json").is_file()
