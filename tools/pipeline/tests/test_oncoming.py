"""Oncoming-direction filter tests (CPU-only). A solid-white-line crossing is a SAME-DIRECTION
violation; an opposite-direction car must be dropped here. We stub the heavy deps (mirrors
test_phase2_wrappers) and use synthetic caches with hand-built motion."""
import os
import sys
import types
from unittest import mock

sys.modules.setdefault("cv2", mock.MagicMock(name="cv2"))
sys.modules.setdefault("torch", mock.MagicMock(name="torch"))
if "ultralytics" not in sys.modules:
    _u = types.ModuleType("ultralytics")
    _u.YOLO = mock.MagicMock(name="YOLO")
    sys.modules["ultralytics"] = _u

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import violation_consumer as vcons  # noqa: E402


def _cache():
    """tid 1 = oncoming (bbox slides DOWN the frame); tid 2 = same-direction (stable, high up)."""
    H = W = 1000
    frames = []
    for i in range(20):
        y1 = 100 + i * 25                                  # oncoming track descends fast
        frames.append({"frame": i, "shift": [0, 0], "vehicles": [
            {"track_id": 1, "bbox": [400, y1, 480, y1 + 80]},
            {"track_id": 2, "bbox": [450, 200, 520, 280]},
        ]})
    return {"prefix": "syn", "path": "syn.mp4", "fps": 30.0, "w": W, "h": H,
            "total": 20, "frames": frames}


def test_classify_marks_descending_track_oncoming():
    onc = vcons.classify_track_direction(_cache())
    assert onc[1] > onc[2]
    assert onc[1] >= vcons.DEFAULT_ONCOMING_P              # descending = oncoming -> filtered
    assert onc[2] < vcons.DEFAULT_ONCOMING_P               # stable/high = same-direction -> kept


def _patched_find(cache, **kw):
    events = [(1, 5, 8), (2, 5, 8)]
    model = {"features": ["distance", "angle_off"]}
    with mock.patch.object(vcons.cvt, "ensure_shifts", side_effect=lambda c: c), \
         mock.patch.object(vcons.cvt, "clip_timeline", return_value="TL"), \
         mock.patch.object(vcons, "events_from_timeline", return_value=events), \
         mock.patch.object(vcons.vc, "event_features",
                           side_effect=lambda c, ev: {"distance": 10.0, "angle_off": 0.1}), \
         mock.patch.object(vcons.vc, "confidence_lr", return_value=0.7):
        return vcons.find_violations(cache, model, prefix="syn", **kw)


def test_find_violations_drops_oncoming_track():
    out = _patched_find(_cache())
    tids = {o["track_id"] for o in out}
    assert tids == {2}                                     # oncoming tid 1 dropped, same-dir tid 2 kept


def test_filter_oncoming_can_be_disabled():
    out = _patched_find(_cache(), filter_oncoming=False)
    assert {o["track_id"] for o in out} == {1, 2}          # both kept when the gate is off
