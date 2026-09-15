"""
Two worker outputs, end to end without a GPU or a network:

  * the plate picture -- cut by the evidence stage, packed into the export bundle, found again
    by worker_common after unpacking, uploaded to R2 by worker.py and carried by the violation
    record (plateImagePath);
  * worker.py --keep-annotated -- the drive's annotated video survives ship() deleting the job dir.

The OCR reader, the plate cropper, the pipeline and the S3 client are fakes.
"""
import contextlib
import os
import sys
import types

import numpy as np
import pytest

import worker_common as wc
from lpr.evidence import EvidenceCollector
from violations import export as vx
from violations.event import ViolationEvent, ViolationType


def _event(vid=7, key_frame=120):
    return ViolationEvent(vehicle_id=vid, violation_type=ViolationType.SPEEDING, key_frame=key_frame,
                          details={"est_speed_kmh": 96.0, "speed_limit_kmh": 80, "over_by_kmh": 16.0})


def _collector(cropper):
    """Every crop reads as plate 12-345-67; "sharpness" is the mean pixel value, so tests pick
    which picture wins by the value they fill it with."""
    return EvidenceCollector(lambda crop: ("12-345-67", 0.9), min_area=1,
                             sharpness_fn=lambda crop: float(crop.mean()), plate_cropper=cropper)


def _frame(value):
    return np.full((60, 80, 3), value, np.uint8)


@pytest.fixture(scope="module")
def worker():
    """worker.py, imported without leaking the credentials its .env loader copies into os.environ."""
    saved = dict(os.environ)
    try:
        import worker as module
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return module


# --------------------------------------------------------------------------- #
# the plate picture
# --------------------------------------------------------------------------- #
def test_collector_cuts_a_plate_from_each_crop_and_keeps_the_clearest():
    def cropper(crop):
        value = int(crop[0, 0, 0])
        if value == 200:
            return np.full((4, 10, 3), 200, np.uint8)     # sharpest car picture, but a tiny plate
        if value == 100:
            return np.full((20, 50, 3), 100, np.uint8)    # 25x the pixels: the clearer plate
        return None                                       # no plate in this picture

    collector = _collector(cropper)
    for frame_id, value in enumerate((200, 100, 50)):
        collector.observe_vehicle(frame_id, _frame(value), 7, (0, 0, 80, 60))
    result = collector.collect_evidence(_event())

    assert [int(c[0, 0, 0]) for c in result.evidence_crops] == [200, 100, 50]
    assert [None if p is None else p.shape[:2] for p in result.evidence_plate_crops] == \
        [(4, 10), (20, 50), None]
    assert result.plate_crop is result.evidence_plate_crops[1]


def test_no_plate_picture_never_costs_the_violation():
    collector = EvidenceCollector(lambda crop: (None, 0.0), min_area=1)     # no cropper at all
    collector.observe_vehicle(0, _frame(9), 7, (0, 0, 80, 60))
    result = collector.collect_evidence(_event())
    assert result.evidence_plate_crops == [] and result.plate_crop is None

    def broken(crop):
        raise RuntimeError("plate model crashed")

    collector = _collector(broken)
    collector.observe_vehicle(0, _frame(9), 7, (0, 0, 80, 60))
    result = collector.collect_evidence(_event())
    assert result.evidence_plate_crops == [None] and result.plate_crop is None
    assert result.plate == "12-345-67" and len(result.evidence_crops) == 1


def test_plate_picture_travels_from_the_bundle_to_the_violation_record(tmp_path):
    collector = _collector(lambda crop: crop[10:30, 20:60].copy())
    collector.observe_vehicle(0, _frame(120), 7, (0, 0, 80, 60))
    with_plate = _event(vid=7)
    without_plate = _event(vid=8, key_frame=300)
    records = [(with_plate, collector.collect_evidence(with_plate)), (without_plate, None)]

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    bundle_path = out_dir / "drive_violations_bundle.tar.gz"
    bundle_path.write_bytes(vx.bundle_targz(vx.build_export(records, created_utc=0.0)))
    violations_dir = out_dir / "violations"
    manifest = wc.unpack_bundle(str(bundle_path), str(violations_dir))

    stem, other = vx.violation_id(with_plate), vx.violation_id(without_plate)
    plates = wc.collect_plate_images(manifest, str(out_dir), str(violations_dir))
    assert plates == {stem: f"violations/{stem}/plate.png"}
    import cv2
    assert cv2.imread(str(out_dir / plates[stem])).shape[:2] == (20, 40)

    key = f"session_1/out/{stem}_plate.png"
    payloads = wc.build_violation_payloads(manifest, {}, drive_id="d1", session_id="session_1",
                                           session_dir=str(tmp_path), detected_at="now",
                                           plate_images={stem: key})
    by_id = {p["violationId"]: p for p in payloads}
    assert by_id[stem]["plateImagePath"] == key and by_id[stem]["evidence"]["plateImage"] == key
    assert by_id[other]["plateImagePath"] is None


class _FakeS3:
    def __init__(self, fail_for=()):
        self.uploads = []
        self.fail_for = fail_for

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        if any(part in key for part in self.fail_for):
            raise ConnectionError("R2 unreachable")
        self.uploads.append((os.path.basename(path), key, ExtraArgs))


def test_worker_uploads_plate_pictures_and_skips_a_failed_one(worker, tmp_path):
    plates = {}
    for stem in ("v7_SPEEDING_f120", "v8_SOLID_LINE_CROSSING_f300"):
        (tmp_path / "violations" / stem).mkdir(parents=True)
        (tmp_path / "violations" / stem / "plate.png").write_bytes(b"png")
        plates[stem] = f"violations/{stem}/plate.png"

    s3 = _FakeS3(fail_for=("v8_",))
    keys = worker.upload_plate_images(s3, plates, "session_1", str(tmp_path))

    assert keys == {"v7_SPEEDING_f120": "session_1/out/v7_SPEEDING_f120_plate.png"}
    assert s3.uploads == [("plate.png", "session_1/out/v7_SPEEDING_f120_plate.png",
                           {"ContentType": "image/png"})]


def test_violation_post_carries_the_plate_picture_key(worker, monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(json)
        return types.SimpleNamespace(ok=True)

    monkeypatch.setattr(worker.requests, "post", fake_post)
    worker.report_violation({"driveId": "d1", "videoClipPath": "s/out/v.mp4", "carId": "12-345-67",
                             "calculatedSpeed": 96.0, "lat": 32.0, "lon": 34.8,
                             "violationType": "speeding", "plateImagePath": "s/out/v_plate.png",
                             "plate": "12-345-67"})
    assert sent["plateImagePath"] == "s/out/v_plate.png"
    assert "plate" not in sent                    # still only the fields the API accepts


# --------------------------------------------------------------------------- #
# --keep-annotated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("keep", [True, False])
def test_annotated_video_survives_ship_only_with_keep_annotated(worker, tmp_path, monkeypatch, keep):
    work = tmp_path / "work"
    monkeypatch.setattr(worker, "WORK_DIR", str(work))
    monkeypatch.setattr(worker, "ANNOTATED_DIR", str(work / "annotated"))
    monkeypatch.setattr(worker, "KEEP_ANNOTATED", keep)

    def fake_download(s3, files, job_dir):
        path = os.path.join(job_dir, "roadguard_1.mp4")
        open(path, "wb").close()
        return path

    def fake_pipeline(video, yolo, lane, tire, *, out_dir, **kwargs):
        annotated = os.path.join(out_dir, "roadguard_1_annotated.mp4")
        with open(annotated, "wb") as fh:
            fh.write(b"annotated")
        return {"annotated_video": annotated, "vehicles_csv": None, "violations": []}

    completed = []
    monkeypatch.setattr(worker, "download_session", fake_download)
    monkeypatch.setattr(worker, "complete", lambda drive_id, status, error=None: completed.append(status))
    monkeypatch.setattr(worker.wc, "attach_models", lambda main, models: None)
    monkeypatch.setattr(worker.wc, "reset_speed_limit_lookup", lambda: None)
    monkeypatch.setitem(sys.modules, "main", types.SimpleNamespace(process_video_with_models=fake_pipeline))

    job = {"driveId": "d1", "sessionId": "session_1", "files": {"video": "session_1/roadguard_1.mp4"}}
    timer = types.SimpleNamespace(measure=lambda key: contextlib.nullcontext())
    models = types.SimpleNamespace(yolo=None, lane=None, tire=None)
    worker.ship(None, worker.process(None, job, models, timer))   # 0 violations: complete, then rmtree

    assert completed == ["processed"]
    assert not (work / "session_1").exists()
    kept = work / "annotated" / "session_1_annotated.mp4"
    assert kept.exists() is keep
    if keep:
        assert kept.read_bytes() == b"annotated"


def test_keep_annotated_copes_with_a_drive_that_rendered_no_video(worker, tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ANNOTATED_DIR", str(tmp_path / "annotated"))
    worker.keep_annotated_video(None, "session_2")                    # e.g. a --benchmark run
    worker.keep_annotated_video(str(tmp_path / "missing.mp4"), "session_2")
    assert not (tmp_path / "annotated").exists()
