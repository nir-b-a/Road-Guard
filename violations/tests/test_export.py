"""
Tests for violations.export -- the LEXICOGRAPHIC type-tiered priority, the manifest shape, the
.tar.gz bundling, and the multipart HTTP POST. No cv2, no requests, no backend: the PNG encoder is
injected and the HTTP session is a fake, so this runs on a bare interpreter.
"""
import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from violations.event import ViolationEvent, ViolationType
from violations import export as vx


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fake_encoder(crop):
    """Deterministic stand-in for cv2.imencode: encode the crop's repr to bytes."""
    return b"PNG:" + repr(crop).encode("utf-8")


def _evidence(plate="12-345-67", plate_score=0.8, crops=("a", "b", "c"),
              frame_ids=(10, 11, 12), plate_crop=SimpleNamespace(size=1), manual_review=False):
    return SimpleNamespace(plate=plate, plate_score=plate_score, n_reads=len(crops),
                           evidence_crops=list(crops), evidence_frame_ids=list(frame_ids),
                           plate_crop=plate_crop, manual_review=manual_review)


def _speeding(vid, over_by, conf=1.0, key_frame=100):
    return ViolationEvent(vehicle_id=vid, violation_type=ViolationType.SPEEDING,
                          key_frame=key_frame, confidence=conf,
                          details={"est_speed_kmh": 50 + over_by, "speed_limit_kmh": 50,
                                   "over_by_kmh": over_by})


def _crossing(vid, conf, key_frame=200):
    return ViolationEvent(vehicle_id=vid, violation_type=ViolationType.SOLID_LINE_CROSSING,
                          key_frame=key_frame, confidence=conf, details={"crossing_frac": 0.4})


def _yellow(vid, key_frame, conf=0.5):
    return ViolationEvent(vehicle_id=vid, violation_type=ViolationType.YELLOW_LINE_RIGHT,
                          key_frame=key_frame, confidence=conf, details={"shoulder_overlap_frac": 0.5})


def _order(events):
    """The vehicle_ids in queue order for a list of events (via build_export)."""
    recs = [(e, None) for e in events]
    bundle = vx.build_export(recs, png_encoder=_fake_encoder, created_utc=0.0)
    return [v["vehicle_id"] for v in bundle.violations]


# --------------------------------------------------------------------------- #
# confidence axis (orders WITHIN a tier)
# --------------------------------------------------------------------------- #
def test_normalize_confidence_clamps_and_defaults():
    assert vx.normalize_confidence(_crossing(1, 0.5)) == 0.5
    assert vx.normalize_confidence(_crossing(1, 1.7)) == 1.0          # over-1 clamps down
    assert vx.normalize_confidence(_crossing(1, -0.2)) == 0.0
    ev = ViolationEvent(vehicle_id=1, violation_type="X", key_frame=0, confidence=None)
    assert vx.normalize_confidence(ev) == 1.0                         # missing -> hard-rule default


def test_speeding_magnitude_drives_confidence():
    at_limit = vx.normalize_confidence(_speeding(0, over_by=0))
    marginal = vx.normalize_confidence(_speeding(1, over_by=2))
    brutal = vx.normalize_confidence(_speeding(2, over_by=60))
    assert at_limit < marginal < brutal
    assert at_limit == pytest.approx(vx.SPEEDING_CONF_FLOOR, rel=1e-6)
    assert brutal == pytest.approx(1.0, rel=1e-6)


def test_speeding_confidence_defaults_high_without_magnitude_info():
    ev = ViolationEvent(vehicle_id=1, violation_type=ViolationType.SPEEDING, key_frame=0,
                        confidence=1.0, details={})   # no over_by -> recall-first, don't bury it
    assert vx.normalize_confidence(ev) == 1.0


# --------------------------------------------------------------------------- #
# HARD TYPE TIERS -- the type dominates; confidence only orders within a tier
# --------------------------------------------------------------------------- #
def test_tiers_are_solid_then_speeding_then_other_then_yellow():
    assert vx.tier_for(_crossing(1, 0.5)) == 0
    assert vx.tier_for(_speeding(1, over_by=10)) == 1
    assert vx.tier_for(ViolationEvent(1, "SOME_NEW_RULE", 0)) == vx.DEFAULT_TIER == 2
    assert vx.tier_for(_yellow(1, 10)) == 3


def test_type_dominates_confidence_across_tiers():
    # a barely-confident crossing STILL outranks a maximally-confident speeder (type beats confidence)
    weak_crossing = _crossing(1, conf=0.10)
    brutal_speeder = _speeding(2, over_by=60)             # confidence ~1.0
    assert _order([brutal_speeder, weak_crossing]) == [1, 2]


def test_within_crossing_tier_orders_by_confidence_desc():
    assert _order([_crossing(1, 0.3), _crossing(2, 0.9), _crossing(3, 0.6)]) == [2, 3, 1]


def test_within_speeding_tier_brutal_outranks_marginal():
    assert _order([_speeding(1, over_by=3), _speeding(2, over_by=60)]) == [2, 1]


def test_yellow_tier_is_chronological_not_by_confidence():
    # yellow ignores confidence -> ordered by key_frame ascending regardless of conf
    out = _order([_yellow(1, key_frame=900, conf=0.99), _yellow(2, key_frame=100, conf=0.01)])
    assert out == [2, 1]


def test_full_queue_is_lexicographic_across_all_tiers():
    events = [_yellow(40, key_frame=10), _speeding(30, over_by=5),
              _crossing(20, conf=0.4), _crossing(10, conf=0.95)]
    # crossings (by conf) -> speeding -> yellow
    assert _order(events) == [10, 20, 30, 40]


def test_unknown_type_gets_neutral_default_severity_weight():
    ev = ViolationEvent(vehicle_id=1, violation_type="SOME_NEW_RULE", key_frame=0, confidence=1.0)
    assert vx.severity_for(ev) == vx.DEFAULT_SEVERITY_WEIGHT     # severity kept as informational only


# --------------------------------------------------------------------------- #
# plate confidence MUST NOT influence ranking
# --------------------------------------------------------------------------- #
def test_plate_confidence_excluded_from_ranking():
    # two identical-tier, identical-confidence crossings; only plate evidence differs.
    crisp = (_crossing(7, conf=0.6, key_frame=100), _evidence(plate="11-111-11", plate_score=0.99))
    unreadable = (_crossing(8, conf=0.6, key_frame=200),
                  _evidence(plate=None, plate_score=0.0, manual_review=True))
    bundle = vx.build_export([crisp, unreadable], png_encoder=_fake_encoder, created_utc=0.0)
    by_id = {v["vehicle_id"]: v for v in bundle.violations}
    # identical tier + confidence -> the tie-break is key_frame, NOT plate quality
    assert by_id[7]["detector_confidence"] == by_id[8]["detector_confidence"]
    assert by_id[7]["plate_score"] == 0.99            # plate still travels as evidence
    assert by_id[8]["manual_review"] is True


# --------------------------------------------------------------------------- #
# manifest assembly + payload fields
# --------------------------------------------------------------------------- #
def test_build_export_sorts_worst_first_and_registers_files():
    records = [
        (_speeding(2, over_by=3), _evidence(frame_ids=(1, 2, 3))),     # tier 1
        (_crossing(1, conf=0.95), _evidence(frame_ids=(4, 5, 6))),     # tier 0 -> first
    ]
    bundle = vx.build_export(records, video_meta={"filename": "clip.mp4", "width": 1920},
                             png_encoder=_fake_encoder, created_utc=123.0)
    m = bundle.manifest
    assert m["manifest_version"] == vx.MANIFEST_VERSION
    assert m["created_utc"] == 123.0
    assert m["source_video"]["filename"] == "clip.mp4"
    assert m["violation_count"] == 2
    assert [v["vehicle_id"] for v in m["violations"]] == [1, 2]   # crossing (tier 0) first
    assert [v["tier"] for v in m["violations"]] == [0, 1]
    assert m["priority_model"]["model"] == "lexicographic_type_tier"
    assert m["priority_model"]["plate_confidence_excluded"] is True

    import hashlib
    for v in m["violations"]:
        for crop in v["evidence"]["crops"]:
            assert crop["file"] in bundle.files
            assert hashlib.sha256(bundle.files[crop["file"]]).hexdigest() == crop["sha256"]
        plate = v["evidence"]["plate_crop"]
        assert plate["file"] in bundle.files


def test_payload_fields_and_crossing_confidence_semantics():
    records = [(_crossing(1, conf=0.8), _evidence(plate="AA-11", plate_score=0.7)),
               (_speeding(2, over_by=20), _evidence(plate=None, plate_score=0.0, manual_review=True))]
    bundle = vx.build_export(records, png_encoder=_fake_encoder, created_utc=0.0)
    by_id = {v["vehicle_id"]: v for v in bundle.violations}
    crossing = by_id[1]
    # agreed flat payload fields present on the row
    for fld in ("vehicle_id", "violation", "plate", "plate_score", "n_reads", "manual_review",
                "crossing_solid_line_confidence"):
        assert fld in crossing
    assert crossing["violation"] == ViolationType.SOLID_LINE_CROSSING
    assert crossing["plate"] == "AA-11" and crossing["plate_score"] == 0.7
    # crossing confidence is set ONLY on a crossing; null on the speeder
    assert crossing["crossing_solid_line_confidence"] == pytest.approx(0.8)
    assert by_id[2]["crossing_solid_line_confidence"] is None


def test_build_export_folds_in_the_evidence_clip():
    from violations.clip_extract import ClipAsset, clip_window
    win = clip_window(key_frame=300, fps=30.0)
    clip = ClipAsset(data=b"\x00\x01MP4-EVIDENCE", window=win, container="mp4")
    record = (_crossing(5, conf=0.9, key_frame=300), _evidence(), clip)

    bundle = vx.build_export([record], png_encoder=_fake_encoder, created_utc=0.0)
    clip_meta = bundle.violations[0]["evidence"]["clip"]
    assert clip_meta is not None
    assert clip_meta["file"] in bundle.files
    assert bundle.files[clip_meta["file"]] == b"\x00\x01MP4-EVIDENCE"
    import hashlib
    assert clip_meta["sha256"] == hashlib.sha256(b"\x00\x01MP4-EVIDENCE").hexdigest()
    assert clip_meta["window"]["key_frame"] == 300
    assert clip_meta["window"]["start_frame"] == 0          # clamped (key_frame 300 < 10s*30)
    assert clip_meta["window"]["end_frame"] == 300 + 150


def test_build_export_folds_in_the_speeding_docx_report():
    from violations import docx_report
    ev = _speeding(9, over_by=46, key_frame=1200)
    report = docx_report.speeding_report(ev)
    bundle = vx.build_export([(ev, _evidence(), None, report)],
                             png_encoder=_fake_encoder, created_utc=0.0)
    rep_meta = bundle.violations[0]["evidence"]["report"]
    assert rep_meta is not None and rep_meta["file"].endswith("report.docx")
    assert rep_meta["kind"] == "speeding_docx"
    data = bundle.files[rep_meta["file"]]
    import hashlib
    assert hashlib.sha256(data).hexdigest() == rep_meta["sha256"]
    assert data[:2] == b"PK"                                 # a .docx is a zip


def test_build_export_handles_missing_evidence():
    bundle = vx.build_export([(_crossing(1, conf=0.5), None)],
                             png_encoder=_fake_encoder, created_utc=0.0)
    rec = bundle.violations[0]
    ev = rec["evidence"]
    assert ev["plate"] is None and ev["manual_review"] is True and ev["crops"] == []
    # no evidence media, but the violation is still reported AND self-described by its two documents
    assert set(bundle.files) == {rec["record_file"], rec["summary_file"]}
    assert rec["record_file"].endswith("violation.json") and rec["summary_file"].endswith(".txt")
    assert rec["tier"] == 0


def test_each_crop_carries_its_own_plate_crop():
    ev = SimpleNamespace(plate="AA-11", plate_score=0.7, n_reads=2, manual_review=False,
                         evidence_crops=["veh0", "veh1"], evidence_frame_ids=[10, 11],
                         evidence_plate_crops=["plate0", "plate1"], plate_crop=None)
    bundle = vx.build_export([(_crossing(5, conf=0.8), ev)],
                             png_encoder=_fake_encoder, created_utc=0.0)
    crops = bundle.violations[0]["evidence"]["crops"]
    assert len(crops) == 2
    for c in crops:                                          # one plate pic per vehicle pic
        assert c["file"].endswith(".png") and c["file"] in bundle.files
        assert c["plate"]["file"].endswith("_plate.png") and c["plate"]["file"] in bundle.files


def test_summary_txt_carries_the_reference_key_value_fields():
    ev = _evidence(plate="85-082-71", plate_score=0.883, crops=("a",))
    bundle = vx.build_export([(_speeding(7, over_by=46), ev)],
                             png_encoder=_fake_encoder, created_utc=0.0)
    rec = bundle.violations[0]
    txt = bundle.files[rec["summary_file"]].decode("utf-8")
    for key in ("vehicle_id=7", "violation=SPEEDING", "plate=85-082-71",
                "plate_score=0.883", "manual_review=False", "description="):
        assert key in txt


def test_each_violation_gets_a_self_describing_json_sidecar():
    records = [(_crossing(1, conf=0.8, key_frame=300), _evidence(plate="AA-11")),
               (_speeding(2, over_by=46, key_frame=1200), _evidence(plate=None, manual_review=True))]
    bundle = vx.build_export(records, png_encoder=_fake_encoder, created_utc=0.0)
    for rec in bundle.violations:
        path = rec["record_file"]
        assert path == f"{rec['violation_id']}/violation.json" and path in bundle.files
        sidecar = json.loads(bundle.files[path].decode("utf-8"))
        # the sidecar IS the manifest record for that violation (self-contained per folder)
        assert sidecar["violation"] == rec["violation"]
        assert sidecar["vehicle_id"] == rec["vehicle_id"]
        assert sidecar["description"] and rec["description"] == sidecar["description"]
    # the human-readable description names the actual violation
    by_id = {v["vehicle_id"]: v for v in bundle.violations}
    assert "solid white" in by_id[1]["description"].lower()
    assert "speed" in by_id[2]["description"].lower() and "km/h" in by_id[2]["description"]


# --------------------------------------------------------------------------- #
# compression / bundling
# --------------------------------------------------------------------------- #
def test_bundle_targz_contains_manifest_and_every_file():
    bundle = vx.build_export([(_crossing(1, conf=0.9), _evidence(frame_ids=(7, 8, 9)))],
                             png_encoder=_fake_encoder, created_utc=0.0)
    blob = vx.bundle_targz(bundle)
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = set(tar.getnames())
        assert vx.MANIFEST_NAME in names
        for path in bundle.files:
            assert path in names
        manifest_in_tar = json.loads(tar.extractfile(vx.MANIFEST_NAME).read().decode("utf-8"))
        assert manifest_in_tar["violation_count"] == 1


def test_bundle_targz_is_byte_reproducible():
    records = [(_crossing(1, conf=0.9), _evidence())]
    a = vx.bundle_targz(vx.build_export(records, png_encoder=_fake_encoder, created_utc=5.0))
    b = vx.bundle_targz(vx.build_export(records, png_encoder=_fake_encoder, created_utc=5.0))
    assert a == b


# --------------------------------------------------------------------------- #
# HTTP transport (mocked)
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class _FakeSession:
    """Captures the last .post(...) call so the multipart payload can be asserted."""
    def __init__(self, status_code=200, text="ok"):
        self._resp = _FakeResponse(status_code, text)
        self.calls = []

    def post(self, url, files=None, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "files": files, "data": data,
                           "headers": headers, "timeout": timeout})
        return self._resp


def test_post_bundle_builds_multipart_and_reports_success():
    bundle = vx.build_export([(_crossing(1, conf=0.9), _evidence())],
                             png_encoder=_fake_encoder, created_utc=0.0)
    session = _FakeSession(status_code=201, text="created")
    resp = vx.post_bundle("https://ingest.example/api/v1/violations", bundle,
                          session=session, extra_fields={"clip_id": "abc"}, timeout=12)

    assert resp.ok and resp.status_code == 201
    call = session.calls[0]
    assert call["url"].endswith("/violations")
    assert call["timeout"] == 12
    fname, payload, ctype = call["files"]["bundle"]
    assert fname == "violations.tar.gz" and ctype == "application/gzip"
    assert isinstance(payload, bytes) and payload[:2] == b"\x1f\x8b"   # gzip magic
    assert call["data"]["clip_id"] == "abc"
    assert call["data"]["manifest_version"] == vx.MANIFEST_VERSION


def test_post_bundle_accepts_precompressed_bytes_and_flags_http_error():
    raw = vx.bundle_targz(vx.build_export([(_crossing(1, conf=0.9), _evidence())],
                                          png_encoder=_fake_encoder, created_utc=0.0))
    session = _FakeSession(status_code=500, text="boom")
    resp = vx.post_bundle("https://ingest.example/x", raw, session=session)
    assert session.calls[0]["files"]["bundle"][1] == raw
    assert resp.ok is False and resp.status_code == 500
