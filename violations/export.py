"""
violation_export -- bundle violations + evidence into one prioritised payload for the backend.

This is the CLIENT-SIDE (vision-pipeline) half of the Road Guard backend integration contract.
The live pipeline produces ``ViolationEvent`` records (violations/event.py) and, for each one,
an ``EvidenceResult`` (lpr/evidence.py: voted plate + best-picture crops). This module turns a
batch of those into:

  1. a ``manifest.json`` describing every violation, RANKED by priority, and
  2. a single compressed bundle (``.tar.gz``) carrying that manifest plus the lossless PNG crop
     evidence, the short evidence VIDEO clip, and (for speeding) a ``.docx`` report, ready to POST
     to a generic ingest endpoint.

PRIORITY -- what the human-review queue sorts on (HARD TYPE-TIERED / LEXICOGRAPHIC)
-----------------------------------------------------------------------------------
The review queue is ordered by an ABSOLUTE type hierarchy; the detector's confidence only orders
incidents WITHIN a tier, never across tiers. The type dominates: a solid-line crossing at
confidence 0.10 still outranks a speeder at 0.99.

    sort key = (TYPE_TIER[type], <intra-tier order>)

  Tiers (``TYPE_TIER``), best-first:
    0  SOLID_LINE_CROSSING  -- always first; ordered within by its crossing confidence DESC
    1  SPEEDING             -- second; ordered within by confidence DESC, where speeding's
                               confidence is derived from the MARGIN over the limit
                               (``normalize_confidence`` -> a 3 km/h margin is estimator noise =
                               low confidence; a 40+ km/h margin = full confidence)
    2  <any other type>     -- middle band; ordered within by confidence DESC
    3  YELLOW_LINE_RIGHT    -- last; NO confidence ranking -- ordered CHRONOLOGICALLY (key_frame)

This REPLACES the earlier multiplicative ``confidence * severity`` blend: severity no longer
interleaves types -- the TIER does. ``SEVERITY_WEIGHTS`` is kept below as INFORMATIONAL metadata
only (stamped into the manifest for the backend's reference) and is NOT consulted when ordering.

PLATE CONFIDENCE IS DELIBERATELY EXCLUDED FROM PRIORITY.
  A plate read is EVIDENCE for the human reviewer, never a reason to rank a violation up or down.
  An unreadable plate (HITL: "UNKNOWN - manual review") must NOT bury a brutal violation, and a
  crisp plate read must NOT promote a marginal one. ``plate_score`` rides in the manifest only as
  evidence and is never read by the sort key.

Kept dependency-light on purpose: the scoring + manifest logic is pure stdlib so it unit-tests on
any interpreter. ``cv2`` (PNG encode) and ``requests`` (HTTP) are lazy-imported and injectable, so
neither is needed to test the math, the bundling, or the payload shape.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from violations.event import ViolationEvent, ViolationType

MANIFEST_VERSION = "2.0"   # 2.0: lexicographic type-tiered priority (was 1.0: confidence x severity)

# --------------------------------------------------------------------------- #
# PRIORITY MODEL -- hard type tiers (lexicographic). The TYPE decides the band; confidence only
# orders incidents WITHIN a band. Lower tier number == higher enforcement priority (shown first).
# --------------------------------------------------------------------------- #
TYPE_TIER: dict[str, int] = {
    ViolationType.SOLID_LINE_CROSSING: 0,   # ALWAYS first
    ViolationType.SPEEDING: 1,              # second, ordered by margin-derived confidence
    ViolationType.YELLOW_LINE_RIGHT: 3,     # LAST, chronological, no confidence ranking
}
DEFAULT_TIER = 2                            # any other / unknown type -> middle band, conf-ordered

# --------------------------------------------------------------------------- #
# Severity model -- INFORMATIONAL ONLY under the tiered priority above.
#
# Kept as a static per-type config table and stamped into the manifest so the backend has a tunable
# enforcement weight to reason about, but the SORT KEY no longer reads it (the tier does the
# type-level ordering). Left here so a future model that wants a softer blend has the anchors.
# --------------------------------------------------------------------------- #
SEVERITY_WEIGHTS: dict[str, float] = {
    ViolationType.SPEEDING: 1.00,
    ViolationType.WRONG_WAY: 1.00,
    ViolationType.SOLID_LINE_CROSSING: 0.80,
    ViolationType.TRAFFIC_ISLAND: 0.75,
    ViolationType.YELLOW_LINE_RIGHT: 0.70,
}
DEFAULT_SEVERITY_WEIGHT = 0.50

# Speeding has NO graded classifier score -- "the measured speed is over the limit" is a measurement.
# Its CONFIDENCE therefore comes from the MARGIN over the limit: a few km/h over is within the
# monocular estimator's noise (low confidence it is a real violation), a large margin is unambiguous
# (high confidence). This lands speeding on the SAME [0, 1] confidence axis as the graded lane rules,
# so it sorts sensibly WITHIN the speeding tier. (Mirrors overspeed.py's OVERSPEED_MARGIN_KMH, which
# exists precisely because the estimator over-estimates, so a small margin is the unreliable case.)
SPEEDING_CONF_FLOOR = 0.30                    # confidence when barely over the limit
SPEEDING_CONF_FULL_OVER_KMH = 40.0           # >= this many km/h over the limit -> full confidence


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return lo
    if x != x:                                # NaN guard
        return lo
    return lo if x < lo else hi if x > hi else x


def _speeding_confidence(details: dict) -> float:
    """Speeding's confidence in [SPEEDING_CONF_FLOOR, 1.0], ramped from how far over the limit it was.
    Missing magnitude -> 1.0 (recall-first: never bury a violation we cannot grade)."""
    over = details.get("over_by_kmh")
    if over is None:                          # fall back to est - limit if the explicit field is absent
        est, lim = details.get("est_speed_kmh"), details.get("speed_limit_kmh")
        if est is not None and lim is not None:
            over = est - lim
    if over is None:
        return 1.0
    over = max(0.0, float(over))
    ramp = _clamp(over / SPEEDING_CONF_FULL_OVER_KMH)
    return SPEEDING_CONF_FLOOR + (1.0 - SPEEDING_CONF_FLOOR) * ramp


def normalize_confidence(event: ViolationEvent) -> float:
    """Detector confidence on one comparable [0, 1] axis for EVERY rule (used to order WITHIN a tier).

    Graded rules (yellow-line, crossing) already emit a calibrated probability -> clamp it.
    Speeding is a measurement, not a classifier (it emits a flat 1.0), so its real confidence is
    derived from the over-limit MARGIN here. A missing confidence on a non-speeding rule defaults to
    1.0 (the ViolationEvent default for a hard/binary rule).
    """
    if event.violation_type == ViolationType.SPEEDING:
        return _speeding_confidence(event.details or {})
    conf = getattr(event, "confidence", 1.0)
    if conf is None:
        conf = 1.0
    return _clamp(conf)


def severity_for(event: ViolationEvent) -> float:
    """Static per-type enforcement weight in (0, 1] -- INFORMATIONAL ONLY (stamped in the manifest,
    NOT read by the sort key under the tiered model). A pure config-table lookup."""
    return _clamp(SEVERITY_WEIGHTS.get(event.violation_type, DEFAULT_SEVERITY_WEIGHT))


def tier_for(event: ViolationEvent) -> int:
    """The hard priority TIER of an event's type (lower == higher priority, shown first)."""
    return TYPE_TIER.get(event.violation_type, DEFAULT_TIER)


def sort_key(event: ViolationEvent) -> tuple:
    """Lexicographic sort key -- ascending order == review queue order (worst/most-urgent first).

    Element 0 is the TYPE TIER (dominates absolutely). Within a tier:
      * YELLOW_LINE_RIGHT  -> ordered CHRONOLOGICALLY by key_frame (no confidence ranking),
      * every other tier   -> ordered by confidence DESC (so -conf ascending).
    key_frame then vehicle_id break any remaining ties for a stable, deterministic order.
    """
    tier = tier_for(event)
    kf = int(getattr(event, "key_frame", 0) or 0)
    vid = int(getattr(event, "vehicle_id", 0) or 0)
    if event.violation_type == ViolationType.YELLOW_LINE_RIGHT:
        return (tier, float(kf), kf, vid)              # chronological within the yellow tier
    return (tier, -normalize_confidence(event), kf, vid)   # confidence DESC within all other tiers


def describe_violation(event: ViolationEvent) -> str:
    """One human-readable sentence describing the violation -- the SINGLE source of truth shared by
    the per-violation ``violation.json`` and the burnt-in caption on the annotated clip."""
    vid = getattr(event, "vehicle_id", "?")
    d = event.details or {}
    t = event.violation_type
    if t == ViolationType.SOLID_LINE_CROSSING:
        return (f"Vehicle {vid} crossed a solid white lane line "
                f"(crossing confidence {normalize_confidence(event):.2f}).")
    if t == ViolationType.SPEEDING:
        est, lim = d.get("est_speed_kmh"), d.get("speed_limit_kmh")
        over = d.get("over_by_kmh")
        if over is None and est is not None and lim is not None:
            over = est - lim
        if est is not None and lim is not None:
            return (f"Vehicle {vid} was speeding: ~{float(est):.0f} km/h in a {float(lim):.0f} km/h "
                    f"zone (+{float(over):.0f} km/h over the limit).")
        return f"Vehicle {vid} was speeding (over the posted limit)."
    if t == ViolationType.YELLOW_LINE_RIGHT:
        return f"Vehicle {vid} drove on/over the solid yellow (right-shoulder) line."
    if t == ViolationType.WRONG_WAY:
        return f"Vehicle {vid} was driving the wrong way against traffic."
    if t == ViolationType.TRAFFIC_ISLAND:
        return f"Vehicle {vid} drove over a traffic island."
    return f"Vehicle {vid} committed a {t} violation."


# --------------------------------------------------------------------------- #
# PNG encoding -- injectable so the bundling logic tests without cv2/numpy.
# --------------------------------------------------------------------------- #
PngEncoder = Callable[[Any], bytes]


def _default_png_encoder(crop: Any) -> bytes:
    """Encode a BGR ndarray to LOSSLESS PNG bytes via cv2 (lazy import). Evidence is lossless on
    purpose: a reviewer (and any downstream re-OCR) must see the pixels the detector saw."""
    import cv2
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("cv2.imencode failed to encode crop to PNG")
    return buf.tobytes()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _slug(violation_type: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(violation_type))


# --------------------------------------------------------------------------- #
# Export records + manifest assembly.
# --------------------------------------------------------------------------- #
@dataclass
class ExportBundle:
    """The complete client-side payload: the manifest dict + every file it references.

    ``files`` maps the in-bundle path (as named in the manifest) to its raw bytes. Keeping them
    separate keeps the manifest pure JSON and lets callers inspect/serialise either half on its own.
    """
    manifest: dict
    files: dict[str, bytes] = field(default_factory=dict)

    @property
    def violations(self) -> list:
        return self.manifest.get("violations", [])


def _crop_entries(event: ViolationEvent, evidence, png_encoder: PngEncoder,
                  stem: str, files: dict) -> tuple[list, Optional[dict]]:
    """Encode an EvidenceResult's crops to PNG, register them in ``files``, and return the manifest
    crop descriptors. Reference layout: each vehicle picture ``<base>_<i>_f<frame>.png`` is paired
    with its plate crop ``<base>_<i>_f<frame>_plate.png`` (when a plate crop is available)."""
    crops: list = []
    base = f"v{event.vehicle_id}_{_slug(event.violation_type)}"
    crop_imgs = getattr(evidence, "evidence_crops", None) or []
    plate_imgs = getattr(evidence, "evidence_plate_crops", None) or []
    frame_ids = getattr(evidence, "evidence_frame_ids", None) or []
    for i, crop in enumerate(crop_imgs):
        frame_id = frame_ids[i] if i < len(frame_ids) else None
        png = png_encoder(crop)
        fstem = f"{base}_{i}" + (f"_f{frame_id}" if frame_id is not None else "")
        vpath = f"{stem}/{fstem}.png"
        files[vpath] = png
        entry = {"file": vpath, "frame_id": frame_id, "bytes": len(png), "sha256": _sha256_hex(png)}
        # the plate crop for THIS picture (one plate pic per vehicle pic, as in the reference layout)
        if i < len(plate_imgs) and plate_imgs[i] is not None:
            ppng = png_encoder(plate_imgs[i])
            ppath = f"{stem}/{fstem}_plate.png"
            files[ppath] = ppng
            entry["plate"] = {"file": ppath, "bytes": len(ppng), "sha256": _sha256_hex(ppng)}
        crops.append(entry)

    # Legacy single "best" plate crop (lpr.evidence.EvidenceResult.plate_crop), kept for back-compat.
    plate_entry = None
    plate_crop = getattr(evidence, "plate_crop", None)
    if plate_crop is not None and getattr(plate_crop, "size", 1):
        png = png_encoder(plate_crop)
        path = f"{stem}/plate.png"
        files[path] = png
        plate_entry = {"file": path, "bytes": len(png), "sha256": _sha256_hex(png)}
    # Unconditional fallback: when no dedicated plate crop exists (FastALPR found nothing),
    # export the sharpest vehicle crop as vehicle_crop.png so the authority dashboard always
    # has an image for manual plate review instead of a blank card.
    if plate_entry is None and crop_imgs:
        png = png_encoder(crop_imgs[0])
        path = f"{stem}/vehicle_crop.png"
        files[path] = png
        plate_entry = {"file": path, "bytes": len(png), "sha256": _sha256_hex(png)}
    return crops, plate_entry


def _summary_txt(rec: dict) -> str:
    """Flat key=value human-readable document for one violation (the reference ``v{id}_{TYPE}.txt``).
    Present in EVERY violation folder so each is self-describing at a glance."""
    lines = []
    for k in ("vehicle_id", "violation", "plate", "plate_score", "n_reads", "manual_review",
              "crossing_solid_line_confidence", "tier", "detector_confidence"):
        v = rec.get(k)
        if k == "plate" and v is None:
            v = "UNKNOWN"
        lines.append(f"{k}={v}")
    det = rec.get("details") or {}
    for dk in ("est_speed_kmh", "speed_limit_kmh", "over_by_kmh", "shoulder_overlap_frac"):
        if det.get(dk) is not None:
            lines.append(f"{dk}={det[dk]}")
    lines.append(f"description={rec.get('description', '')}")
    return "\n".join(lines) + "\n"


def _clip_entry(clip, stem: str, files: dict) -> Optional[dict]:
    """Register the evidence VIDEO clip (10s pre / 5s post the incident) in ``files`` and return its
    manifest descriptor. ``clip`` is a clip_extract.ClipAsset (or any object with ``read_bytes`` /
    ``window`` / ``container``); ``None`` when no clip was cut."""
    if clip is None:
        return None
    container = getattr(clip, "container", "mp4") or "mp4"
    data = clip.read_bytes() if hasattr(clip, "read_bytes") else bytes(clip)
    path = f"{stem}/clip.{container}"
    files[path] = data
    window = getattr(clip, "window", None)
    return {
        "file": path,
        "bytes": len(data),
        "sha256": _sha256_hex(data),
        "container": container,
        "recompressed": bool(getattr(clip, "recompressed", False)),
        "window": window.to_dict() if hasattr(window, "to_dict") else window,
    }


def _report_entry(report, stem: str, files: dict) -> Optional[dict]:
    """Register a per-violation document (the speeding ``.docx``) in ``files`` and return its
    manifest descriptor. ``report`` is raw bytes (or an object with ``read_bytes``); ``None`` when
    no report applies (e.g. a non-speeding violation)."""
    if not report:
        return None
    data = (report.read_bytes() if hasattr(report, "read_bytes")
            else bytes(report))
    path = f"{stem}/report.docx"
    files[path] = data
    return {"file": path, "bytes": len(data), "sha256": _sha256_hex(data),
            "kind": "speeding_docx",
            "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


def _evidence_block(event: ViolationEvent, evidence, png_encoder: PngEncoder,
                    stem: str, files: dict, clip=None, report=None) -> dict:
    """The manifest's per-violation ``evidence`` block. Plate read is EVIDENCE ONLY -- its score
    is recorded here for the reviewer and is never consulted by the priority sort. ``clip`` is the
    short evidence video around the incident; ``report`` is the speeding ``.docx`` (when applicable)."""
    clip_entry = _clip_entry(clip, stem, files)
    report_entry = _report_entry(report, stem, files)
    if evidence is None:
        return {"plate": None, "plate_score": None, "n_reads": 0, "manual_review": True,
                "crops": [], "plate_crop": None, "clip": clip_entry, "report": report_entry,
                "note": "no plate/crop evidence collected (recall-first: violation still reported)"}
    crops, plate_entry = _crop_entries(event, evidence, png_encoder, stem, files)
    return {
        "plate": getattr(evidence, "plate", None),
        # plate_score is EVIDENCE QUALITY, NOT a ranking input -- kept out of the sort key by design.
        "plate_score": getattr(evidence, "plate_score", None),
        "n_reads": getattr(evidence, "n_reads", None),
        "manual_review": bool(getattr(evidence, "manual_review", evidence is None
                                      or not getattr(evidence, "plate", None))),
        "clip": clip_entry,          # the trimmed evidence video (primary artefact)
        "report": report_entry,      # speeding .docx (None for non-speeding)
        "crops": crops,              # quick-look stills (3 high-res frames of the vehicle)
        "plate_crop": plate_entry,
    }


def violation_id(event: ViolationEvent) -> str:
    """Stable, human-legible id for one incident: one vehicle, one type, one key frame."""
    return f"v{event.vehicle_id}_{_slug(event.violation_type)}_f{event.key_frame}"


def build_export(records: Iterable[tuple], video_meta: Optional[dict] = None, *,
                 client_info: Optional[dict] = None,
                 png_encoder: PngEncoder = _default_png_encoder,
                 created_utc: Optional[float] = None) -> ExportBundle:
    """Assemble the ranked manifest + evidence files from ``(ViolationEvent, EvidenceResult|None)``.

    Args:
        records:      iterable of ``(event, evidence)``, ``(event, evidence, clip)`` or
                      ``(event, evidence, clip, report)``. ``evidence`` may be ``None`` (the
                      violation is still reported -- recall-first -- with an empty evidence block).
                      ``clip`` (a clip_extract.ClipAsset) is the trimmed evidence video; ``report``
                      is the speeding ``.docx`` bytes. Both optional.
        video_meta:   source-video integrity dict (see violations.video_integrity.VideoMeta.to_dict).
        client_info:  optional {app, version, ...} stamped into the manifest.
        png_encoder:  crop ndarray -> PNG bytes. Injectable; default uses cv2 (lossless). When crops
                      are ALREADY PNG bytes (e.g. extracted with ffmpeg), pass ``bytes`` as identity.
        created_utc:  override timestamp (for reproducible tests).

    Returns:
        ExportBundle(manifest, files). ``manifest["violations"]`` is sorted by the lexicographic
        :func:`sort_key`, so the backend can fan the queue out worst-first without re-ranking:
        ALL crossings first (by confidence), then ALL speeders (by margin-confidence), then the
        middle band, then yellow-line (chronological).
    """
    files: dict[str, bytes] = {}
    scored: list[tuple] = []     # (sort_key, record_dict)
    for record in records:
        event, evidence = record[0], record[1]
        clip = record[2] if len(record) > 2 else None      # optional trimmed evidence video
        report = record[3] if len(record) > 3 else None     # optional speeding .docx
        conf = normalize_confidence(event)
        stem = violation_id(event)
        is_crossing = event.violation_type == ViolationType.SOLID_LINE_CROSSING
        ev_block = _evidence_block(event, evidence, png_encoder, stem, files, clip=clip, report=report)
        rec = {
            "violation_id": stem,
            "vehicle_id": event.vehicle_id,
            "violation": event.violation_type,            # backend payload field
            "violation_type": event.violation_type,       # alias kept for back-compat
            "tier": tier_for(event),
            "key_frame": event.key_frame,
            "detector_confidence": round(conf, 6),
            # crossing confidence is meaningful ONLY for a solid-line crossing; null otherwise
            # (e.g. a speeder that crossed no solid white line).
            "crossing_solid_line_confidence": round(conf, 6) if is_crossing else None,
            # flat payload fields the backend reads directly (mirrored from the evidence block):
            "plate": ev_block.get("plate"),
            "plate_score": ev_block.get("plate_score"),   # EVIDENCE quality, excluded from ranking
            "n_reads": ev_block.get("n_reads"),
            "manual_review": ev_block.get("manual_review"),
            "severity_weight": round(severity_for(event), 6),   # informational only
            "description": describe_violation(event),     # human-readable, shared with the clip caption
            "details": dict(event.details or {}),
            "evidence": ev_block,
        }
        # Per-violation documents: each folder is self-describing so a reviewer can read one folder
        # in isolation -- a human-readable .txt (reference layout) AND a machine-readable .json that
        # carries the exact record the manifest lists for this violation.
        base = f"v{event.vehicle_id}_{_slug(event.violation_type)}"
        rec["record_file"] = f"{stem}/violation.json"
        rec["summary_file"] = f"{stem}/{base}.txt"
        files[rec["summary_file"]] = _summary_txt(rec).encode("utf-8")
        files[rec["record_file"]] = json.dumps(rec, ensure_ascii=False, indent=2).encode("utf-8")
        scored.append((sort_key(event), rec))

    scored.sort(key=lambda t: t[0])                        # ascending lexicographic == queue order
    violations = [rec for _, rec in scored]

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": float(created_utc if created_utc is not None else time.time()),
        "client": client_info or {"app": "roadguard", "component": "violation_export"},
        "source_video": dict(video_meta) if video_meta else None,
        "priority_model": {
            "model": "lexicographic_type_tier",
            "summary": "Hard type tiers; confidence orders incidents WITHIN a tier only, never "
                       "across tiers. Solid-line crossing > Speeding > <other> > Yellow-line.",
            "tiers": [
                {"rank": 0, "type": ViolationType.SOLID_LINE_CROSSING, "intra_order": "confidence_desc"},
                {"rank": 1, "type": ViolationType.SPEEDING,
                 "intra_order": "confidence_desc (margin-over-limit)"},
                {"rank": 2, "type": "<other>", "intra_order": "confidence_desc"},
                {"rank": 3, "type": ViolationType.YELLOW_LINE_RIGHT,
                 "intra_order": "chronological (key_frame asc, no confidence ranking)"},
            ],
            "type_tier": dict(TYPE_TIER),
            "default_tier": DEFAULT_TIER,
            "speeding_confidence_from_margin": {
                "floor": SPEEDING_CONF_FLOOR,
                "full_over_kmh": SPEEDING_CONF_FULL_OVER_KMH,
            },
            "severity_weights_informational": dict(SEVERITY_WEIGHTS),
            "plate_confidence_excluded": True,
        },
        "violation_count": len(violations),
        "violations": violations,
    }
    return ExportBundle(manifest=manifest, files=files)


# --------------------------------------------------------------------------- #
# Compression + transport.
# --------------------------------------------------------------------------- #
MANIFEST_NAME = "manifest.json"


def bundle_targz(bundle: ExportBundle, *, mtime: float = 0.0) -> bytes:
    """Pack the manifest + every evidence file into ONE gzip-compressed tar (.tar.gz) byte string.

    One stream keeps the manifest and the exact bytes it references together (their sha256s match
    on the far side), and gzip shrinks the JSON; the PNGs/clip ride along losslessly. ``mtime=0``
    makes the archive byte-reproducible for tests.
    """
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as tar:
        def _add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = mtime
            tar.addfile(info, io.BytesIO(data))

        manifest_bytes = json.dumps(bundle.manifest, ensure_ascii=False, indent=2).encode("utf-8")
        _add(MANIFEST_NAME, manifest_bytes)
        for path, data in sorted(bundle.files.items()):   # sorted -> deterministic archive
            _add(path, data)
    return raw.getvalue()


def gzip_manifest(bundle: ExportBundle) -> bytes:
    """The manifest alone, gzip-compressed (when a caller wants metadata without the heavy files)."""
    return gzip.compress(json.dumps(bundle.manifest, ensure_ascii=False, indent=2).encode("utf-8"))


@dataclass
class IngestResponse:
    """Outcome of the upload, decoupled from the underlying HTTP client."""
    ok: bool
    status_code: int
    body: str = ""


def post_bundle(url: str, bundle: ExportBundle | bytes, *,
                session: Any = None,
                field_name: str = "bundle",
                filename: str = "violations.tar.gz",
                extra_fields: Optional[dict] = None,
                headers: Optional[dict] = None,
                timeout: float = 30.0) -> IngestResponse:
    """POST the bundle to a generic ingest endpoint as ``multipart/form-data``.

    The compressed archive rides in the ``field_name`` file part; ``extra_fields`` become sibling
    form fields (e.g. a clip id, an auth/idempotency token, the manifest version). ``session`` is
    injectable -- pass anything exposing ``.post(url, files=, data=, headers=, timeout=)`` (a
    ``requests.Session`` in production, a fake/mock in tests). Defaults to a one-shot ``requests``
    call, lazy-imported so this module imports without ``requests`` present.

    Returns an IngestResponse; raises nothing for HTTP error codes (the caller decides what a
    non-2xx means for retry/queue policy) -- only a transport-level failure propagates.
    """
    payload = bundle if isinstance(bundle, (bytes, bytearray)) else bundle_targz(bundle)
    files = {field_name: (filename, bytes(payload), "application/gzip")}
    data = dict(extra_fields or {})
    data.setdefault("manifest_version", MANIFEST_VERSION)

    if session is None:
        import requests                       # lazy: keep module importable without requests
        session = requests

    resp = session.post(url, files=files, data=data, headers=headers, timeout=timeout)
    status = int(getattr(resp, "status_code", 0))
    text = getattr(resp, "text", "") or ""
    return IngestResponse(ok=200 <= status < 300, status_code=status, body=text)


# --------------------------------------------------------------------------- #
# OPTION B -- Cloudflare presigned-URL upload (the architecture we now build to).
#
# The backend no longer stores the bytes: it issues a one-time presigned PUT URL (R2/S3) for the
# bundle and a lightweight webhook to be told the upload finished + the metadata. So instead of
# POSTing the heavy tarball THROUGH the Node server (`post_bundle`), the brain PUTs the tarball
# straight to object storage, then POSTs a small JSON notification to the backend.
#
#   1. put_bundle(presigned_put_url, bundle)   -> raw HTTP PUT of the .tar.gz to Cloudflare
#   2. notify_backend(webhook_url, metadata)   -> small JSON POST to the Node server
#   upload_bundle_presigned(...)               -> does both and returns both outcomes
#
# TTL trap (document for the backend): mint the upload URL AFTER analysis (or with a generous TTL).
# If the PUT URL is created at job dispatch but GPU analysis takes minutes, the URL can expire
# before the brain uploads. See the handoff doc section 3.
# --------------------------------------------------------------------------- #
def put_bundle(presigned_url: str, bundle: ExportBundle | bytes, *,
               session: Any = None,
               content_type: str = "application/gzip",
               headers: Optional[dict] = None,
               timeout: float = 120.0) -> IngestResponse:
    """HTTP PUT the compressed bundle straight to a presigned object-storage URL (Cloudflare R2 /
    S3). The bundle bytes are the raw request body -- NOT multipart (presigned PUT takes the object
    bytes verbatim). ``presigned_url`` is used exactly as given (it is already signed; appending to
    it breaks the signature). ``session`` is injectable (``requests``/``requests.Session`` in
    production, a fake in tests). Returns an IngestResponse; a non-2xx is reported, not raised."""
    payload = bundle if isinstance(bundle, (bytes, bytearray)) else bundle_targz(bundle)
    payload = bytes(payload)
    hdrs = {"Content-Type": content_type, "Content-Length": str(len(payload))}
    if headers:
        hdrs.update(headers)
    if session is None:
        import requests                       # lazy: keep module importable without requests
        session = requests
    resp = session.put(presigned_url, data=payload, headers=hdrs, timeout=timeout)
    status = int(getattr(resp, "status_code", 0))
    text = getattr(resp, "text", "") or ""
    return IngestResponse(ok=200 <= status < 300, status_code=status, body=text)


def build_notify_payload(bundle: ExportBundle | bytes, *,
                         job_id: Optional[str] = None,
                         source_video: Optional[str] = None,
                         object_key: Optional[str] = None,
                         object_url: Optional[str] = None,
                         extra: Optional[dict] = None) -> dict:
    """The small JSON the brain POSTs to the backend webhook once the bundle is in object storage.

    Carries everything the backend needs to record the violation set + locate/verify the uploaded
    object WITHOUT the heavy bytes ever transiting the Node server: the bundle sha256 + size (so the
    backend can verify the R2 object), where it was stored (key/url), the manifest version, the
    violation count, and the ranked violation index (ids/types/tiers/plates -> straight into Mongo).
    """
    if isinstance(bundle, (bytes, bytearray)):
        raw = bytes(bundle)
        manifest = None
        violations = []
    else:
        raw = bundle_targz(bundle)
        manifest = bundle.manifest
        violations = bundle.violations
    payload: dict = {
        "manifest_version": MANIFEST_VERSION,
        "job_id": job_id,
        "source_video": source_video,
        "bundle": {
            "object_key": object_key,
            "object_url": object_url,
            "bytes": len(raw),
            "sha256": _sha256_hex(raw),
            "content_type": "application/gzip",
            "filename": "violations.tar.gz",
        },
        "violation_count": len(violations),
        # a compact, already-ranked index so the backend can persist rows before/without unpacking
        "violations": [
            {"violation_id": v.get("violation_id"), "violation": v.get("violation"),
             "tier": v.get("tier"), "vehicle_id": v.get("vehicle_id"),
             "detector_confidence": v.get("detector_confidence"),
             "plate": v.get("plate"), "manual_review": v.get("manual_review")}
            for v in violations
        ],
    }
    if manifest is not None:
        payload["created_utc"] = manifest.get("created_utc")
        payload["priority_model"] = (manifest.get("priority_model") or {}).get("model")
    if extra:
        payload.update(extra)
    return payload


def notify_backend(webhook_url: str, payload: dict, *,
                   session: Any = None,
                   headers: Optional[dict] = None,
                   timeout: float = 30.0) -> IngestResponse:
    """POST the lightweight completion ``payload`` (see :func:`build_notify_payload`) as JSON to the
    backend webhook. ``session`` is injectable. Returns an IngestResponse (non-2xx reported, not
    raised, so the caller owns retry/queue policy)."""
    if session is None:
        import requests
        session = requests
    resp = session.post(webhook_url, json=payload, headers=headers, timeout=timeout)
    status = int(getattr(resp, "status_code", 0))
    text = getattr(resp, "text", "") or ""
    return IngestResponse(ok=200 <= status < 300, status_code=status, body=text)


@dataclass
class PresignedUploadResult:
    """Outcome of the two-step presigned upload: the object PUT + the backend notify."""
    put: IngestResponse
    notify: Optional[IngestResponse] = None
    bundle_bytes: int = 0
    bundle_sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.put.ok and (self.notify is None or self.notify.ok)


def upload_bundle_presigned(upload_url: str, bundle: ExportBundle | bytes, *,
                            notify_url: Optional[str] = None,
                            session: Any = None,
                            job_id: Optional[str] = None,
                            source_video: Optional[str] = None,
                            object_key: Optional[str] = None,
                            object_url: Optional[str] = None,
                            put_headers: Optional[dict] = None,
                            notify_headers: Optional[dict] = None,
                            extra_notify: Optional[dict] = None,
                            timeout: float = 120.0) -> PresignedUploadResult:
    """The full Option-B upload: PUT the bundle to ``upload_url`` (Cloudflare), then (if
    ``notify_url`` is given) POST the completion webhook to the backend. Notify is SKIPPED when the
    PUT failed (don't tell the backend an object exists when it doesn't). Returns a
    :class:`PresignedUploadResult` with both outcomes + the bundle's sha256/size."""
    raw = bytes(bundle if isinstance(bundle, (bytes, bytearray)) else bundle_targz(bundle))
    put = put_bundle(upload_url, raw, session=session, headers=put_headers, timeout=timeout)
    result = PresignedUploadResult(put=put, bundle_bytes=len(raw), bundle_sha256=_sha256_hex(raw))
    if notify_url and put.ok:
        payload = build_notify_payload(bundle, job_id=job_id, source_video=source_video,
                                       object_key=object_key, object_url=object_url,
                                       extra=extra_notify)
        result.notify = notify_backend(notify_url, payload, session=session,
                                       headers=notify_headers, timeout=timeout)
    return result
