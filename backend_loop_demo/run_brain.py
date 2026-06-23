"""
run_brain -- the CLIENT ("brain") side of the loop, end-to-end against a backend (real or the mock).

Flow (what `run_once` does):
  1. PULL    -- download the source video from the backend over HTTP and FAIL-FAST verify the copy
                against the backend's reference fingerprint (violations.ingest_client).
  2. ANALYSE -- produce the violation list. To keep the demo fast/deterministic and avoid a multi-
                minute GPU re-run, this loads the REAL cached analysis of DeNnDugXxP0 (a prior run --
                18 solid-line crossings) and takes the top-K by confidence. Because that clip has NO
                GPS and therefore NO speeding (and no yellow-line event), we INJECT one clearly
                labelled synthetic SPEEDING and one synthetic YELLOW event so all three priority
                tiers + the speeding .docx are actually exercised. Every synthetic event is flagged
                ``_synthetic: true`` in its details -- nothing fake is hidden.
  3. EVIDENCE-- for each selected violation, cut the 10s-pre / 5s-post clip (ffmpeg, lossless copy),
                grab 3 high-res frames as the "3 pictures", and for SPEEDING render a .docx report.
  4. BUNDLE  -- assemble the prioritised manifest (violations.export: hard type tiers, crossings ->
                speeding -> yellow) + all evidence into one .tar.gz.
  5. PUSH    -- POST the bundle back to the backend's /ingest endpoint.

Run it (Terminal 2, after mock_backend.py is up in Terminal 1)::

    python backend_loop_demo/run_brain.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from violations import docx_report, export as vx, ingest_client
from violations.annotate_clip import annotate_clip
from violations.clip_extract import clip_window, extract_clip
from violations.event import ViolationEvent, ViolationType

DEFAULT_VIDEO_NAME = "DeNnDugXxP0.mp4"


# --------------------------------------------------------------------------- #
# Evidence stand-in -- duck-types lpr.evidence.EvidenceResult for violations.export.
# (Real pipeline hands export the genuine EvidenceResult; here we build the demo equivalent.)
# --------------------------------------------------------------------------- #
@dataclass
class DemoEvidence:
    plate: Optional[str]
    plate_score: float
    n_reads: int
    manual_review: bool
    evidence_crops: list = field(default_factory=list)        # vehicle-crop PNG byte strings
    evidence_frame_ids: list = field(default_factory=list)
    evidence_plate_crops: list = field(default_factory=list)  # plate-crop PNG bytes, parallel to crops
    plate_crop: Any = None


# --------------------------------------------------------------------------- #
# Evidence pictures: a VEHICLE crop + a PLATE crop per shot, cut from the real bbox.
# --------------------------------------------------------------------------- #
def _clamp(v, lo, hi):
    return max(lo, min(hi, int(round(v))))


def _crop_png(frame, box):
    import cv2
    x1, y1, x2, y2 = box
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    ok, buf = cv2.imencode(".png", sub)
    return buf.tobytes() if ok else None


def _vehicle_and_plate(frame, bbox):
    """A padded vehicle crop + a heuristic plate-region crop (lower-centre of the bbox -- where a
    plate sits) from a full frame given the vehicle bbox. Returns (vehicle_png, plate_png)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = (x2 - x1), (y2 - y1)
    veh = _crop_png(frame, (_clamp(x1 - 0.08 * bw, 0, w), _clamp(y1 - 0.08 * bh, 0, h),
                            _clamp(x2 + 0.08 * bw, 0, w), _clamp(y2 + 0.08 * bh, 0, h)))
    plate = _crop_png(frame, (_clamp(x1 + 0.30 * bw, 0, w), _clamp(y2 - 0.32 * bh, 0, h),
                              _clamp(x2 - 0.30 * bw, 0, w), _clamp(y2 - 0.02 * bh, 0, h)))
    return veh, plate


def _pick3(frames: list) -> list:
    if len(frames) <= 3:
        return list(frames)
    return [frames[0], frames[len(frames) // 2], frames[-1]]


def _evidence_frames_for(event: ViolationEvent, track_boxes: dict) -> list:
    """Three frame indices (where the violating track HAS a bbox) for the 3 evidence pictures."""
    fids = sorted(track_boxes.keys())
    if not fids:
        return []
    d = event.details or {}
    s, e = d.get("start_frame"), d.get("end_frame")
    window = [f for f in fids if s <= f <= e] if (s is not None and e is not None and e > s) else []
    if len(window) < 3:                                  # short span -> widen around the key frame
        kf = event.key_frame
        window = [f for f in fids if abs(f - kf) <= 45] or fids
    return _pick3(window)


# --------------------------------------------------------------------------- #
# Analysis stage -- real cached crossings + clearly-labelled synthetic speeding/yellow.
# --------------------------------------------------------------------------- #
def _cached_json_path(prefix: str) -> Path:
    return Path(_REPO_ROOT) / "outputs" / "pipeline" / f"{prefix}_violations.json"


def load_track_boxes(prefix: str) -> dict:
    """Per-frame bounding boxes from the cached analysis -> {track_id: {frame_id: (x1,y1,x2,y2)}}.
    Used to draw the red box on the REAL violating vehicle in the annotated clip."""
    path = Path(_REPO_ROOT) / "outputs" / "violation_cache" / f"{prefix}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes: dict = {}
    for fr in data.get("frames", []):
        fid = int(fr.get("frame"))
        for v in fr.get("vehicles", []):
            tid = v.get("track_id")
            if tid is None or "bbox" not in v:
                continue
            boxes.setdefault(int(tid), {})[fid] = tuple(v["bbox"])
    return boxes


def _pick_track_at(boxes: dict, frame: int):
    """The largest-area tracked vehicle present at ``frame`` -> its track_id (or None). Used to
    anchor a SYNTHETIC event's box on a real on-screen car for the demo."""
    best, best_area = None, -1
    for tid, fmap in boxes.items():
        bb = fmap.get(frame)
        if bb:
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if area > best_area:
                best, best_area = tid, area
    return best


def load_cached_crossings(prefix: str, top_k: int) -> list:
    """Load the top-K highest-confidence REAL solid-line crossings from the cached analysis."""
    path = _cached_json_path(prefix)
    if not path.exists():
        print(f"[analyse] no cached analysis at {path} -- proceeding with synthetic events only",
              flush=True)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    items = sorted(data.get("violations", []), key=lambda v: -float(v.get("confidence", 0)))[:top_k]
    events = []
    for it in items:
        s, e = int(it["start_frame"]), int(it["end_frame"])
        events.append(ViolationEvent(
            vehicle_id=int(it["track_id"]),
            violation_type=ViolationType.SOLID_LINE_CROSSING,
            key_frame=(s + e) // 2,
            confidence=float(it["confidence"]),
            details={"start_frame": s, "end_frame": e, "source": "cached_real_analysis",
                     "orig_violation_id": it.get("violation_id")}))
    return events


def synthetic_events() -> list:
    """One SPEEDING + one YELLOW event so the demo exercises tiers 1 and 3 + the .docx. CLEARLY
    flagged synthetic -- this crossing clip genuinely has neither (no GPS, no yellow shoulder)."""
    speeding = ViolationEvent(
        vehicle_id=901, violation_type=ViolationType.SPEEDING, key_frame=1200, confidence=1.0,
        details={"est_speed_kmh": 96.0, "speed_limit_kmh": 50.0, "over_by_kmh": 46.0,
                 "_synthetic": True,
                 "_note": "demo-injected: this clip has no GPS/speeding; synthetic event to "
                          "exercise the speeding tier + .docx report"})
    yellow = ViolationEvent(
        vehicle_id=951, violation_type=ViolationType.YELLOW_LINE_RIGHT, key_frame=1500,
        confidence=0.55,
        details={"shoulder_overlap_frac": 0.6, "_synthetic": True,
                 "_note": "demo-injected: synthetic yellow-line event to exercise tier 3 "
                          "(chronological, no confidence ranking)"})
    return [speeding, yellow]


def _build_evidence(event: ViolationEvent, cap, track_boxes: dict) -> DemoEvidence:
    """3 vehicle crops + a plate crop each, cut from the violating vehicle's real bbox per frame."""
    import cv2
    veh_crops, plate_crops, ids = [], [], []
    for fid in _evidence_frames_for(event, track_boxes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        bbox = track_boxes.get(fid)
        if bbox is None:                               # no box that frame -> ship the full frame
            okp, buf = cv2.imencode(".png", frame)
            if okp:
                veh_crops.append(buf.tobytes()); plate_crops.append(None); ids.append(fid)
            continue
        veh, plate = _vehicle_and_plate(frame, bbox)
        if veh is None:
            okp, buf = cv2.imencode(".png", frame); veh = buf.tobytes() if okp else None
        if veh is not None:
            veh_crops.append(veh); plate_crops.append(plate); ids.append(fid)

    synth = bool((event.details or {}).get("_synthetic"))
    if event.violation_type == ViolationType.SPEEDING and synth:
        # demo plate row matching the agreed example payload (plate is EVIDENCE, never ranks)
        return DemoEvidence(plate="40-418-79", plate_score=0.971, n_reads=6, manual_review=False,
                            evidence_crops=veh_crops, evidence_frame_ids=ids,
                            evidence_plate_crops=plate_crops)
    # real crossings on this clip read no plate -> UNKNOWN / manual review (recall-first)
    return DemoEvidence(plate=None, plate_score=0.0, n_reads=0, manual_review=True,
                        evidence_crops=veh_crops, evidence_frame_ids=ids,
                        evidence_plate_crops=plate_crops)


# --------------------------------------------------------------------------- #
# The whole loop.
# --------------------------------------------------------------------------- #
def run_once(base_url: str, *, video_name: str = DEFAULT_VIDEO_NAME,
             work_dir: Optional[str] = None, prefix: Optional[str] = None,
             top_k_crossings: int = 3, with_synthetic: bool = True, annotate: bool = True,
             session: Any = None, ingest_path: str = "/ingest",
             ingest_url: Optional[str] = None, job_id: Optional[str] = None) -> dict:
    """Pull -> analyse -> evidence -> bundle -> push. Returns the backend's JSON receipt (plus a
    local 'manifest' echo). Raises violations.video_integrity.IntegrityError if the download is
    corrupt, or requests exceptions on transport failure."""
    own_tmp = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="roadguard_brain_")
    os.makedirs(work_dir, exist_ok=True)
    if session is None:
        import requests
        session = requests.Session()
    prefix = prefix or Path(video_name).stem

    # 1. PULL + fail-fast integrity ------------------------------------------------------------
    local_path = os.path.join(work_dir, video_name)
    local_path, ref_meta, local_meta = ingest_client.download_and_verify(
        base_url, video_name, local_path, session=session)
    fps = local_meta.fps or 30.0
    total_frames = local_meta.frame_count or 0
    print(f"[pull]   downloaded {video_name} -> integrity OK "
          f"({local_meta.width}x{local_meta.height} @ {fps:g}fps, {total_frames} frames)", flush=True)

    # 2. ANALYSE -------------------------------------------------------------------------------
    events = load_cached_crossings(prefix, top_k_crossings)
    if with_synthetic:
        events += synthetic_events()
    print(f"[analyse] {len(events)} violations "
          f"({sum(e.violation_type == ViolationType.SOLID_LINE_CROSSING for e in events)} real "
          f"crossings + synthetic)", flush=True)

    # 3. EVIDENCE: annotated clip (red box + caption) + vehicle/plate crops (+ .docx for speeding) -
    import cv2
    boxes = load_track_boxes(prefix)          # per-frame bboxes -> real box on the real vehicle
    cap = cv2.VideoCapture(local_path)
    records = []
    try:
        for ev in events:
            stem = vx.violation_id(ev)
            win = clip_window(ev.key_frame, fps, total_frames=total_frames)
            dest = os.path.join(work_dir, f"{stem}.mp4")
            synth = bool((ev.details or {}).get("_synthetic"))
            # real crossings -> their own track; synthetic events -> anchored to a real on-screen car
            box_track = (ev.vehicle_id if ev.violation_type == ViolationType.SOLID_LINE_CROSSING
                         else _pick_track_at(boxes, ev.key_frame))
            track_boxes = boxes.get(box_track, {})

            if annotate:
                caption = vx.describe_violation(ev) + (" [SYNTHETIC DEMO EVENT]" if synth else "")
                tag = f"VEH {ev.vehicle_id} - {ev.violation_type}"
                clip = annotate_clip(local_path, dest, win, box_for_frame=track_boxes.get,
                                     caption=caption, fps=fps, tag=tag)
            else:
                clip = extract_clip(local_path, dest, win)

            evidence = _build_evidence(ev, cap, track_boxes)
            report = None
            if ev.violation_type == ViolationType.SPEEDING:
                report = docx_report.speeding_report(ev, video_meta=local_meta.to_dict(),
                                                      window=win.to_dict())
            records.append((ev, evidence, clip, report))
    finally:
        cap.release()

    # 4. BUNDLE: prioritised manifest + evidence (crops are already PNG bytes -> identity encoder)
    bundle = vx.build_export(records, video_meta=local_meta.to_dict(),
                             client_info={"app": "roadguard", "component": "run_brain",
                                          "demo": True},
                             png_encoder=bytes)
    targz = vx.bundle_targz(bundle)
    queue = [(v["violation"], v["vehicle_id"], v["detector_confidence"]) for v in bundle.violations]
    print(f"[bundle]  {len(bundle.violations)} violations, {len(bundle.files)} files, "
          f"{len(targz):,} bytes\n          queue order: {queue}", flush=True)

    # 5. PUSH (to the callback URL when dispatched by the backend, else base_url + ingest_path) -
    target = ingest_url or (base_url.rstrip("/") + ingest_path)
    extra = {"source_video": video_name}
    if job_id:
        extra["job_id"] = job_id                          # so the backend correlates the callback
    resp = vx.post_bundle(target, targz, session=session, extra_fields=extra)
    print(f"[push]    POST {target} -> {resp.status_code} (ok={resp.ok})", flush=True)

    if own_tmp:
        # leave the tmp dir for inspection; it is small and self-cleaning on reboot
        pass

    receipt: dict = {}
    try:
        receipt = json.loads(resp.body) if resp.body else {}
    except json.JSONDecodeError:
        receipt = {"raw_body": resp.body}
    return {"ok": resp.ok, "status_code": resp.status_code, "receipt": receipt,
            "manifest": bundle.manifest, "bundle_bytes": len(targz), "work_dir": work_dir}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Road Guard brain: pull a video, analyse, push evidence.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--video-name", default=DEFAULT_VIDEO_NAME)
    ap.add_argument("--top-k-crossings", type=int, default=3)
    ap.add_argument("--no-synthetic", action="store_true",
                    help="do NOT inject the synthetic speeding/yellow events (crossings only)")
    ap.add_argument("--no-annotate", action="store_true",
                    help="ship the lossless clip instead of the annotated (red box + caption) clip")
    args = ap.parse_args(argv)

    out = run_once(args.base_url, video_name=args.video_name,
                   top_k_crossings=args.top_k_crossings, with_synthetic=not args.no_synthetic,
                   annotate=not args.no_annotate)
    print("\n=== backend receipt ===")
    print(json.dumps(out["receipt"], indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
