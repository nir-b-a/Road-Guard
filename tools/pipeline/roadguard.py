"""
RoadGuard -- end-to-end clip processor.

For one clip: run the heavy pass (BoT-SORT vehicles + lanes), detect solid-white-line violations,
then for each VIOLATING car read its plate using SHARPEST-FRAME selection (area x Laplacian
sharpness, not just biggest) + PER-CHARACTER temporal voting, and render a dynamically-scaled
annotated video (works for 9:16 Shorts and 16:9). Reuses the tested pipeline components; heavy
models load only at run time, so this module imports fine on CPU (cv2 lazy in render).

Run (roadguard-dl env):
  python tools/pipeline/roadguard.py 7E35VSQbAH8
  (auto-finds the video under tests_videos/raw_videos/solid_line_crossing_with_pl/<prefix>.mp4)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_PIPELINE_DIR)
_REPO = os.path.dirname(_TOOLS_DIR)
for _p in (_PIPELINE_DIR, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import heavy_pass                         # noqa: E402
import violation_consumer                 # noqa: E402
import lpr_consumer                        # noqa: E402
import plate_char_voter as pcv             # noqa: E402
import renderer                            # noqa: E402
import evidence                            # noqa: E402
from run_pipeline import build_ocr_object, DEFAULT_OUT   # noqa: E402

VIOLATION_TYPE = "SOLID WHITE LINE CROSSING"
NEW_CLIPS_DIR = os.path.join(_REPO, "tests_videos", "raw_videos", "solid_line_crossing_with_pl")


def _find_video(prefix: str) -> str | None:
    for d in (NEW_CLIPS_DIR, os.path.join(_REPO, "tests_videos", "raw_videos", "crossing_solid_line")):
        hits = [f for f in glob.glob(os.path.join(d, prefix + "*.mp4")) if "_annotated" not in f]
        if hits:
            return hits[0]
    return None


class RoadGuard:
    def __init__(self, reader_kind: str = "fast_alpr",
                 k_sec: float = violation_consumer.DEFAULT_K_SEC,
                 max_frames: int = 8, min_area: float = lpr_consumer.DEFAULT_MIN_AREA,
                 out_dir: str = DEFAULT_OUT, hold_sec: float = 0.6):
        self._reader_obj = build_ocr_object(reader_kind)   # LPRReader object (for plate localisation), or None
        self.reader = ((lambda crop: self._reader_obj.read_plate_with_conf(crop))
                       if self._reader_obj is not None else None)   # crop -> (plate, conf), or None
        self.k_sec = k_sec
        self.max_frames = max_frames
        self.min_area = min_area
        self.out_dir = out_dir
        self.hold_sec = hold_sec        # keep each violation box on screen this long so a human can see it
        self.model = violation_consumer.load_confidence_model()

    # --- plate reading: sharpest frames + per-character voting --------------- #
    def read_track_plate(self, cache, tid, frame_provider, index, n_evidence: int = 3):
        """Vote a plate from the track's sharpest crops. Returns (plate|None, score, evidence_crops)
        where evidence_crops are the top-`n_evidence` sharpest vehicle crops (for zoom cards) --
        kept even when the read fails so UNKNOWN cars still get a human-review card."""
        candidates = [(f, b) for f, b in index.get(tid, [])
                      if lpr_consumer.bbox_area(b) > self.min_area]
        scored = []
        for frame_id, bbox in candidates:
            frame = frame_provider(frame_id)
            if frame is None:
                continue
            x1, y1, x2, y2 = (int(round(c)) for c in bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if getattr(crop, "size", 0) == 0:
                continue
            sharp = lpr_consumer.laplacian_variance(crop)         # readability = size x sharpness
            scored.append((lpr_consumer.bbox_area(bbox) * sharp, crop))
        scored.sort(key=lambda t: t[0], reverse=True)
        reads = []
        for _, crop in scored[:self.max_frames]:
            plate, conf = self.reader(crop)
            if plate:
                reads.append((plate, conf))
        plate, score = pcv.vote_characters(reads)
        evidence_crops = [crop for _, crop in scored[:n_evidence]]
        return plate, score, evidence_crops

    # --- annotated full-video render (dynamic scaling) ----------------------- #
    def _render(self, cache, violations, plate_map, prefix, evidence_crops=None,
                diagnostic=False) -> str | None:
        import cv2
        evidence_crops = evidence_crops or {}
        # per-frame active violations: frame -> {track_id: (confidence, plate)}. We HOLD each box
        # for hold_sec past the event so a 1-frame K-of-M hit isn't an invisible ~33ms flash (the
        # box follows the still-tracked car via bbox_by_frame). Essential for human review.
        hold_frames = max(0, round(self.hold_sec * float(cache.get("fps", 30.0))))
        active: dict = defaultdict(dict)
        for v in violations:
            tid = v["track_id"]
            plate = plate_map.get(tid, {}).get("plate_candidate")
            for f in range(v["start_frame"], v["end_frame"] + 1 + hold_frames):
                prev = active[f].get(tid)
                if prev is None or v["confidence"] > prev[0]:
                    active[f][tid] = (v["confidence"], plate)
        bbox_by_frame = {fr["frame"]: {vv["track_id"]: vv["bbox"] for vv in fr.get("vehicles", [])}
                         for fr in cache["frames"]}

        os.makedirs(self.out_dir, exist_ok=True)
        suffix = "diagnostic_tracking" if diagnostic else "annotated"
        out_path = os.path.join(self.out_dir, f"{prefix}_{suffix}.mp4")
        cap = cv2.VideoCapture(cache["path"])
        if not cap.isOpened():
            print(f"[render] cannot open {cache['path']}")
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or float(cache.get("fps", 30.0))
        writer, fi, dims = None, 0, None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                dims = (w, h)
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            # diagnostic: label EVERY tracked vehicle with its raw track id, every frame (drawn
            # first so the red violation overlay sits on top where they coincide)
            if diagnostic:
                for tid, bbox in bbox_by_frame.get(fi, {}).items():
                    renderer.draw_track_box(frame, bbox=bbox, track_id=tid)
            for tid, (conf, plate) in active.get(fi, {}).items():
                bbox = bbox_by_frame.get(fi, {}).get(tid)
                if bbox:
                    renderer.draw_violation(frame, bbox=bbox, track_id=tid, plate=plate,
                                            violation_type=VIOLATION_TYPE, confidence=conf)
            writer.write(frame)
            fi += 1

        # tail: one held zoomed-plate evidence page per violating car (for human verification)
        if writer is not None and dims is not None and evidence_crops:
            self._append_evidence(writer, dims, fps, plate_map, evidence_crops, prefix)

        cap.release()
        if writer is not None:
            writer.release()
        print(f"   annotated -> {out_path}")
        return out_path

    def _append_evidence(self, writer, dims, fps, plate_map, evidence_crops, prefix) -> None:
        """Append held evidence cards to the open writer and save each as a standalone PNG."""
        import cv2
        w, h = dims
        hold = max(1, round(fps * evidence.HOLD_SEC))
        for tid in sorted(evidence_crops):
            crops = evidence_crops.get(tid) or []
            zoom_imgs, has_plate = [], False
            for crop in crops[:3]:
                z, is_plate = evidence.zoom_crop(crop, self._reader_obj)
                zoom_imgs.append(z)
                has_plate = has_plate or is_plate
            pm = plate_map.get(tid, {})
            card = evidence.compose_card(w, h, track_id=tid,
                                         plate=pm.get("plate_candidate"),
                                         score=pm.get("plate_confidence_score", 0.0),
                                         violation_type=VIOLATION_TYPE,
                                         zoom_imgs=zoom_imgs, has_plate=has_plate)
            for _ in range(hold):
                writer.write(card)
            png = os.path.join(self.out_dir, f"{prefix}_car{tid}_evidence.png")
            cv2.imwrite(png, card)
            print(f"   evidence -> {png}")

    def process(self, prefix, video_path=None, refresh=True, diagnostic=False) -> dict:
        video_path = video_path or _find_video(prefix)
        cache = heavy_pass.run_heavy_pass(prefix, video_path=video_path, refresh=refresh)
        violations = violation_consumer.find_violations(cache, self.model, k_sec=self.k_sec, prefix=prefix)
        index = lpr_consumer.build_track_index(cache)

        plate_map, evidence_crops = {}, {}
        if self.reader is not None and violations:
            fp = lpr_consumer.make_video_frame_provider(cache["path"])
            for tid in sorted({v["track_id"] for v in violations}):
                plate, score, crops = self.read_track_plate(cache, tid, fp, index)
                plate_map[tid] = {"plate_candidate": plate, "plate_confidence_score": round(score, 4)}
                evidence_crops[tid] = crops

        print(f"\n=== {prefix}: {len(violations)} violation(s), "
              f"{len(plate_map)} car(s) ===")
        for tid in sorted(plate_map):
            pm = plate_map[tid]
            print(f"  car#{tid:<4} -> {pm['plate_candidate'] or 'UNKNOWN':<12} (score {pm['plate_confidence_score']:.2f})")
        self._render(cache, violations, plate_map, prefix, evidence_crops, diagnostic=diagnostic)
        return {"prefix": prefix, "n_violations": len(violations), "plate_map": plate_map}


def main() -> None:
    ap = argparse.ArgumentParser(description="RoadGuard end-to-end clip processor.")
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--video", default=None, help="explicit path for a single prefix")
    ap.add_argument("--reader", default="fast_alpr", choices=["fast_alpr", "paddle", "none"])
    ap.add_argument("--no-refresh", action="store_true", help="reuse existing cache if present")
    ap.add_argument("--diagnostic", action="store_true",
                    help="label EVERY tracked vehicle with its raw track id on every frame "
                         "(outputs <prefix>_diagnostic_tracking.mp4) to trace tracker ID continuity")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--hold-sec", type=float, default=0.6,
                    help="hold each violation box on screen this long so a human reviewer can see "
                         "it (avoids invisible 1-frame flashes); default 0.6s")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    rg = RoadGuard(reader_kind=args.reader, out_dir=args.out, hold_sec=args.hold_sec)
    summary = []
    for prefix in args.prefixes:
        video = args.video if (args.video and len(args.prefixes) == 1) else None
        summary.append(rg.process(prefix, video_path=video, refresh=not args.no_refresh,
                                   diagnostic=args.diagnostic))
    with open(os.path.join(args.out, "roadguard_report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\n[done] {len(summary)} clip(s) -> {args.out}")


if __name__ == "__main__":
    main()
