"""
In-memory evidence collection for traffic violations (license-plate read + best pictures).

Recipe (lifted from the offline RoadGuard harness) for each VIOLATING vehicle:
  1. rank the vehicle's frames by readability = bbox area x Laplacian sharpness,
  2. OCR the sharpest crops, then temporally VOTE one plate (length-first, then
     per-position) -> (plate, score),
  3. keep the top-N sharpest crops as evidence images a human can verify.

Key difference from the offline version: we never re-open the video to seek frames after
the fact (double-decoding hurts on a 1060). While a vehicle is tracked LIVE we keep a
bounded, per-vehicle rolling buffer of its sharpest crops (online top-K), so by the time
a ViolationEvent fires the best frames are already in memory -- zero extra I/O.

Decoupled from Vehicle/World and from any specific OCR model: the OCR reader and the
sharpness function are injected, so the core logic unit-tests on CPU with mocks.
``observe_vehicle`` is the one hook to drop into the live frame loop (e.g. main.processFrame).
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Optional, TYPE_CHECKING

from lpr.plate_char_voter import vote_characters

if TYPE_CHECKING:
    from violations.event import ViolationEvent

# Mirrors Constants.LPR.MIN_VEHICLE_AREA -- kept local so this module makes no assumption
# about repo root. ~150x100 px: below this a plate has too few pixels to read.
DEFAULT_MIN_AREA = 15_000
# Keep this many sharpest crops per tracked vehicle. >= max_ocr_frames so the vote has
# enough material; bounded, so memory stays flat no matter how long the clip runs.
DEFAULT_BUFFER_SIZE = 12
# OCR at most this many of the sharpest crops when a violation fires.
DEFAULT_MAX_OCR_FRAMES = 8
# How many crops to hand back as human-review evidence images.
DEFAULT_N_EVIDENCE = 3

# crop -> (validated plate | None, OCR confidence in [0,1])
OCRReader = Callable[[Any], "tuple[Optional[str], float]"]
# crop -> sharpness score (higher = sharper)
SharpnessFn = Callable[[Any], float]


def bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def laplacian_variance(crop) -> float:
    """Variance of the Laplacian = focus/sharpness (higher = sharper). Lazy-imports cv2 so
    this module stays importable on a machine without OpenCV."""
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if getattr(crop, "ndim", 2) == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class EvidenceResult:
    """Outcome of the evidence stage for one violation."""
    vehicle_id: int
    violation_type: str
    plate: Optional[str]             # voted Israeli plate, or None (UNKNOWN -> manual review)
    plate_score: float               # vote agreement x mean OCR confidence, in [0,1]
    n_reads: int                     # how many crops yielded a plate string for the vote
    n_buffered: int                  # crops available in the buffer at collect time
    evidence_crops: list             # top-N sharpest vehicle crops (BGR nd-arrays)
    evidence_frame_ids: list         # frame ids those crops came from

    @property
    def manual_review(self) -> bool:
        """True when no plate could be read -> a human must confirm from the crops.
        HITL rule: an unreadable plate NEVER drops the violation, it just routes to review."""
        return not self.plate

    @property
    def plate_label(self) -> str:
        """What to write next to the evidence: the voted plate, or the manual-review flag."""
        return self.plate if self.plate else "UNKNOWN - manual review"


class EvidenceCollector:
    """Per-vehicle rolling top-K sharpest-crop buffer + the violation-time evidence stage.

    Live loop:   call ``observe_vehicle(frame_id, frame, vehicle_id, bbox)`` for every
                 tracked vehicle, every frame. Cheap and memory-bounded.
    On an event: call ``collect_evidence(event)`` -> EvidenceResult (plate + best pictures),
                 using only buffered crops (no video re-read).
    """

    def __init__(self, ocr_reader: OCRReader, *,
                 min_area: float = DEFAULT_MIN_AREA,
                 buffer_size: int = DEFAULT_BUFFER_SIZE,
                 max_ocr_frames: int = DEFAULT_MAX_OCR_FRAMES,
                 n_evidence: int = DEFAULT_N_EVIDENCE,
                 sharpness_fn: SharpnessFn = laplacian_variance):
        self._ocr = ocr_reader
        self.min_area = min_area
        self.buffer_size = buffer_size
        self.max_ocr_frames = max_ocr_frames
        self.n_evidence = n_evidence
        self._sharpness = sharpness_fn
        # vehicle_id -> min-heap of (score, seq, frame_id, crop); size capped at buffer_size,
        # so heap[0] is always the WEAKEST kept crop -> O(1) eviction test.
        self._buffers: dict[int, list] = {}
        self._seen: dict[int, int] = {}        # vehicle_id -> # of above-area-gate observations
        self._counter = itertools.count()      # global monotone tie-breaker (never compares crops)
        # vehicle_id -> (plate, score, n_reads): READ-ONCE memo. A car that speeds AND crosses
        # a line is OCR'd once; later violations reuse the established plate (no re-OCR).
        self._plates: dict[int, tuple] = {}

    # --- live ingest --------------------------------------------------------- #
    def observe_vehicle(self, frame_id: int, frame, vehicle_id: int, bbox) -> None:
        """Offer one tracked vehicle's bbox crop to its rolling buffer. Area-gates first
        (free); only computes sharpness + copies pixels for a crop worth keeping."""
        area = bbox_area(bbox)
        if area < self.min_area:
            return
        x1, y1, x2, y2 = (int(round(c)) for c in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2]
        if getattr(crop, "size", 0) == 0:
            return
        self._seen[vehicle_id] = self._seen.get(vehicle_id, 0) + 1
        score = area * self._sharpness(crop)
        heap = self._buffers.setdefault(vehicle_id, [])
        if len(heap) >= self.buffer_size and score <= heap[0][0]:
            return                              # not better than the weakest kept -> skip the copy
        entry = (score, next(self._counter), frame_id, crop.copy())
        if len(heap) < self.buffer_size:
            heapq.heappush(heap, entry)
        else:
            heapq.heapreplace(heap, entry)

    def drop_vehicle(self, vehicle_id: int) -> None:
        """Free a vehicle's buffer once its track ends (memory hygiene for long videos)."""
        self._buffers.pop(vehicle_id, None)
        self._seen.pop(vehicle_id, None)

    # --- introspection (used by selectors / tests) --------------------------- #
    def observations(self, vehicle_id: int) -> int:
        return self._seen.get(vehicle_id, 0)

    def buffered_vehicle_ids(self) -> list:
        return [vid for vid, h in self._buffers.items() if h]

    def well_tracked_vehicles(self, min_observations: int = 5) -> list:
        """Vehicles with at least `min_observations` close (above-area-gate) frames AND a
        non-empty crop buffer -- i.e. good candidates for a meaningful plate read."""
        return [vid for vid, n in self._seen.items()
                if n >= min_observations and self._buffers.get(vid)]

    def sharpest_frame_id(self, vehicle_id: int) -> Optional[int]:
        heap = self._buffers.get(vehicle_id)
        if not heap:
            return None
        return max(heap, key=lambda e: e[0])[2]

    # --- violation-time evidence -------------------------------------------- #
    def collect_evidence(self, event: "ViolationEvent",
                         known_plate: Optional[str] = None) -> EvidenceResult:
        """Read the plate + gather best-picture crops for event.vehicle_id, using ONLY the
        in-memory buffer (no video re-read).

        READ-ONCE: if the vehicle's plate is already known -- either passed in via
        ``known_plate`` (e.g. ``vehicle.license_plate``) or memorized from an earlier
        violation by this same collector -- we reuse it and skip OCR entirely. So a car that
        speeds AND crosses a line is OCR'd once; every violation it commits resolves to the
        same plate.

        RECALL-FIRST: evidence crops are gathered and returned WHATEVER the OCR outcome. If no
        plate can be read, ``result.manual_review`` is True and ``result.plate_label`` is
        "UNKNOWN - manual review" -- the violation is never dropped; a human reads the crops.
        """
        heap = self._buffers.get(event.vehicle_id, [])
        ranked = sorted(heap, key=lambda e: e[0], reverse=True)   # sharpest x largest first
        crops = [e[3] for e in ranked]
        frame_ids = [e[2] for e in ranked]

        cached = self._plates.get(event.vehicle_id)
        if known_plate:                                  # caller already established it
            plate, score, n_reads = known_plate, (cached[1] if cached else 1.0), 0
        elif cached and cached[0]:                       # this collector read it on a prior event
            plate, score, n_reads = cached
        else:                                            # first readable look at this vehicle -> OCR
            reads = []
            for crop in crops[:self.max_ocr_frames]:
                p, conf = self._ocr(crop)
                if p:
                    reads.append((p, conf))
            plate, score = vote_characters(reads)
            n_reads = len(reads)
        # Memorize even a failed read so a later, better-framed violation can retry (recall),
        # while a successful read short-circuits all future events for this vehicle.
        self._plates[event.vehicle_id] = (plate, score, n_reads)

        return EvidenceResult(
            vehicle_id=event.vehicle_id,
            violation_type=event.violation_type,
            plate=plate,
            plate_score=score,
            n_reads=n_reads,
            n_buffered=len(heap),
            evidence_crops=crops[:self.n_evidence],
            evidence_frame_ids=frame_ids[:self.n_evidence],
        )

    def save_evidence(self, result: EvidenceResult, out_dir: str, *, plate_reader=None) -> str:
        """RECALL-FIRST disk write: always persist the top-N sharpest crops + a sidecar label,
        readable plate or not. ``plate_reader`` (optional) adds a tight plate zoom when its
        ``crop_plate`` localises one. Returns ``out_dir``."""
        import os
        import cv2
        os.makedirs(out_dir, exist_ok=True)
        stem = f"v{result.vehicle_id}_{result.violation_type}"
        for i, (crop, fid) in enumerate(zip(result.evidence_crops, result.evidence_frame_ids)):
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{i}_f{fid}.png"), crop)
            if plate_reader is not None:
                pc = plate_reader.crop_plate(crop)
                if pc is not None and getattr(pc, "size", 0):
                    cv2.imwrite(os.path.join(out_dir, f"{stem}_{i}_f{fid}_plate.png"), pc)
        with open(os.path.join(out_dir, f"{stem}.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"vehicle_id={result.vehicle_id}\n"
                     f"violation={result.violation_type}\n"
                     f"plate={result.plate_label}\n"
                     f"plate_score={result.plate_score:.3f}\n"
                     f"n_reads={result.n_reads}\n"
                     f"manual_review={result.manual_review}\n")
        return out_dir
