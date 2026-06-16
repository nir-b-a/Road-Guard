"""
Module B -- LPR Consumer (cache-driven, model-free core).

Reads the offline per-frame cache produced by the heavy pass and decides, PER TRACK,
the single most trustworthy license-plate string. It never touches the model graph:
OCR, blur measurement and raw-frame access are *injected* as callables, so this module
stays pure, fast and unit-testable. In production those callables wrap the real OCR
reader, cv2's Laplacian, and a cv2.VideoCapture; in tests they are mocked.

Per track:
  1. gather frames where the vehicle bbox area > min_area   (too small => plate unreadable)
  2. crop the bbox from the raw frame, drop motion-blurred crops (Laplacian-variance gate)
  3. OCR each surviving crop -> (plate_string | None, ocr_confidence)
  4. majority vote over plate strings; tie -> largest bbox AREA, then smallest CENTER offset
  5. score the winner = vote_fraction * mean_OCR_confidence

Cache schema consumed (unchanged from the heavy pass):
  cache = {"w", "h", "frames": [{"frame": int,
                                 "vehicles": [{"track_id": int, "bbox": [x1,y1,x2,y2]}],
                                 "shift": [dx, dy]}]}
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional

# Mirrors Constants.LPR.MIN_VEHICLE_AREA -- kept local so this module imports with no repo
# root on sys.path. ~150x100 px: below this the plate has too few pixels to read.
DEFAULT_MIN_AREA = 15_000
# cv2.Laplacian variance below this == too blurry to OCR. Tunable per camera/clip.
DEFAULT_BLUR_MIN_VARIANCE = 100.0

# A frame provider maps a frame index -> the raw BGR frame (or None if unavailable).
FrameProvider = Callable[[int], Optional[Any]]
# An OCR reader maps a crop -> (validated plate string | None, ocr confidence in [0,1]).
OCRReader = Callable[[Any], "tuple[Optional[str], float]"]
# A blur function maps a crop -> a sharpness score (higher = sharper).
BlurFn = Callable[[Any], float]


@dataclass(frozen=True)
class PlateRead:
    """One successful OCR read of a plate, with the geometry that ranks its trust."""
    plate: str
    ocr_conf: float
    area: float          # bbox area in px^2 -- larger == closer == more readable
    center_dist: float   # distance of bbox centre from frame centre -- smaller == better
    frame: int


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox) -> "tuple[float, float]":
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(bbox, w: float, h: float) -> float:
    cx, cy = bbox_center(bbox)
    return math.hypot(cx - w / 2.0, cy - h / 2.0)


def build_track_index(cache: dict) -> "dict[int, list[tuple[int, list]]]":
    """track_id -> [(frame, bbox), ...] sorted by frame. Shared with Module D (joiner)."""
    index: "dict[int, list[tuple[int, list]]]" = defaultdict(list)
    for fr in cache["frames"]:
        for v in fr.get("vehicles", []):
            index[v["track_id"]].append((fr["frame"], v["bbox"]))
    for tid in index:
        index[tid].sort(key=lambda fb: fb[0])
    return dict(index)


# --------------------------------------------------------------------------- #
# per-track read collection + voting
# --------------------------------------------------------------------------- #
def collect_track_reads(
    cache: dict,
    track_id: int,
    frame_provider: FrameProvider,
    ocr_reader: OCRReader,
    blur_fn: BlurFn,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    blur_min_variance: float = DEFAULT_BLUR_MIN_VARIANCE,
    track_index: "Optional[dict[int, list[tuple[int, list]]]]" = None,
) -> "list[PlateRead]":
    """Walk one track's frames, keeping only big-enough, sharp-enough, OCR-readable crops."""
    w, h = cache["w"], cache["h"]
    index = track_index if track_index is not None else build_track_index(cache)
    reads: "list[PlateRead]" = []
    for frame_id, bbox in index.get(track_id, []):
        if bbox_area(bbox) <= min_area:                      # 1. size gate
            continue
        frame = frame_provider(frame_id)
        if frame is None:
            continue
        x1, y1, x2, y2 = (int(round(c)) for c in bbox)
        crop = frame[y1:y2, x1:x2]
        if getattr(crop, "size", 0) == 0:
            continue
        if blur_fn(crop) < blur_min_variance:                # 2. blur gate
            continue
        plate, conf = ocr_reader(crop)                       # 3. OCR
        if not plate:
            continue
        reads.append(PlateRead(plate=plate, ocr_conf=float(conf),
                               area=bbox_area(bbox),
                               center_dist=center_distance(bbox, w, h),
                               frame=frame_id))
    return reads


def vote_plate(reads: "list[PlateRead]") -> "tuple[Optional[str], float]":
    """Majority vote with the Directive-4 tie-breaker. Returns (plate | None, score).

    Ranking key per candidate plate = (votes, best-read area, -best-read center_dist):
    most votes wins; ties broken by the largest bbox seen for that plate, then the read
    closest to frame centre. Score of the winner = vote_fraction * mean_OCR_confidence.
    """
    if not reads:
        return None, 0.0
    groups: "dict[str, list[PlateRead]]" = defaultdict(list)
    for r in reads:
        groups[r.plate].append(r)

    def rank_key(item: "tuple[str, list[PlateRead]]"):
        _, rs = item
        best = max(rs, key=lambda r: (r.area, -r.center_dist))
        return (len(rs), best.area, -best.center_dist)

    winner_plate, winner_reads = max(groups.items(), key=rank_key)
    vote_fraction = len(winner_reads) / len(reads)
    mean_conf = sum(r.ocr_conf for r in winner_reads) / len(winner_reads)
    return winner_plate, vote_fraction * mean_conf


def run_lpr(
    cache: dict,
    frame_provider: FrameProvider,
    ocr_reader: OCRReader,
    blur_fn: BlurFn,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    blur_min_variance: float = DEFAULT_BLUR_MIN_VARIANCE,
) -> "dict[int, dict]":
    """Run the LPR vote for every track in the cache.

    Returns: track_id -> {"plate_candidate": str|None,
                          "plate_confidence_score": float,   # vote_fraction * mean_OCR_conf
                          "n_reads": int}
    """
    index = build_track_index(cache)
    out: "dict[int, dict]" = {}
    for tid in index:
        reads = collect_track_reads(cache, tid, frame_provider, ocr_reader, blur_fn,
                                    min_area=min_area, blur_min_variance=blur_min_variance,
                                    track_index=index)
        plate, score = vote_plate(reads)
        out[tid] = {"plate_candidate": plate,
                    "plate_confidence_score": round(score, 6),
                    "n_reads": len(reads)}
    return out


def run_lpr_for_tracks(
    cache: dict,
    track_ids,
    frame_provider: FrameProvider,
    ocr_reader: OCRReader,
    blur_fn: BlurFn,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    blur_min_variance: float = DEFAULT_BLUR_MIN_VARIANCE,
    max_frames_per_track: int = 8,
    track_index: "Optional[dict[int, list[tuple[int, list]]]]" = None,
) -> "dict[int, dict]":
    """Violation-driven LPR: read plates ONLY for the given tracks (e.g. the violators), and only
    on each track's LARGEST-area frames (best plate pixels), capped at max_frames_per_track. Far
    cheaper than run_lpr over every track x every frame -- it targets exactly the cars we must bill.

    Returns the same shape as run_lpr: track_id -> {plate_candidate, plate_confidence_score, n_reads}.
    """
    index = track_index if track_index is not None else build_track_index(cache)
    w, h = cache["w"], cache["h"]
    out: "dict[int, dict]" = {}
    for tid in track_ids:
        ordered = sorted((fb for fb in index.get(tid, []) if bbox_area(fb[1]) > min_area),
                         key=lambda fb: bbox_area(fb[1]), reverse=True)[:max_frames_per_track]
        reads: "list[PlateRead]" = []
        for frame_id, bbox in ordered:
            frame = frame_provider(frame_id)
            if frame is None:
                continue
            x1, y1, x2, y2 = (int(round(c)) for c in bbox)
            crop = frame[y1:y2, x1:x2]
            if getattr(crop, "size", 0) == 0:
                continue
            if blur_fn(crop) < blur_min_variance:
                continue
            plate, conf = ocr_reader(crop)
            if not plate:
                continue
            reads.append(PlateRead(plate=plate, ocr_conf=float(conf), area=bbox_area(bbox),
                                   center_dist=center_distance(bbox, w, h), frame=frame_id))
        plate, score = vote_plate(reads)
        out[tid] = {"plate_candidate": plate,
                    "plate_confidence_score": round(score, 6), "n_reads": len(reads)}
    return out


# --------------------------------------------------------------------------- #
# production-side adapters (kept here so the core stays import-light; mocked in tests)
# --------------------------------------------------------------------------- #
def laplacian_variance(crop) -> float:
    """Sharpness score = variance of the Laplacian. Lazy cv2 import keeps tests model-free."""
    import cv2  # local import: never loaded during pure-logic unit tests
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def make_video_frame_provider(video_path: str) -> FrameProvider:
    """Random-access frame provider over a video file (production use)."""
    import cv2
    cap = cv2.VideoCapture(video_path)

    def provider(frame_id: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        return frame if ok else None

    return provider
