#!/usr/bin/env python3
"""
bitrate_plate_sweep.py -- find the bitrate/codec knee where compression starts costing
license-plate reads.

THE EXPERIMENT
    Take one representative drive video, re-encode it at a ladder of codec/bitrate rungs,
    and measure how much plate accuracy each rung costs against plates YOU read by eye.

    The comparison is PAIRED: YOLO tracks the REFERENCE video once, and every rung reuses
    those exact track IDs, boxes, AND frame choices. So the only variable between rungs is
    image quality -- not which cars were found, not which frames got OCR'd, not how the
    tracker numbered things. Re-tracking per rung would renumber every ID and make the
    per-vehicle comparison meaningless.

    Freezing the frame choice matters more than it looks. The live pipeline picks frames to
    OCR by `area * laplacian_variance` (lpr/evidence.py). Compression LOWERS Laplacian
    variance, so a low-bitrate rung would rank different frames as "sharpest" and you would
    be measuring frame selection tangled up with image quality. The ranking is computed once,
    on the reference, and reused.

TWO TRACKS, because plate accuracy alone hides half the damage:
    A) PLATE ACCURACY (needs your labels) -- fixed reference boxes -> crop the same pixels
       from each rung -> FastALPRReader -> vote_characters -> compare to truth. This is the
       real pipeline's own OCR path, not an approximation.
    B) DETECTION RECALL (needs NO labels) -- re-run YOLO detection on sampled frames of each
       rung and match boxes to the reference by IoU. A car the detector loses at 3 Mbps never
       reaches OCR at all, which track A cannot see. The reference IS the ground truth here,
       so this comes free.

USAGE (run from the malshinon/ directory)

    # 1. track the reference once + write the labelling template & crop sheets
    python tools/bitrate_plate_sweep.py label --video real_vids/drive.mp4 --out bitrate_sweep

    # 2. open bitrate_sweep/labels.csv, fill the `plate` column for the vehicles whose
    #    plate you can READ BY EYE in bitrate_sweep/label_crops/track_XXXX.jpg.
    #    Leave the rest blank (or `?`) -- they are excluded. See "LABELLING" below.

    # 3. run the sweep
    python tools/bitrate_plate_sweep.py run --video real_vids/drive.mp4 --out bitrate_sweep

OUTPUTS (in --out)
    tracks.json        reference tracking result (boxes + frozen sharpness ranking)
    label_crops/       one contact sheet per track, for eyeball labelling
    labels.csv         the template you fill in; re-read on every `run`
    encodes/           the re-encoded rungs (deleted unless --keep-encodes)
    results.csv        one row per rung: size, real bitrate, every accuracy metric
    per_vehicle.csv    truth vs. prediction per (rung, vehicle) -- shows WHICH cars fail first
    sweep.png          accuracy-vs-bitrate curves + size-vs-bitrate (needs matplotlib)

LABELLING
    Only label a plate a human can read at REFERENCE quality. A plate you cannot read at
    12 Mbps is not a valid test item -- it fails at every rung equally and just adds noise
    that flattens the curve. 15-30 labelled vehicles is a usable sample; below ~10 the
    exact-match number swings too much to locate a knee (character accuracy still works).

READING THE RESULT -- two biases that partly cancel, and one that does not
    1. OPTIMISTIC: libx264/libx265 are meaningfully better than a phone's HARDWARE encoder at
       equal bitrate (hardware encoders give up roughly 20-30% of rate-distortion for speed
       and power). Whatever knee this script finds, the on-device knee sits HIGHER. Add margin.
    2. PESSIMISTIC: the reference is itself already a lossy ~12 Mbps AVC capture, so every
       rung is a SECOND generation. A phone shooting natively at that bitrate would look a
       little better than the same rung here.
    (2) softens (1) but does not cancel it, and neither is measurable from inside this script.
    Treat the output as the SHAPE of the curve and as a codec A/B -- then pick an operating
    point above the knee, not on it.
"""

import argparse
import csv
import heapq
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # malshinon/
sys.path.insert(0, ROOT)

import Constants  # noqa: E402
from lpr.evidence import (  # noqa: E402
    DEFAULT_BUFFER_SIZE, DEFAULT_MAX_OCR_FRAMES, bbox_area, laplacian_variance,
)
from lpr.plate_char_voter import digits_only, vote_characters  # noqa: E402

FFMPEG = os.environ.get("FFMPEG_BIN", "ffmpeg")

# Default ladder: AVC and HEVC at matched bitrates, so the codec A/B reads straight off the
# table. The phone currently shoots AVC at 12 Mbps (RecordingActivity.VIDEO_BITRATE), which is
# the top AVC rung -- and the `source` rung is the untouched capture, the ceiling for everything.
DEFAULT_LADDER = [
    ("h264", 12000), ("h264", 8000), ("h264", 6000),
    ("h264", 4000), ("h264", 3000), ("h264", 2000),
    ("hevc", 8000), ("hevc", 6000), ("hevc", 4000),
    ("hevc", 3000), ("hevc", 2000), ("hevc", 1500),
]

# Match the phone: 1 s GOP at 30 fps. Also keeps random-access seeks cheap for clip cutting.
DEFAULT_GOP = 30
# A track needs at least this many above-area-gate frames before it is worth labelling.
DEFAULT_MIN_OBS = 5
# Detection-recall sampling: every Nth frame. 15 = twice a second at 30 fps.
DEFAULT_DETECT_STRIDE = 15
IOU_MATCH = 0.5


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def levenshtein(a: str, b: str) -> int:
    """Plain DP edit distance. Local so the script adds no dependency."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_accuracy(truth: str, pred: str) -> float:
    """1 - normalised edit distance, on digits only. Unlike exact-match this degrades
    SMOOTHLY, so the knee is visible with far fewer labelled vehicles."""
    if not truth:
        return 0.0
    if not pred:
        return 0.0
    return max(0.0, 1.0 - levenshtein(truth, pred) / max(len(truth), len(pred)))


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def safe_crop(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(c)) for c in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def probe_bitrate_kbps(path: str) -> float:
    """Real delivered bitrate = filesize / duration. Measured, not requested -- VBR rarely
    lands exactly on target, and the size is what actually costs money in R2."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    dur = (n / fps) if fps > 0 and n > 0 else 0.0
    if dur <= 0:
        return 0.0
    return os.path.getsize(path) * 8 / dur / 1000.0


# --------------------------------------------------------------------------- #
# stage 1 -- reference tracking pass
# --------------------------------------------------------------------------- #
def build_reference_tracks(video: str, *, imgsz: int, conf: float, min_area: float,
                           buffer_size: int, detect_stride: int, limit_frames: int | None):
    """Track the REFERENCE video once and record, per track, the top-`buffer_size` frames
    ranked by `area * laplacian_variance` -- the exact ranking EvidenceCollector uses live.

    Only scores and boxes are kept (never crops), so memory stays flat over a long drive.
    Also snapshots every vehicle box on each `detect_stride`-th frame: that is the ground
    truth for the label-free detection-recall track.
    """
    from ultralytics import YOLO

    model = YOLO(Constants.YOLO_VERSION)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open reference video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    heaps: dict[int, list] = {}          # track_id -> min-heap of (score, frame_id, bbox)
    seen: dict[int, int] = defaultdict(int)
    detect_ref: dict[int, list] = {}
    seq = 0
    frame_id = 0

    print(f"[ref] tracking {video} with {Constants.YOLO_VERSION} imgsz={imgsz} conf={conf}")
    while True:
        ok, frame = cap.read()
        if not ok or (limit_frames is not None and frame_id >= limit_frames):
            break
        results = model.track(frame, persist=True, tracker=Constants.YOLO_TRACKER,
                              verbose=False, classes=Constants.DETECTION_CLASSES,
                              conf=conf, imgsz=imgsz)
        boxes = results[0].boxes
        sampled = (frame_id % detect_stride == 0)
        if sampled:
            detect_ref[frame_id] = []
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls.item())
                if not Constants.is_vehicle(cls):
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)
                if sampled:
                    detect_ref[frame_id].append(bbox)
                if box.id is None:
                    continue
                area = bbox_area(bbox)
                if area < min_area:            # area-gate first, same as the live collector
                    continue
                tid = int(box.id.item())
                seen[tid] += 1
                heap = heaps.setdefault(tid, [])
                crop = safe_crop(frame, bbox)
                if crop is None:
                    continue
                score = area * laplacian_variance(crop)
                entry = (score, seq, frame_id, bbox)
                seq += 1
                if len(heap) < buffer_size:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        frame_id += 1
        if frame_id % 300 == 0:
            print(f"[ref]   frame {frame_id}, {len(heaps)} tracks")
    cap.release()

    tracks = {}
    for tid, heap in heaps.items():
        ranked = sorted(heap, key=lambda e: e[0], reverse=True)
        tracks[str(tid)] = {
            "n_obs": seen[tid],
            "best": [[float(s), int(f), [int(c) for c in bb]] for s, _, f, bb in ranked],
        }
    return {
        "video": os.path.abspath(video), "fps": fps, "width": width, "height": height,
        "n_frames": frame_id, "imgsz": imgsz, "conf": conf, "min_area": min_area,
        "model": Constants.YOLO_VERSION, "tracker": Constants.YOLO_TRACKER,
        "detect_stride": detect_stride,
        "detect_ref": {str(k): v for k, v in detect_ref.items()},
        "tracks": tracks,
    }


# --------------------------------------------------------------------------- #
# stage 2 -- labelling aids
# --------------------------------------------------------------------------- #
def write_label_assets(video: str, ref: dict, out_dir: str, *, min_obs: int, n_sheet: int):
    """Write one contact sheet per candidate track + the labels.csv template.

    The sheet shows the SHARPEST crops of that track, upscaled, so you can read the plate by
    eye and type it in. Reading the video and hunting for track IDs by hand is the part of
    this experiment that would otherwise take an afternoon.
    """
    crops_dir = os.path.join(out_dir, "label_crops")
    os.makedirs(crops_dir, exist_ok=True)

    candidates = {tid: t for tid, t in ref["tracks"].items()
                  if t["n_obs"] >= min_obs and t["best"]}
    # frame_id -> [(track_id, bbox, rank)] so one sequential pass grabs every crop we need
    wanted = defaultdict(list)
    for tid, t in candidates.items():
        for rank, (_score, fid, bbox) in enumerate(t["best"][:n_sheet]):
            wanted[fid].append((tid, bbox, rank))

    collected: dict[str, list] = defaultdict(list)
    cap = cv2.VideoCapture(video)
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, bbox, rank in wanted.get(fid, []):
            crop = safe_crop(frame, bbox)
            if crop is not None:
                collected[tid].append((rank, fid, crop))
        fid += 1
    cap.release()

    rows = []
    for tid, t in sorted(candidates.items(), key=lambda kv: -kv[1]["n_obs"]):
        items = sorted(collected.get(tid, []))
        if not items:
            continue
        panels = []
        for _rank, f_id, crop in items:
            h, w = crop.shape[:2]
            scale = 260.0 / max(1, h)
            panel = cv2.resize(crop, (max(1, int(w * scale)), 260),
                               interpolation=cv2.INTER_CUBIC)
            cv2.putText(panel, f"f{f_id}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
        sheet = cv2.hconcat(panels)
        banner = np.zeros((34, sheet.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, f"track {tid}  ({t['n_obs']} close frames)", (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(crops_dir, f"track_{int(tid):04d}.jpg"),
                    cv2.vconcat([banner, sheet]))
        rows.append({"track_id": tid, "n_obs": t["n_obs"],
                     "best_frame": t["best"][0][1],
                     "best_area": int(bbox_area(t["best"][0][2])), "plate": ""})

    labels_path = os.path.join(out_dir, "labels.csv")
    if os.path.exists(labels_path):
        # Never clobber typing already done. Merge: keep existing plates, add new tracks.
        existing = {r["track_id"]: r.get("plate", "") for r in read_csv(labels_path)}
        for r in rows:
            r["plate"] = existing.get(r["track_id"], "")
        print(f"[label] merged into existing {labels_path} (your plates kept)")
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["track_id", "n_obs", "best_frame",
                                            "best_area", "plate"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"[label] {len(rows)} candidate tracks -> {crops_dir}")
    print(f"[label] fill the `plate` column in {labels_path}")
    print("[label] ONLY label plates you can read by eye at reference quality; "
          "leave the rest blank.")


def read_csv(path: str) -> list:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def load_labels(path: str) -> dict:
    """track_id -> ground-truth digits. Blank / '?' / 'skip' rows are dropped."""
    out = {}
    for row in read_csv(path):
        raw = (row.get("plate") or "").strip()
        if not raw or raw in ("?", "-", "skip", "SKIP"):
            continue
        d = digits_only(raw)
        if len(d) not in (7, 8):
            print(f"[labels] track {row['track_id']}: {raw!r} is not 7 or 8 digits -- skipped")
            continue
        out[str(row["track_id"])] = d
    return out


# --------------------------------------------------------------------------- #
# stage 3 -- encode ladder
# --------------------------------------------------------------------------- #
def encode_rung(src: str, dst: str, codec: str, kbps: int, *, preset: str, gop: int) -> None:
    """Re-encode `src` at one rung.

    Bitrate-targeted VBR with a 2x peak, NOT CRF: the phone's encoder is given a bitrate
    target, so the ladder has to be in the same units as the decision we are making.
    `-fps_mode passthrough` is load-bearing -- any frame duplication or drop would shift
    every frame index and silently misalign the whole experiment against the reference.
    """
    vcodec = {"h264": "libx264", "hevc": "libx265"}[codec]
    cmd = [
        FFMPEG, "-y", "-loglevel", "error", "-i", src,
        "-c:v", vcodec, "-preset", preset,
        "-b:v", f"{kbps}k", "-maxrate", f"{2 * kbps}k", "-bufsize", f"{4 * kbps}k",
        "-g", str(gop), "-pix_fmt", "yuv420p",
        "-fps_mode", "passthrough", "-an", "-sn",
    ]
    if codec == "hevc":
        cmd += ["-x265-params", "log-level=error", "-tag:v", "hvc1"]
    cmd += [dst]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise RuntimeError(f"ffmpeg failed for {codec}@{kbps}k: "
                           f"{(proc.stderr or '').strip()[:400]}")


# --------------------------------------------------------------------------- #
# stage 4 -- score one rung
# --------------------------------------------------------------------------- #
def score_rung(video: str, ref: dict, labels: dict, reader, *,
               max_ocr_frames: int, detect_model):
    """One sequential pass over `video`, cropping the REFERENCE boxes at the REFERENCE frames.

    Sequential, never `cap.set(POS_FRAMES)`: seeking long-GOP video through OpenCV is both
    slow and, on some builds, off by a frame or two -- which would quietly compare different
    pictures across rungs and invent a compression effect that is not there.
    """
    wanted = defaultdict(list)            # frame_id -> [(track_id, bbox)]
    for tid in labels:
        for _score, fid, bbox in ref["tracks"][tid]["best"][:max_ocr_frames]:
            wanted[fid].append((tid, bbox))
    detect_ref = {int(k): v for k, v in ref["detect_ref"].items()} if detect_model else {}

    reads: dict[str, list] = defaultdict(list)     # track_id -> [(plate, conf)]
    attempts: dict[str, int] = defaultdict(int)
    det_hit = det_total = 0
    det_hit_big = det_total_big = 0
    iou_sum = 0.0

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open rung video: {video}")
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, bbox in wanted.get(fid, []):
            crop = safe_crop(frame, bbox)
            if crop is None:
                continue
            attempts[tid] += 1
            plate, conf = reader.read_plate_with_conf(crop)
            if plate:
                reads[tid].append((plate, conf))
        if detect_model is not None and fid in detect_ref:
            gt = detect_ref[fid]
            if gt:
                res = detect_model.predict(frame, verbose=False,
                                           classes=Constants.DETECTION_CLASSES,
                                           conf=ref["conf"], imgsz=ref["imgsz"])
                pred = [tuple(int(v) for v in b.xyxy[0].tolist())
                        for b in res[0].boxes
                        if Constants.is_vehicle(int(b.cls.item()))]
                used = set()
                for g in gt:
                    best_i, best_j = 0.0, -1
                    for j, p in enumerate(pred):
                        if j in used:
                            continue
                        v = iou(g, p)
                        if v > best_i:
                            best_i, best_j = v, j
                    matched = best_i >= IOU_MATCH
                    if matched:
                        used.add(best_j)
                        iou_sum += best_i
                    det_total += 1
                    det_hit += int(matched)
                    if bbox_area(g) >= ref["min_area"]:   # the ones big enough to carry a plate
                        det_total_big += 1
                        det_hit_big += int(matched)
        fid += 1
    cap.release()

    if fid != ref["n_frames"]:
        print(f"[warn] {os.path.basename(video)} has {fid} frames, reference has "
              f"{ref['n_frames']} -- frame indices may be misaligned")

    per_vehicle, exact, chars, read_ok, confs, scores = [], [], [], [], [], []
    for tid, truth in labels.items():
        rs = reads.get(tid, [])
        plate, vote_score = vote_characters(rs)
        pred = digits_only(plate)
        is_exact = bool(pred) and pred == truth
        ca = char_accuracy(truth, pred)
        exact.append(float(is_exact))
        chars.append(ca)
        read_ok.append(float(bool(pred)))
        scores.append(vote_score)
        if rs:
            confs.append(float(np.mean([c for _p, c in rs])))
        per_vehicle.append({
            "track_id": tid, "truth": truth, "predicted": pred or "",
            "exact": int(is_exact), "char_acc": round(ca, 4),
            "n_valid_reads": len(rs), "n_ocr_attempts": attempts.get(tid, 0),
            "vote_score": round(vote_score, 4),
        })

    total_attempts = sum(attempts.values())
    total_valid = sum(len(v) for v in reads.values())
    return {
        "n_vehicles": len(labels),
        "exact_pct": 100.0 * float(np.mean(exact)) if exact else 0.0,
        "char_pct": 100.0 * float(np.mean(chars)) if chars else 0.0,
        "read_pct": 100.0 * float(np.mean(read_ok)) if read_ok else 0.0,
        "frame_read_pct": 100.0 * total_valid / total_attempts if total_attempts else 0.0,
        "mean_ocr_conf": float(np.mean(confs)) if confs else 0.0,
        "mean_vote_score": float(np.mean(scores)) if scores else 0.0,
        "det_recall_pct": 100.0 * det_hit / det_total if det_total else float("nan"),
        "det_recall_big_pct": (100.0 * det_hit_big / det_total_big
                               if det_total_big else float("nan")),
        "mean_matched_iou": iou_sum / det_hit if det_hit else float("nan"),
    }, per_vehicle


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
# (key, column width, decimals, header). Width is shared by header and cells so the table
# lines up; `None` decimals means the value is printed as a plain left-aligned string.
COLUMNS = [
    ("rung", 14, None, "rung"), ("size_mb", 8, 1, "size MB"),
    ("real_kbps", 9, 0, "real kbps"), ("exact_pct", 7, 1, "exact%"),
    ("char_pct", 7, 1, "char%"), ("read_pct", 7, 1, "read%"),
    ("frame_read_pct", 8, 1, "fread%"), ("mean_ocr_conf", 6, 3, "conf"),
    ("det_recall_pct", 7, 1, "det%"), ("det_recall_big_pct", 9, 1, "det-big%"),
]


def print_table(rows: list) -> None:
    header = "  ".join(f"{h:<{w}}" if d is None else f"{h:>{w}}"
                       for _k, w, d, h in COLUMNS)
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for key, width, decimals, _h in COLUMNS:
            v = r.get(key)
            if decimals is None:
                cells.append(f"{str(v):<{width}}")
            else:
                try:
                    cells.append(f"{float(v):>{width}.{decimals}f}")
                except (TypeError, ValueError):
                    cells.append(f"{'-':>{width}}")
        print("  ".join(cells))


def plot_sweep(rows: list, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed -- skipping sweep.png")
        return
    by_codec = defaultdict(list)
    for r in rows:
        if r["codec"] == "source":
            continue
        by_codec[r["codec"]].append(r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    styles = {"h264": ("o-", "tab:blue"), "hevc": ("s-", "tab:red")}
    for codec, rs in by_codec.items():
        rs = sorted(rs, key=lambda r: r["real_kbps"])
        mk, col = styles.get(codec, ("^-", "tab:green"))
        x = [r["real_kbps"] / 1000.0 for r in rs]
        ax1.plot(x, [r["char_pct"] for r in rs], mk, color=col, label=f"{codec} char acc")
        ax1.plot(x, [r["exact_pct"] for r in rs], marker=mk[0], linestyle="--",
                 color=col, alpha=0.4, label=f"{codec} exact")
        ax2.plot(x, [r["size_mb"] for r in rs], mk, color=col, label=codec)
    src = next((r for r in rows if r["codec"] == "source"), None)
    if src:
        ax1.axhline(src["char_pct"], color="grey", ls=":",
                    label=f"source ({src['real_kbps']/1000:.1f} Mbps)")
    ax1.set_xlabel("delivered bitrate (Mbps)")
    ax1.set_ylabel("plate accuracy (%)")
    ax1.set_title("Plate accuracy vs bitrate")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set_xlabel("delivered bitrate (Mbps)")
    ax2.set_ylabel("file size (MB)")
    ax2.set_title("File size vs bitrate")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[plot] {path}")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_label(args):
    os.makedirs(args.out, exist_ok=True)
    tracks_path = os.path.join(args.out, "tracks.json")
    if os.path.exists(tracks_path) and not args.retrack:
        print(f"[label] reusing {tracks_path} (--retrack to redo the YOLO pass)")
        ref = json.load(open(tracks_path, encoding="utf-8"))
    else:
        ref = build_reference_tracks(
            args.video, imgsz=args.imgsz, conf=args.conf, min_area=args.min_area,
            buffer_size=args.buffer_size, detect_stride=args.detect_stride,
            limit_frames=args.limit_frames)
        with open(tracks_path, "w", encoding="utf-8") as fh:
            json.dump(ref, fh)
        print(f"[label] {tracks_path}: {len(ref['tracks'])} tracks over "
              f"{ref['n_frames']} frames")
    write_label_assets(args.video, ref, args.out, min_obs=args.min_obs,
                       n_sheet=args.sheet_crops)


def cmd_run(args):
    tracks_path = os.path.join(args.out, "tracks.json")
    labels_path = os.path.join(args.out, "labels.csv")
    if not os.path.exists(tracks_path):
        raise SystemExit(f"{tracks_path} missing -- run the `label` command first")
    if not os.path.exists(labels_path):
        raise SystemExit(f"{labels_path} missing -- run the `label` command first")
    ref = json.load(open(tracks_path, encoding="utf-8"))
    labels = load_labels(labels_path)
    labels = {t: p for t, p in labels.items() if t in ref["tracks"]}
    if not labels:
        raise SystemExit(f"no usable labels in {labels_path} -- fill the `plate` column")
    print(f"[run] {len(labels)} labelled vehicles")
    if len(labels) < 10:
        print("[run] WARNING: under 10 vehicles. exact% will swing wildly; "
              "read char% instead, and label more cars before trusting a knee.")

    ladder = parse_ladder(args.ladder) if args.ladder else DEFAULT_LADDER
    enc_dir = os.path.join(args.out, "encodes")
    os.makedirs(enc_dir, exist_ok=True)

    from lpr.reader import FastALPRReader
    reader = FastALPRReader()                       # loaded once, reused for every rung
    detect_model = None
    if not args.no_detect_recall:
        from ultralytics import YOLO
        detect_model = YOLO(Constants.YOLO_VERSION)

    rungs = [("source", 0, args.video)]
    for codec, kbps in ladder:
        rungs.append((codec, kbps, os.path.join(enc_dir, f"{codec}_{kbps}k.mp4")))

    rows, per_vehicle_rows = [], []
    results_path = os.path.join(args.out, "results.csv")
    for codec, kbps, path in rungs:
        name = "source" if codec == "source" else f"{codec}@{kbps}k"
        if codec != "source" and not os.path.exists(path):
            print(f"[enc] {name} ...")
            encode_rung(args.video, path, codec, kbps,
                        preset=args.preset, gop=args.gop)
        size_mb = os.path.getsize(path) / 1e6
        print(f"[ocr] {name}  ({size_mb:.1f} MB)")
        metrics, pv = score_rung(path, ref, labels, reader,
                                 max_ocr_frames=args.max_ocr_frames,
                                 detect_model=detect_model)
        row = {"rung": name, "codec": codec, "target_kbps": kbps,
               "size_mb": round(size_mb, 2),
               "real_kbps": round(probe_bitrate_kbps(path), 1), **metrics}
        rows.append(row)
        for r in pv:
            per_vehicle_rows.append({"rung": name, **r})
        write_rows(results_path, rows)          # incremental: a crash keeps what ran
        if codec != "source" and not args.keep_encodes:
            os.remove(path)

    write_rows(os.path.join(args.out, "per_vehicle.csv"), per_vehicle_rows)
    if not args.keep_encodes:
        shutil.rmtree(enc_dir, ignore_errors=True)
    print_table(rows)
    plot_sweep(rows, os.path.join(args.out, "sweep.png"))
    print(f"\n[run] {results_path}")
    print("[run] The knee is where char% first drops clearly below the source row. Ship a "
          "bitrate ABOVE it: libx264/libx265 beat a phone's hardware encoder, so the "
          "on-device knee is higher than this one.")


def write_rows(path: str, rows: list) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def parse_ladder(spec: str) -> list:
    """'h264:6000,hevc:4000' -> [('h264', 6000), ('hevc', 4000)]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        codec, _, kbps = part.partition(":")
        codec = codec.strip().lower()
        if codec not in ("h264", "hevc"):
            raise SystemExit(f"unknown codec {codec!r} (use h264 or hevc)")
        out.append((codec, int(kbps)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--video", required=True, help="reference capture (phone-encoded)")
    common.add_argument("--out", default="bitrate_sweep", help="work directory")
    common.add_argument("--imgsz", type=int, default=Constants.YOLO_IMGSZ)
    common.add_argument("--conf", type=float, default=Constants.CONFIDENCE_LVL)
    common.add_argument("--min-area", type=float, default=Constants.LPR.MIN_VEHICLE_AREA)

    p_lab = sub.add_parser("label", parents=[common],
                           help="track the reference once, write crop sheets + labels.csv")
    p_lab.add_argument("--min-obs", type=int, default=DEFAULT_MIN_OBS,
                       help="min close frames before a track is worth labelling")
    p_lab.add_argument("--sheet-crops", type=int, default=4,
                       help="crops per contact sheet")
    p_lab.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE)
    p_lab.add_argument("--detect-stride", type=int, default=DEFAULT_DETECT_STRIDE)
    p_lab.add_argument("--limit-frames", type=int, default=None,
                       help="stop after N frames (smoke test)")
    p_lab.add_argument("--retrack", action="store_true",
                       help="redo the YOLO pass even if tracks.json exists")
    p_lab.set_defaults(func=cmd_label)

    p_run = sub.add_parser("run", parents=[common], help="encode the ladder and score it")
    p_run.add_argument("--ladder", default=None,
                       help="e.g. 'h264:12000,hevc:6000,hevc:4000' (default: full A/B ladder)")
    p_run.add_argument("--preset", default="medium", help="x264/x265 preset")
    p_run.add_argument("--gop", type=int, default=DEFAULT_GOP)
    p_run.add_argument("--max-ocr-frames", type=int, default=DEFAULT_MAX_OCR_FRAMES,
                       help="sharpest frames OCR'd per vehicle (pipeline default: 8)")
    p_run.add_argument("--no-detect-recall", action="store_true",
                       help="skip the label-free YOLO recall track (faster)")
    p_run.add_argument("--keep-encodes", action="store_true",
                       help="keep the re-encoded rungs for eyeballing")
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
