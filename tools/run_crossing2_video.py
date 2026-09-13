"""Offline crossing2 runner: annotated video + event log, for one or more clips.

Fully local. It never contacts a server, never uploads to Cloudflare, never deletes
anything, and never touches the evidence/export chain - it only reads the clip and
writes into --out-dir.

Pipeline per clip (two passes over the video, models loaded once for all clips):

  pass 1  YOLO vehicle tracking  +  YOLOv8-seg lane model
          seg contours -> centerline polylines (crossing2.contour_to_lane)
          -> one FrameObservation per frame
  solve   crossing2.detect_crossings over the WHOLE clip (offline: centered
          smoothing, two-pass lane association, bidirectional confirmation)
  pass 2  re-read the clip and render the debug overlay

The output video carries FOUR layers so the result can be debugged, not just watched:

  1. RAW lane-model output    - translucent fill + thin outline, per seg class
  2. TRACKED lines (stage 0)  - thick polyline, "L<id> <type>", plus DBL/GORE tags
  3. vehicles                 - green clear / amber strafing / RED crossing, with the
                                ground anchor, the corrected footprint bar, a magenta
                                tick where the marking crosses the contact row, and
                                the live offset u in lane widths
  4. EVENTS log               - every confirmed event with vehicle id and frame range,
                                highlighted while active

A thick tracked line with no raw outline underneath is a coasted/interpolated track -
i.e. the lane model dropped the marking and Stage 0 carried it. That is exactly the
case the old detector lost crossings on, so it is worth looking for.

--cache-dir stores the pass-1 model output, so re-running with different thresholds
skips both networks and takes about a second. --diag prints per-stage rejection
counts, which is how you tell "nothing happened in this clip" from "a gate threw
everything away".

Examples
--------
  python tools/run_crossing2_video.py roadguard_videos/h1/hafrada_1.mp4
  python tools/run_crossing2_video.py roadguard_videos/h*/hafrada_*.mp4 --out-dir outputs/c2
  python tools/run_crossing2_video.py clip.mp4 --cache-dir outputs/cache --diag
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import os
import pickle
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import Constants  # noqa: E402
from line_crossing.crossing2 import (  # noqa: E402
    CrossingConfig, EdgeContactConfig, FrameObservation, VehicleBox, contour_to_lane,
    detect_crossings, detect_edge_contacts, format_events,
)
from line_crossing.crossing2.overlay import CrossingOverlay  # noqa: E402

# The lane-seg head's own class names -> the lane_types contract.
SEG_TO_LANETYPE = {
    "solid_white_lane": "solid_white",
    "yellow_solid_lane": "solid_yellow",
    "dashed_lane": "dashed",
}
# A traffic island is an AREA, not a line: its centerline is meaningless as something
# to cross, and its edges are usually segmented as solid_white_lane anyway. Excluded
# from the line set by default, but still DRAWN, so gore clips stay debuggable.
ISLAND_CLASS = "traffic_island"

DEFAULT_LANE_WEIGHTS = os.path.join(REPO_ROOT, "weights", "phase3_v3_yellowprotect.pt")
CACHE_VERSION = 1


def parse_args():
    ap = argparse.ArgumentParser(description="Offline crossing2 runner (no network, no deletes)")
    ap.add_argument("videos", nargs="+", help="one or more clips (globs are expanded)")
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "outputs", "crossing2"))
    ap.add_argument("--model", default=Constants.YOLO_VERSION, help="vehicle YOLO weights")
    ap.add_argument("--lane-weights", default=DEFAULT_LANE_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=Constants.YOLO_IMGSZ)
    ap.add_argument("--conf", type=float, default=Constants.CONFIDENCE_LVL)
    ap.add_argument("--lane-conf", type=float, default=0.25)
    ap.add_argument("--lane-every-n", type=int, default=1,
                    help="run the lane model every N frames (>1 is faster; Stage 0 "
                         "interpolates the skipped frames, near-field accuracy drops)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=0, help="0 = whole clip")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ego-min-y", type=int, default=0,
                    help="drop boxes whose bottom y exceeds this (ego hood filter); 0 = off")
    ap.add_argument("--include-islands", action="store_true",
                    help="also treat traffic_island contours as crossable lines")
    ap.add_argument("--cache-dir", default=None,
                    help="cache pass-1 model output here; re-runs on the same clip/window/"
                         "weights skip both models, so threshold sweeps take a second")
    ap.add_argument("--diag", action="store_true",
                    help="print per-stage rejection counts")
    ap.add_argument("--no-raw-lanes", action="store_true", help="hide the raw seg layer")
    ap.add_argument("--no-fill", action="store_true", help="outline the raw seg layer only")
    ap.add_argument("--no-video", action="store_true", help="CSV + console only, skip pass 2")
    # crossing2 thresholds worth sweeping by hand
    ap.add_argument("--traverse-margin", type=float, default=None,
                    help="hysteresis band in LANE WIDTHS (default 0.15 ~ 0.5 m)")
    ap.add_argument("--traverse-hold", type=float, default=None, help="seconds held per side")
    ap.add_argument("--straddle-frac", type=float, default=None,
                    help="footprint fraction over the line for the strafe channel")
    ap.add_argument("--near-gate", type=float, default=None,
                    help="min bbox width as a fraction of the local lane width (default 0.25)")
    ap.add_argument("--max-event-aspect", type=float, default=None,
                    help="reject events whose vehicle shows its FLANK (median bbox w/h "
                         "above this) - junction cross traffic. 99 disables (default 2.0)")
    ap.add_argument("--min-samples", type=int, default=None,
                    help="min usable samples before a vehicle-line pair is scored")
    ap.add_argument("--no-strafe", action="store_true", help="report full crossings only")
    ap.add_argument("--no-extrap", action="store_true",
                    help="do NOT guess where a line continues past the paint the model "
                         "actually segmented: lane geometry stops at its observed "
                         "extent. Affects both the detection and the drawn lines")
    # ---- v3: the half-edge contact rule -------------------------------------
    ap.add_argument("--detector", choices=("v2", "v3"), default="v2",
                    help="v2 = signed-offset temporal detector; "
                         "v3 = per-frame bbox half-edge contact rule")
    ap.add_argument("--v3-buffer", type=float, default=0.0,
                    help="inflate the bbox by this fraction of its WIDTH on all sides "
                         "before testing (scales with distance automatically); 0 = off")
    ap.add_argument("--v3-min-frames", type=int, default=1,
                    help="consecutive contact frames needed to report (1 = the raw rule)")
    ap.add_argument("--v3-geometry", choices=("tracked", "raw"), default="tracked",
                    help="test against the Stage 0 tracked lines, or against this "
                         "frame's raw lane polylines")
    ap.add_argument("--v3-use-gates", action="store_true",
                    help="also apply the Stage 1 validity gates (truncated box, "
                         "occluded wheels, too-far vehicle)")
    ap.add_argument("--v3-latch", action="store_true",
                    help="once flagged, keep the vehicle flagged for the rest of its "
                         "track (what the legacy CrossingMonitor did)")
    ap.add_argument("--v3-lane-types", default="",
                    help="comma-separated lane types v3 may fire on, e.g. solid_yellow "
                         "(what main.py runs: yellow to v3, white+gore to the old "
                         "detector). Empty = every solid line. Stage 0 still sees all "
                         "classes either way")
    return ap.parse_args()


def build_config(args) -> CrossingConfig:
    cfg = CrossingConfig()
    if args.traverse_margin is not None:
        cfg.traverse_margin = args.traverse_margin
    if args.traverse_hold is not None:
        cfg.traverse_hold_sec = args.traverse_hold
    if args.straddle_frac is not None:
        cfg.straddle_min_frac = args.straddle_frac
    if args.near_gate is not None:
        cfg.min_bbox_lane_frac = args.near_gate
    if args.min_samples is not None:
        cfg.min_valid_samples = args.min_samples
    if args.max_event_aspect is not None:
        cfg.max_event_aspect = args.max_event_aspect
    if args.no_extrap:
        # Both places the pipeline extends a line beyond what was segmented: Stage 0's
        # per-frame row fill, and Stage 2's bottom-tangent reach for a close vehicle
        # whose wheels sit below the lowest detected row (v2 only).
        cfg.row_fill_extrap_rows = 0
        cfg.extrapolate_frac = 0.0
    return cfg


def build_ec_config(args) -> EdgeContactConfig:
    types = tuple(t.strip() for t in args.v3_lane_types.split(",") if t.strip())
    return EdgeContactConfig(
        buffer_frac=args.v3_buffer, min_frames=args.v3_min_frames,
        use_gates=args.v3_use_gates, geometry=args.v3_geometry, latch=args.v3_latch,
        lane_types=types or None)


# --------------------------------------------------------------------------- #
# pass 1: models
# --------------------------------------------------------------------------- #
def seg_lane_records(lane_model, frame, conf: float, device) -> list[dict]:
    """One frame of the lane model, in the same record shape main.py's `seg_lanes`
    produces: [{"cls": name, "conf": float, "contour": [[x, y], ...]}]."""
    res = lane_model.predict(frame, conf=conf, verbose=False, device=device)[0]
    out: list[dict] = []
    if res.masks is None or res.boxes is None or not len(res.boxes):
        return out
    for poly, c, cf in zip(res.masks.xy, res.boxes.cls.tolist(), res.boxes.conf.tolist()):
        poly = np.asarray(poly, dtype=np.float32)
        if len(poly) < 3:
            continue
        cnt = poly.round().astype(np.int32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(cnt, epsilon=1.5, closed=True).reshape(-1, 2)
        out.append({"cls": res.names[int(c)], "conf": float(cf), "contour": approx.tolist()})
    return out


def collect(args, cap, yolo, lane_model, n, device):
    """Pass 1. Returns (frames, raw) where frames is [(frame_id, [VehicleBox])] and
    raw is {frame_id: [seg record]}. Deliberately stores the RAW seg output, not the
    derived polylines, so the cache stays valid across lane-geometry option changes."""
    frames: list[tuple[int, list[VehicleBox]]] = []
    raw: dict[int, list] = {}
    last_records: list = []
    t_v, t_l = [], []

    for k in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        fid = args.start + k

        t0 = time.perf_counter()
        res = yolo.track(frame, persist=True, verbose=False, tracker=Constants.YOLO_TRACKER,
                         classes=Constants.DETECTION_CLASSES, conf=args.conf,
                         imgsz=args.imgsz, device=device)
        boxes = res[0].boxes if res else None
        names = res[0].names if res else {}
        vehicles: list[VehicleBox] = []
        if boxes is not None:
            for b in boxes:
                if b.id is None:
                    continue
                cls_id = int(b.cls.item())
                if not Constants.is_vehicle(cls_id):
                    continue                    # traffic lights are tracked too; not here
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                if args.ego_min_y and y2 > args.ego_min_y:
                    continue
                vehicles.append(VehicleBox(int(b.id.item()), (x1, y1, x2, y2), names.get(cls_id)))
        t1 = time.perf_counter()

        if k % args.lane_every_n == 0:
            last_records = seg_lane_records(lane_model, frame, args.lane_conf, device)
        t2 = time.perf_counter()

        raw[fid] = last_records
        frames.append((fid, vehicles))
        if k:
            t_v.append(t1 - t0)
            t_l.append(t2 - t1)
        if k % 200 == 0:
            print(f"    frame {fid:>5}: veh={len(vehicles)} seg={len(last_records)}")

    if t_v:
        print(f"    [time] vehicle {1000*sum(t_v)/len(t_v):.0f} ms/f, "
              f"lane {1000*sum(t_l)/len(t_l):.0f} ms/f")
    return frames, raw


def records_to_lanes(records, frame_width: int, include_islands: bool) -> list:
    """Seg contours -> `lane_types.Lane` centerlines, dropping blobs and (by default)
    traffic islands."""
    lanes = []
    for r in records:
        name = r["cls"]
        lt = SEG_TO_LANETYPE.get(name)
        if lt is None:
            if not (include_islands and name == ISLAND_CLASS):
                continue
            lt = "solid"                       # colour-agnostic solid, per lane_types
        lane = contour_to_lane(r["contour"], frame_width, lane_type=lt, score=r["conf"])
        if lane is not None:
            lanes.append(lane)
    return lanes


def build_observations(frames, raw, frame_width: int, include_islands: bool):
    return [FrameObservation(fid, records_to_lanes(raw.get(fid, ()), frame_width,
                                                   include_islands), vehicles)
            for fid, vehicles in frames]


# --------------------------------------------------------------------------- #
# pass-1 cache
# --------------------------------------------------------------------------- #
def cache_path(args, video: str, name: str) -> str | None:
    """Keyed on everything that changes the MODEL output; lane-geometry options are
    deliberately excluded so they stay tunable against a warm cache."""
    if not args.cache_dir:
        return None
    try:
        mtime = int(os.path.getmtime(video))
    except OSError:
        mtime = 0
    key = "|".join(str(x) for x in (
        CACHE_VERSION, os.path.abspath(video), mtime, args.start, args.frames,
        args.model, args.imgsz, args.conf, args.ego_min_y,
        os.path.basename(args.lane_weights), args.lane_conf, args.lane_every_n))
    return os.path.join(args.cache_dir,
                        f"{name}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.pkl")


def cache_load(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        print(f"  [cache] unreadable ({e}); re-running the models")
        return None


def cache_save(path, frames, raw) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump((frames, raw), fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [cache] saved -> {path}")


# --------------------------------------------------------------------------- #
# pass 2 + outputs
# --------------------------------------------------------------------------- #
def render(args, obs, raw, result, out_path, fps, size):
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("    [render] cannot reopen clip; skipping video")
        return None
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        print(f"    [render] cannot open writer {out_path}; skipping video")
        cap.release()
        return None

    ov = CrossingOverlay(result)
    written = 0
    for ob in obs:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_raw_lanes:
            ov.draw_raw_lanes(frame, raw.get(ob.frame_id), fill=not args.no_fill)
        ov.draw_lanes(frame, ob.frame_id)
        for v in ob.vehicles:
            ov.draw_probe(frame, ob.frame_id, v.track_id)   # v3 only; no-op for v2
        states = [ov.draw_vehicle(frame, ob.frame_id, v.track_id, v.bbox) for v in ob.vehicles]
        ov.draw_banner(frame, ob.frame_id, states,
                       extra=f"seg={len(raw.get(ob.frame_id, ()))} lines={len(result.tracks)}")
        ov.draw_event_log(frame, ob.frame_id)
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()
    print(f"    [render] {written} frames -> {out_path}")
    return out_path


def write_csv(events, path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["vehicle_id", "kind", "lane_type", "line_track", "start_frame",
                    "end_frame", "key_frame", "start_sec", "end_sec", "direction",
                    "peak_overlap", "confidence", "is_gore", "is_double",
                    "extrapolated_frac", "details"])
        for e in events:
            w.writerow([e.vehicle_id, e.kind, e.lane_type, e.track_id, e.start_frame,
                        e.end_frame, e.key_frame, e.details.get("start_sec", ""),
                        e.details.get("end_sec", ""), e.direction,
                        round(e.peak_overlap, 3), round(e.confidence, 3),
                        e.is_gore, e.is_double, round(e.extrapolated_frac, 3), e.details])


def process(args, path, get_yolo, get_lane, cfg, device) -> dict:
    name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  !! could not open {path}")
        return {"video": path, "error": "open failed"}
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    n = total - args.start if not args.frames else min(args.frames, max(0, total - args.start))
    print(f"\n=== {name}  {w}x{h} @ {fps:.1f}fps  {n} frames from {args.start}/{total} ===")

    args.video_path = path
    cpath = cache_path(args, path, name)
    cached = cache_load(cpath)
    if cached is not None:
        frames, raw = cached
        cap.release()
        print(f"  [cache] reusing pass-1 output ({len(frames)} frames)")
    else:
        yolo = get_yolo()
        if getattr(yolo, "predictor", None) is not None:
            yolo.predictor = None                     # fresh track ids per clip
        frames, raw = collect(args, cap, yolo, get_lane(), n, device)
        cap.release()
        cache_save(cpath, frames, raw)
    if not frames:
        print("  !! no frames read")
        return {"video": name, "error": "no frames"}

    obs = build_observations(frames, raw, w, args.include_islands)
    stats: dict = {}
    if args.no_extrap:
        print("  [geom] line continuation OFF - geometry stops at the segmented paint")
    t0 = time.perf_counter()
    if args.detector == "v3":
        result = detect_edge_contacts(obs, w, h, fps, cfg, build_ec_config(args),
                                      stats if args.diag else None)
    else:
        result = detect_crossings(obs, w, h, fps, cfg, stats if args.diag else None)
    solve = time.perf_counter() - t0

    if args.diag:
        print("\n  diagnostics:")
        for stage in ("stage0", "stage1", "stage2", "v3"):
            if stage in stats:
                print(f"    {stage}: {stats[stage]}")

    events = [e for e in result.events if not (args.no_strafe and e.kind == "strafe")]
    for e in events:
        e.details["start_sec"] = round(e.start_frame / fps, 2)
        e.details["end_sec"] = round(e.end_frame / fps, 2)

    print(f"\n  lane tracks ({len(result.tracks)}):")
    for t in sorted(result.tracks, key=lambda t: -t.n_obs)[:12]:
        print(f"    L{t.track_id:<3} {t.lane_type:<13} obs={t.n_obs:<5}"
              f"{' SUPPRESSED' if t.suppressed else ''}"
              f"{' DOUBLE' if t.is_double else ''}{' GORE' if t.is_gore else ''}")
    n_cross = sum(1 for e in events if e.kind == "crossing")
    n_straf = sum(1 for e in events if e.kind == "strafe")
    print(f"\n  EVENTS: {n_cross} crossing(s), {n_straf} strafe(s)   "
          f"({len(result.offsets)} vehicle-line pairs scored, solve {solve:.1f}s)")
    print(format_events(events))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"{name}_crossing2_events.csv")
    write_csv(events, csv_path)
    print(f"  [csv] -> {csv_path}")

    out_video = None
    if not args.no_video:
        result.events = events                        # render only what we report
        out_video = render(args, obs, raw, result,
                           os.path.join(args.out_dir, f"{name}_crossing2.mp4"), fps, (w, h))
    return {"video": name, "crossings": n_cross, "strafes": n_straf,
            "csv": csv_path, "out_video": out_video}


def main() -> None:
    args = parse_args()
    paths: list[str] = []
    for p in args.videos:
        hits = sorted(glob.glob(p))
        paths.extend(hits if hits else [p])
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        raise SystemExit(f"no readable videos in: {args.videos}")
    if not os.path.isfile(args.lane_weights):
        raise SystemExit(f"lane weights not found: {args.lane_weights}")

    import torch
    device = 0 if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device == "cpu":
        print("[warn] CUDA requested but unavailable; running on CPU (slow)")

    # Lazy: a fully cached run never loads a network at all.
    holder: dict = {}

    def get_yolo():
        if "v" not in holder:
            print(f"[load] vehicle model {args.model}")
            holder["v"] = YOLO(args.model)
        return holder["v"]

    def get_lane():
        if "l" not in holder:
            print(f"[load] lane model {os.path.basename(args.lane_weights)}")
            holder["l"] = YOLO(args.lane_weights, task="segment")
        return holder["l"]

    cfg = build_config(args)
    summary = [process(args, p, get_yolo, get_lane, cfg, device) for p in paths]

    print("\n================ SUMMARY ================")
    for s in summary:
        if s.get("error"):
            print(f"  {s['video']}: ERROR {s['error']}")
        else:
            print(f"  {s['video']:<24} {s['crossings']:>3} crossing  "
                  f"{s['strafes']:>3} strafe   -> {s['out_video'] or s['csv']}")
    print(f"\noutputs in {os.path.abspath(args.out_dir)} (nothing uploaded, nothing deleted)")


if __name__ == "__main__":
    main()
