"""
Crossing-violation unit-test + parameter-sweep harness.

Scores the "another vehicle is ON a solid white line" violation logic against labeled
dashcam clips and finds the best parameter set. The violator is ALWAYS another vehicle
(YOLOv8 + ByteTrack), never the ego car.

DESIGN (cache-once / sweep-cheap)
  Pass 1 (expensive, ONCE per clip): run the lane SEG model + the vehicle detector/tracker
    on every frame and cache a compact per-frame record to outputs/violation_cache/<clip>.json.
    Reused on later runs unless --refresh.
  Pass 2 (cheap, sweep): read the cache (no model, no video) and for each parameter combo
    compute violation EVENTS and score them against the labels. 36 combos run in seconds.

RULE B: a vehicle is "on a solid line" in a frame if its tire-contact point (bottom-center
  of its bbox) is within `overlap_tol * frame_width` of a solid_white_lane mask contour.
  Being on the line IS the offense (no side-transition). K-of-M persistence + per-track
  episode grouping + cooldown collapse it into one event per on-line stretch per vehicle.

TIME: clips are 25-60 fps. K / M / gap_close / cooldown / latency are defined in SECONDS and
  converted to frames per-clip via that clip's own fps. overlap_tol is a fraction of width.

Run with the GPU env:
  C:/Users/talgx/miniconda3/envs/roadguard-dl/python.exe tools/crossing_violation_test.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict, deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ghost_mask import (compute_verdict_timeline, events_from_timeline, estimate_ego_shift,  # noqa: E402
                        trigger_points, verdict_segment, GhostMaskTracker, LaneLineTracker,
                        reconcile_components, surface_from_components)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS_DIR = os.path.join(REPO, "tests_videos", "raw_videos", "crossing_solid_line")
CACHE_DIR = os.path.join(REPO, "outputs", "violation_cache")
OUT_DIR = os.path.join(REPO, "outputs", "violation_compare")
LANE_WEIGHTS = os.path.join(REPO, "weights", "phase3_v3_yellowprotect.pt")
VEH_WEIGHTS = os.path.join(REPO, "yolov8m.pt")
LABELS_JSON = os.path.join(REPO, "tools", "violation_labels.json")

VEHICLE_CLASSES = [2, 3, 5, 7]          # COCO car / motorcycle / bus / truck

# Ghost / verdict pipeline (geometry fixed in ghost_mask; these are the tunables)
TTL_SEC = 0.5                           # ghost lifetime after lock-on (lowered 1.0->0.5s: a stale
                                        # snapshot can no longer keep firing for a full second)
PHANTOM_MIN_SEC = 0.0                   # phantom existence gate DISABLED (recall-first): a line is
                                        # readable the instant it appears; color/spatial priors still apply
ISLAND_ERODE_FRAC = 0.05                # shrink island masks 5% before judging
K_SECS = [0.05, 0.10, 0.15, 0.25, 0.40] # swept: verdict-hit must hold this long (seconds -> frames)
FP_WEIGHT = 0.1                         # recall-first: a false positive costs far less than a miss
RENDER_K_SEC = 0.05                     # videos render at THIS K (recall-first), NOT the score-winner.
                                        # ~1-2 frames: flag the instant the car is on the line.


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def find_clip(prefix: str) -> str | None:
    for f in sorted(glob.glob(os.path.join(CLIPS_DIR, prefix + "*.mp4"))):
        if "_annotated" not in os.path.basename(f):
            return f
    return None


# --------------------------------------------------------------------------- #
# Pass 1: build per-clip cache (models run here ONLY)
# --------------------------------------------------------------------------- #
def build_cache(prefix: str, lane_model, veh_weights: str, lane_conf: float, veh_conf: float,
                path: str | None = None) -> dict | None:
    if path is None:
        path = find_clip(prefix)
    if not path:
        print(f"[pass1] {prefix}: NO FILE FOUND, skipping")
        return None
    from ultralytics import YOLO
    veh_model = YOLO(veh_weights)        # fresh per clip => fresh tracker state (no ID bleed)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[pass1] {prefix}: {w}x{h}@{fps:.2f}fps {total} frames -> caching")

    frames = []
    fi = 0
    prev_gray = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # --- ego-motion (optical flow on background asphalt) ---
        cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shift = estimate_ego_shift(prev_gray, cur_gray) if prev_gray is not None else (0.0, 0.0)
        prev_gray = cur_gray
        # --- lanes (seg) ---
        lanes = []
        lres = lane_model.predict(frame, conf=lane_conf, verbose=False)[0]
        if lres.masks is not None and lres.boxes is not None and len(lres.boxes):
            for poly, c, cf, b in zip(lres.masks.xy, lres.boxes.cls.tolist(),
                                      lres.boxes.conf.tolist(), lres.boxes.xyxy.tolist()):
                poly = np.asarray(poly, dtype=np.float32)
                if len(poly) < 3:
                    continue
                # approxPolyDP keeps the true boundary shape (straight stays straight) while
                # dropping only redundant collinear points -> accurate distance test + small cache.
                cnt = poly.round().astype(np.int32).reshape(-1, 1, 2)
                approx = cv2.approxPolyDP(cnt, epsilon=1.5, closed=True).reshape(-1, 2)
                lanes.append({"cls": lres.names[int(c)], "conf": round(float(cf), 3),
                              "bbox": [int(round(x)) for x in b],
                              "contour": approx.tolist()})
        # --- vehicles (track) ---
        vehicles = []
        vres = veh_model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=veh_conf,
                               tracker="bytetrack.yaml", verbose=False)[0]
        if vres.boxes is not None and vres.boxes.id is not None:
            for tid, b in zip(vres.boxes.id.tolist(), vres.boxes.xyxy.tolist()):
                x1, y1, x2, y2 = b
                vehicles.append({"track_id": int(tid),
                                 "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                                 "contact": [int(round((x1 + x2) / 2)), int(round(y2))]})
        frames.append({"frame": fi, "vehicles": vehicles, "lanes": lanes,
                       "shift": [round(shift[0], 2), round(shift[1], 2)]})
        fi += 1
        if fi % 500 == 0:
            print(f"   {prefix}: {fi}/{total}")
    cap.release()

    cache = {"prefix": prefix, "path": path, "fps": fps, "w": w, "h": h, "total": fi, "frames": frames}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f"{prefix}.json"), "w") as fh:
        json.dump(cache, fh)
    print(f"[pass1] {prefix}: cached {fi} frames")
    return cache


def load_cache(prefix: str) -> dict | None:
    p = os.path.join(CACHE_DIR, f"{prefix}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Ego-shift backfill (for caches built before shift-caching existed)
# --------------------------------------------------------------------------- #
def ensure_shifts(cache: dict) -> dict:
    """If a cache lacks per-frame ego-motion, compute it from the video (optical flow only,
    no models -> fast) and persist. Idempotent."""
    if cache["frames"] and "shift" in cache["frames"][0]:
        return cache
    path = cache.get("path")
    if not path or not os.path.isfile(path):
        print(f"[shift] {cache['prefix']}: video missing, defaulting shifts to 0")
        for fr in cache["frames"]:
            fr["shift"] = [0.0, 0.0]
        return cache
    print(f"[shift] {cache['prefix']}: backfilling ego-motion (optical flow)")
    cap = cv2.VideoCapture(path)
    prev_gray = None
    for fr in cache["frames"]:
        ok, frame = cap.read()
        if not ok:
            fr["shift"] = [0.0, 0.0]
            continue
        cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = estimate_ego_shift(prev_gray, cur) if prev_gray is not None else (0.0, 0.0)
        prev_gray = cur
        fr["shift"] = [round(s[0], 2), round(s[1], 2)]
    cap.release()
    with open(os.path.join(CACHE_DIR, f"{cache['prefix']}.json"), "w") as fh:
        json.dump(cache, fh)
    return cache


# --------------------------------------------------------------------------- #
# Pass 2: ghost-mask verdict timeline (compute ONCE) + K-consecutive events (cheap sweep)
# --------------------------------------------------------------------------- #
def clip_timeline(cache: dict):
    """Run line-tracking + ghost state machine once over a clip -> per-track verdict timeline."""
    shifts = [fr.get("shift", [0.0, 0.0]) for fr in cache["frames"]]
    return compute_verdict_timeline(cache["frames"], shifts, cache["h"], cache["w"], cache["fps"],
                                    ttl_sec=TTL_SEC, island_erode_frac=ISLAND_ERODE_FRAC,
                                    phantom_min_sec=PHANTOM_MIN_SEC)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_clip(events: list, windows: list, fps: float) -> dict:
    """Greedy event<->window matching; each window earns at most one TP."""
    used = set()
    tp = fn = fp = 0.0
    lat_pen = 0.0
    details = []
    for wd in windows:
        match = None
        for ei, ev in enumerate(events):
            if ei in used:
                continue
            if ev[2] >= wd["start"] and ev[1] <= wd["end"]:   # overlap
                match = ei
                break
        if match is not None:
            used.add(match)
            tp += wd["weight"]
            lat = abs(events[match][1] - wd["start"]) / fps
            lat_pen += min(0.05, 0.02 * lat)
            details.append((wd, "TP", events[match]))
        else:
            fn += wd["weight"]
            details.append((wd, "FN", None))
    fp = float(len(events) - len(used))
    return {"tp": tp, "fp": fp, "fn": fn, "lat": lat_pen,
            "score": tp - FP_WEIGHT * fp - fn - lat_pen, "details": details,
            "n_events": len(events), "n_used": len(used)}


# --------------------------------------------------------------------------- #
# Winner overlay video
# --------------------------------------------------------------------------- #
CMAP = {"solid_white_lane": (255, 0, 255), "dashed_lane": (0, 255, 0),
        "traffic_island": (255, 255, 0), "yellow_solid_lane": (0, 255, 255)}


def render_winner(cache: dict, events: list, windows: list, panel_w: int = 1280) -> None:
    """Overlay: lanes (color+name), per-vehicle trigger dots + GREEN verdict line, red-tinted
    active ghost masks, and a persistent violation banner on the firing vehicle."""
    path, fps, W, h, prefix = cache["path"], cache["fps"], cache["w"], cache["h"], cache["prefix"]
    active_ids = defaultdict(set)
    for tid, s, e in events:
        for f in range(max(0, s), min(cache["total"] + 1, e + 1)):
            active_ids[f].add(tid)
    in_window = np.zeros(cache["total"] + 1, dtype=bool)
    for wd in windows:
        in_window[wd["start"]:min(len(in_window), wd["end"] + 1)] = True

    fr_by_idx = {f["frame"]: f for f in cache["frames"]}
    ttl = max(1, round(TTL_SEC * fps))
    phantom = max(0, round(PHANTOM_MIN_SEC * fps))
    lt = LaneLineTracker(h, W, phantom)
    gt = GhostMaskTracker(h, W, ttl=ttl, island_erode_frac=ISLAND_ERODE_FRAC)

    cap = cv2.VideoCapture(path)
    pw, ph = panel_w, int(panel_w * h / W)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{prefix}_winner.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rec = fr_by_idx.get(fi)
        firing = active_ids.get(fi, set())
        if rec is not None:
            comps, lab, union = reconcile_components(rec["lanes"], h, W)
            info = lt.update(comps)
            surface = surface_from_components(comps, lab, union, info, h, W, ISLAND_ERODE_FRAC)
            gt.step(rec["vehicles"], surface, tuple(rec.get("shift", [0.0, 0.0])))
            # red-tinted active ghost masks (the "memory" carrying the line under the car)
            if gt.ghosts:
                gmask = np.zeros((h, W), np.uint8)
                for g in gt.ghosts.values():
                    gmask = cv2.bitwise_or(gmask, g["mask"])
                tint = frame.copy()
                tint[gmask > 0] = (0, 0, 255)
                cv2.addWeighted(tint, 0.35, frame, 0.65, 0, frame)
            # draw each tracked physical line with its STABILIZED class + line id + readiness
            for c in comps:
                meta = info.get(c["id"], {})
                cls = meta.get("stable_cls", c["cls"])
                col = CMAP.get(cls, (180, 180, 180))
                cm = (lab == c["id"]).astype(np.uint8)
                cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.polylines(frame, cnts, True, col, 2)
                tag = f"{cls} L{meta.get('line_id', '?')}" + ("" if meta.get("readable") else " (phantom)")
                cv2.putText(frame, tag, (int(c["cx"]), int(c["cy"])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            for v in rec["vehicles"]:
                x1, y1, x2, y2 = v["bbox"]
                tl, tr = trigger_points(v["bbox"])
                (vx0, vy), (vx1, _) = verdict_segment(v["bbox"])
                hit = v["track_id"] in firing
                col = (0, 0, 255) if hit else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if hit else 2)
                cv2.circle(frame, tl, 5, (0, 200, 255), -1)            # trigger dots (amber)
                cv2.circle(frame, tr, 5, (0, 200, 255), -1)
                cv2.line(frame, (vx0, vy), (vx1, vy), (0, 255, 0), 3)   # VERDICT LINE = green
                if hit:
                    cv2.putText(frame, f"VIOLATION #{v['track_id']}", (x1, max(y1 - 10, 28)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        if firing:
            cv2.rectangle(frame, (0, 0), (W - 1, h - 1), (0, 0, 255), 14)
            cv2.rectangle(frame, (0, 14), (W - 1, 70), (0, 0, 0), -1)
            ids = ",".join(f"#{t}" for t in sorted(firing))
            cv2.putText(frame, f"VIOLATION  vehicle {ids}", (20, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
        gtc = (0, 220, 0) if (fi < len(in_window) and in_window[fi]) else (60, 60, 60)
        cv2.rectangle(frame, (0, 0), (W - 1, 12), gtc, -1)
        cv2.putText(frame, f"{prefix} f{fi}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(cv2.resize(frame, (pw, ph)))
        fi += 1
    cap.release()
    writer.release()
    print(f"   winner video -> {out_path}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Crossing-violation unit-test + sweep harness.")
    ap.add_argument("--labels", default=LABELS_JSON)
    ap.add_argument("--lane-weights", default=LANE_WEIGHTS)
    ap.add_argument("--vehicle-weights", default=VEH_WEIGHTS)
    ap.add_argument("--lane-conf", type=float, default=0.25)
    ap.add_argument("--veh-conf", type=float, default=0.30)
    ap.add_argument("--refresh", action="store_true", help="rebuild caches even if present")
    ap.add_argument("--cache-only", action="store_true", help="only run pass 1")
    ap.add_argument("--no-video", action="store_true", help="skip winner videos")
    ap.add_argument("--render-k", type=float, default=RENDER_K_SEC,
                    help="K (sec) the videos render at (recall-first override, not the score-winner)")
    args = ap.parse_args()

    with open(args.labels) as fh:
        labels = json.load(fh)["clips"]
    print(f"[labels] {len(labels)} clips: {list(labels)}")

    # ---- Pass 1: ensure caches ----
    lane_model = None
    caches = {}
    for prefix in labels:
        cache = None if args.refresh else load_cache(prefix)
        if cache is None:
            if lane_model is None:
                from ultralytics import YOLO
                lane_model = YOLO(args.lane_weights, task="segment")
                print(f"[pass1] lane model {args.lane_weights}  classes={lane_model.names}")
            cache = build_cache(prefix, lane_model, args.vehicle_weights, args.lane_conf, args.veh_conf)
        else:
            print(f"[pass1] {prefix}: cache hit ({cache['total']} frames)")
        if cache is not None:
            caches[prefix] = cache

    if args.cache_only:
        print("[done] cache-only")
        return

    # ---- ensure ego-motion present, then compute ghost timelines ONCE per clip ----
    for prefix in caches:
        caches[prefix] = ensure_shifts(caches[prefix])
    print(f"\n[pass2] computing ghost verdict timelines for {len(caches)} clips...")
    timelines = {p: clip_timeline(c) for p, c in caches.items()}

    # ---- sweep K (consecutive verdict-hit frames); everything else fixed ----
    results = []
    for k_sec in K_SECS:
        agg = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "lat": 0.0}
        per_clip = {}
        for p, c in caches.items():
            kf = max(1, round(k_sec * c["fps"]))
            ev = events_from_timeline(timelines[p], kf)
            sc = score_clip(ev, labels[p]["windows"], c["fps"])
            per_clip[p] = sc
            for kk in agg:
                agg[kk] += sc[kk]
        results.append({"k": k_sec, "score": agg["tp"] - FP_WEIGHT * agg["fp"] - agg["fn"] - agg["lat"],
                        "agg": agg, "per_clip": per_clip})
    results.sort(key=lambda r: r["score"], reverse=True)

    print("\n" + "=" * 60)
    print("ghost-mask + spatial-reconcile  (stabilizer dropped)")
    print(f"{'K_s':>5} | {'score':>7} {'TP':>5} {'FP':>4} {'FN':>5} {'lat':>5}")
    print("-" * 60)
    for r in results:
        a = r["agg"]
        print(f"{r['k']:>5} | {r['score']:>7.2f} {a['tp']:>5.1f} {a['fp']:>4.0f} {a['fn']:>5.1f} {a['lat']:>5.2f}")
    print("=" * 60)

    win = results[0]
    print(f"\n[WINNER] K_sec={win['k']}  TTL={TTL_SEC}s  phantom={PHANTOM_MIN_SEC}s  "
          f"FP_w={FP_WEIGHT}  score={win['score']:.2f}")
    for p, sc in win["per_clip"].items():
        print(f"   {p:<14} score={sc['score']:>6.2f}  TP={sc['tp']:.1f} FP={sc['fp']:.0f} "
              f"FN={sc['fn']:.1f}  events={sc['n_events']}")

    if not args.no_video:
        print(f"\n[render] overlay videos at RENDER K={args.render_k}s "
              f"(recall-first; score-winner was K={win['k']}s)...")
        for p, c in caches.items():
            kf = max(1, round(args.render_k * c["fps"]))
            ev = events_from_timeline(timelines[p], kf)
            sc = score_clip(ev, labels[p]["windows"], c["fps"])
            print(f"   {p:<14} @K={args.render_k}s: TP={sc['tp']:.1f} FP={sc['fp']:.0f} "
                  f"FN={sc['fn']:.1f}  events={sc['n_events']}")
            render_winner(c, ev, labels[p]["windows"])
    print("\n[done]")


if __name__ == "__main__":
    main()
