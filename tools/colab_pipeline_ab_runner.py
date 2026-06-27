"""
Colab A/B runner for the INTEGRATED Road Guard pipeline (main.py).

For every test video it runs the integrated main.py TWICE -- once with yolov8m, once
with yolo11x -- as independent subprocesses, each writing its real outputs (annotated
video, _evidence folders, violations CSV, per-frame + per-track CSVs) into its OWN
per-detector folder on Drive. Then it merges the two runs into a frame-by-frame
comparison + a dropped-track recall tally + an OCR-success summary.

Everything is written to Drive INCREMENTALLY (per video) and the run RESUMES from a
checkpoint, so a Colab disconnect never loses a completed video.

WHAT main.py PRODUCES (per detector, in OUTPUT_DIR/<video>/<det>/):
    <name>_annotated.mp4         vehicle overlay video
    <name>_evidence/             best-picture crops + plate per violating vehicle
    <name>_violations.csv        every ViolationEvent (speeding + yellow-line) + plate/OCR
    <name>_perframe.csv          frame, n_vehicles, max_speed_kmh, yellow_conf, yellow_violation
    <name>_tracks.csv            frame, track_id, bbox   (used here for the IoU recall tally)

WHAT THIS RUNNER WRITES (in OUTPUT_DIR/):
    detector_ab_per_frame.csv    v8m vs y11x per frame (speed, yellow conf, yellow violation)
    detector_ab_recall.csv       dropped-track tally (instances + UNIQUE vehicles, near/FAR)
    detector_ab_ocr.csv          per detector: events, plates read, OCR success rate
    checkpoint.json              resume marker

Both YOLO versions come from `ultralytics`; fast_alpr is the OCR backend the evidence
stage needs. Edit CONFIG, then paste this whole file into one Colab cell and run.
"""
from __future__ import annotations

import os
import sys
import csv
import json
import glob
import time
import subprocess
from statistics import median
from collections import defaultdict


# ============================================================================ #
# CONFIG
# ============================================================================ #
DRIVE_ROOT = "/content/drive/MyDrive/malshinon_master"     # repo code AND videos live here
VIDEOS_BASE = os.path.join(DRIVE_ROOT, "tests_videos", "raw_videos")
VIDEO_FOLDERS = ["normal_driving", "crossing_solid_line"]  # process ALL videos in these
LANE_WEIGHTS = os.path.join(DRIVE_ROOT, "weights", "phase3_v3_yellowprotect.pt")
OUTPUT_DIR = os.path.join(DRIVE_ROOT, "ab_test_outputs")

DETECTORS = {"v8m": "yolov8m.pt", "y11x": "yolo11x.pt"}    # order = A, B (A=v8m, B=y11x)
IMGSZ = 1280                 # YOLO inference size for both detectors (1984 = main default, slow)
LANE_CONF = 0.25
MAX_FRAMES = 0               # 0 = whole video; set e.g. 300 for a quick smoke test
BENCHMARK = False            # True -> main.py --benchmark: CSV-only, skips the annotated render
                            # (much faster batch when you only need the comparison CSVs)

IOU_MATCH = 0.5              # box IoU to call two detectors' boxes the same vehicle
FAR_DISTANCE_M = 30.0        # power-law distance >= this counts a miss as "FAR" (distant)
SUBPROCESS_TIMEOUT = 60 * 45 # per (video, detector) hard cap (s)


# ============================================================================ #
# 1. Colab env
# ============================================================================ #
def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def setup_colab() -> None:
    if not in_colab():
        print("[env] not in Colab -- skipping mount / install")
        return
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        print("[env] mounting Drive ...")
        drive.mount("/content/drive")
    print("[env] pip install ultralytics + lapx + fast_alpr + tqdm ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "ultralytics>=8.3.0", "lapx", "fast_alpr", "tqdm"], check=True)


# ============================================================================ #
# 2. Distance (for the FAR bucket) -- import the calibrated model, else inline it
# ============================================================================ #
def get_distance_fn():
    sys.path.insert(0, os.path.join(DRIVE_ROOT, "tools"))
    try:
        from distance import distance_from_pixel_height
        return distance_from_pixel_height
    except Exception:
        A_PL, B_PL = 1541.6977610045572, -0.9219913054628024   # tools/distance.py power-law
        return lambda ph: (A_PL * (ph ** B_PL)) if ph > 0 else 0.0


# ============================================================================ #
# 3. Videos
# ============================================================================ #
def list_videos() -> list[tuple[str, str]]:
    out = []
    for folder in VIDEO_FOLDERS:
        d = os.path.join(VIDEOS_BASE, folder)
        if not os.path.isdir(d):
            print(f"  [WARN] missing folder {d}")
            continue
        vids = []
        for ext in ("*.mp4", "*.MP4", "*.mov", "*.MOV", "*.avi", "*.mkv"):
            vids += glob.glob(os.path.join(d, ext))
        vids = sorted(v for v in vids if "_annotated" not in os.path.basename(v))
        print(f"  {folder}: {len(vids)} videos")
        for v in vids:
            out.append((folder, v))
    return out


# ============================================================================ #
# 4. Run the integrated main.py for one (video, detector)
# ============================================================================ #
def run_main(video_path: str, name: str, det_key: str, det_weights: str, *,
             output_dir: str | None = None, imgsz: int | None = None,
             max_frames: int | None = None) -> str | None:
    det_out = os.path.join(output_dir or OUTPUT_DIR, name, det_key)
    os.makedirs(det_out, exist_ok=True)
    cmd = [sys.executable, os.path.join(DRIVE_ROOT, "main.py"), video_path,
           "--model", det_weights, "--out-dir", det_out,
           "--lane-weights", LANE_WEIGHTS, "--imgsz", str(imgsz or IMGSZ),
           "--lane-conf", str(LANE_CONF),
           "--colab"]            # force headless + smart Drive staging inside main.py
    if BENCHMARK:
        cmd += ["--benchmark"]   # CSV-only: skip the slow annotated-video render
    mf = MAX_FRAMES if max_frames is None else max_frames
    if mf:
        cmd += ["--max-frames", str(mf)]
    log_path = os.path.join(det_out, "run.log")
    print(f"    [{det_key}] main.py -> {det_out}")
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            r = subprocess.run(cmd, cwd=DRIVE_ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=SUBPROCESS_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"    [{det_key}] TIMEOUT after {SUBPROCESS_TIMEOUT}s (see {log_path})")
            return None
    if r.returncode != 0:
        print(f"    [{det_key}] FAILED rc={r.returncode} (see {log_path})")
        return None
    return det_out


# ============================================================================ #
# 5. Read main.py outputs
# ============================================================================ #
def read_perframe(det_out: str, name: str) -> dict:
    path = os.path.join(det_out, f"{name}_perframe.csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[int(row["frame"])] = {
                "nveh": int(row["n_vehicles"]),
                "speed": row["max_speed_kmh"],
                "yconf": row["yellow_conf"],
                "yviol": row["yellow_violation"].strip().lower() == "true",
            }
    return out


def read_tracks(det_out: str, name: str):
    """-> ({frame: [(tid, bbox)]}, {tid: [heights]})."""
    path = os.path.join(det_out, f"{name}_tracks.csv")
    frames, heights = defaultdict(list), defaultdict(list)
    if not os.path.isfile(path):
        return frames, heights
    with open(path) as fh:
        for row in csv.DictReader(fh):
            f = int(row["frame"]); tid = int(row["track_id"])
            bb = (int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))
            frames[f].append((tid, bb))
            heights[tid].append(bb[3] - bb[1])
    return frames, heights


def read_violations(det_out: str, name: str) -> tuple[int, int]:
    """-> (n_events, n_with_plate). A plate counts if it's not blank and not the UNKNOWN flag."""
    path = os.path.join(det_out, f"{name}_violations.csv")
    if not os.path.isfile(path):
        return 0, 0
    n, ok = 0, 0
    with open(path) as fh:
        for row in csv.DictReader(fh):
            n += 1
            plate = (row.get("plate") or "").strip()
            if plate and "UNKNOWN" not in plate.upper():
                ok += 1
    return n, ok


# ============================================================================ #
# 6. Cross-detector IoU recall (instances + unique tracks, near/FAR)
# ============================================================================ #
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match_boxes(a, b, thr):
    cand = sorted(((iou(a[i], b[j]), i, j) for i in range(len(a)) for j in range(len(b))),
                  reverse=True)
    ua, ub, m = set(), set(), []
    for v, i, j in cand:
        if v < thr or i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); m.append((i, j))
    return m, [i for i in range(len(a)) if i not in ua], [j for j in range(len(b)) if j not in ub]


def recall_tally(fa, ha, fb, hb, dist_fn) -> dict:
    rec = {"a_only": 0, "b_only": 0, "a_only_far": 0, "b_only_far": 0}
    matched_a, matched_b = set(), set()
    for f in sorted(set(fa) | set(fb)):
        ba, bb = fa.get(f, []), fb.get(f, [])
        m, only_a, only_b = match_boxes([x[1] for x in ba], [x[1] for x in bb], IOU_MATCH)
        for i, j in m:
            matched_a.add(ba[i][0]); matched_b.add(bb[j][0])
        for i in only_a:
            rec["a_only"] += 1
            if dist_fn(ba[i][1][3] - ba[i][1][1]) >= FAR_DISTANCE_M:
                rec["a_only_far"] += 1
        for j in only_b:
            rec["b_only"] += 1
            if dist_fn(bb[j][1][3] - bb[j][1][1]) >= FAR_DISTANCE_M:
                rec["b_only_far"] += 1

    def tracks(h, matched):
        only = [t for t in h if t not in matched]
        far = sum(1 for t in only if dist_fn(median(h[t])) >= FAR_DISTANCE_M)
        return len(only), far
    rec["a_total_tracks"], rec["b_total_tracks"] = len(ha), len(hb)
    rec["a_only_tracks"], rec["a_only_tracks_far"] = tracks(ha, matched_a)
    rec["b_only_tracks"], rec["b_only_tracks_far"] = tracks(hb, matched_b)
    return rec


# ============================================================================ #
# 7. Incremental writers
# ============================================================================ #
def _ensure(path, header):
    if not os.path.isfile(path):
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(header)


class Writers:
    def __init__(self, outdir, keys):
        os.makedirs(outdir, exist_ok=True)
        self.keys = keys
        self.per_frame = os.path.join(outdir, "detector_ab_per_frame.csv")
        self.recall = os.path.join(outdir, "detector_ab_recall.csv")
        self.ocr = os.path.join(outdir, "detector_ab_ocr.csv")
        self.checkpoint = os.path.join(outdir, "checkpoint.json")
        pf = ["video", "frame", "time_s"]
        for k in keys:
            pf += [f"{k}_nveh", f"{k}_speed_kmh", f"{k}_yellow_conf", f"{k}_yellow_violation"]
        pf += ["agree_yellow_violation"]
        _ensure(self.per_frame, pf)
        _ensure(self.recall, ["video", "v8m_only_instances", "y11x_only_instances",
                              "v8m_only_FAR_instances", "y11x_only_FAR_instances",
                              "v8m_total_tracks", "y11x_total_tracks",
                              "v8m_only_tracks", "y11x_only_tracks",
                              "v8m_only_tracks_FAR", "y11x_only_tracks_FAR"])
        _ensure(self.ocr, ["video", "detector", "n_events", "n_plates_read", "ocr_success_rate"])

    def load_done(self) -> set:
        if os.path.isfile(self.checkpoint):
            try:
                return set(json.load(open(self.checkpoint)).get("done", []))
            except Exception:
                return set()
        return set()

    def mark(self, done):
        json.dump({"done": sorted(done)}, open(self.checkpoint, "w"), indent=2)

    def write_video(self, name, pf_a, pf_b, rec, ocr_a, ocr_b):
        ka, kb = self.keys
        with open(self.per_frame, "a", newline="") as fh:
            wr = csv.writer(fh)
            for f in sorted(set(pf_a) | set(pf_b)):
                ra, rb = pf_a.get(f, {}), pf_b.get(f, {})
                row = [name, f, ""]
                for r in (ra, rb):
                    row += [r.get("nveh", 0), r.get("speed", ""), r.get("yconf", ""),
                            r.get("yviol", "")]
                row.append(ra.get("yviol", False) == rb.get("yviol", False))
                wr.writerow(row)
        with open(self.recall, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [name, rec["a_only"], rec["b_only"], rec["a_only_far"], rec["b_only_far"],
                 rec["a_total_tracks"], rec["b_total_tracks"],
                 rec["a_only_tracks"], rec["b_only_tracks"],
                 rec["a_only_tracks_far"], rec["b_only_tracks_far"]])
        with open(self.ocr, "a", newline="") as fh:
            wr = csv.writer(fh)
            for det, (n, ok) in ((ka, ocr_a), (kb, ocr_b)):
                wr.writerow([name, det, n, ok, round(ok / n, 3) if n else 0.0])


# ============================================================================ #
# 8a. Smoke test -- ONE short clip, both detectors, full chain, ~2 min
# ============================================================================ #
def smoke_test(n_frames: int = 250, imgsz: int = 960):
    """Validate the WHOLE chain end-to-end on the first clip with a small frame cap:
    main.py (both detectors: model load + lane-seg + evidence/OCR + CSV emit) -> merge ->
    recall/OCR. Writes to a SEPARATE ab_test_smoke/ dir (does not touch the batch outputs or
    its checkpoint). Prints PASS/MISSING per expected artifact so you catch a path/weight/OCR
    problem in ~2 minutes instead of 100."""
    setup_colab()
    if not os.path.isfile(os.path.join(DRIVE_ROOT, "main.py")):
        print(f"[smoke] ABORT: no main.py in {DRIVE_ROOT}"); return
    if not os.path.isfile(LANE_WEIGHTS):
        print(f"[smoke] ABORT: lane weights not found: {LANE_WEIGHTS}"); return
    videos = list_videos()
    if not videos:
        print("[smoke] ABORT: no videos found -- check VIDEOS_BASE / VIDEO_FOLDERS"); return

    smoke_dir = os.path.join(DRIVE_ROOT, "ab_test_smoke")
    dist_fn = get_distance_fn()
    folder, video_path = videos[0]
    name = os.path.splitext(os.path.basename(video_path))[0]
    keys = list(DETECTORS)
    print(f"\n[smoke] clip = {folder}/{name}  (<= {n_frames} frames, imgsz {imgsz}, both detectors)")
    print(f"[smoke] outputs -> {smoke_dir}\n")

    t0 = time.time()
    outs = {}
    for det_key, det_weights in DETECTORS.items():
        outs[det_key] = run_main(video_path, name, det_key, det_weights,
                                 output_dir=smoke_dir, imgsz=imgsz, max_frames=n_frames)

    print(f"\n[smoke] artifact check ({time.time()-t0:.0f}s):")
    all_ok = True
    for det in keys:
        d = outs[det]
        if not d:
            print(f"  [{det}] main.py FAILED -> see {os.path.join(smoke_dir, name, det, 'run.log')}")
            all_ok = False
            continue
        expected = [f"{name}_perframe.csv", f"{name}_tracks.csv", f"{name}_violations.csv"]
        if not BENCHMARK:                       # the annotated render is skipped in --benchmark
            expected.append(f"{name}_annotated.mp4")
        for suffix in expected:
            p = os.path.join(d, suffix)
            ok = os.path.isfile(p)
            all_ok = all_ok and ok
            print(f"  [{det}] {'OK     ' if ok else 'MISSING'} {suffix}")
        ev = os.path.join(d, f"{name}_evidence")
        print(f"  [{det}] evidence dir {'present' if os.path.isdir(ev) else 'none (no violations -> expected)'}")

    if not all(outs.values()):
        print("\n[smoke] FAIL: a detector run did not finish -- read the run.log printed above.")
        return

    # exercise the merge path exactly as the batch does (separate smoke writers)
    pf_a, pf_b = read_perframe(outs[keys[0]], name), read_perframe(outs[keys[1]], name)
    fa, ha = read_tracks(outs[keys[0]], name)
    fb, hb = read_tracks(outs[keys[1]], name)
    rec = recall_tally(fa, ha, fb, hb, dist_fn)
    ocr_a, ocr_b = read_violations(outs[keys[0]], name), read_violations(outs[keys[1]], name)
    Writers(smoke_dir, keys).write_video(name, pf_a, pf_b, rec, ocr_a, ocr_b)

    print(f"\n[smoke] merge OK: per_frame rows v8m={len(pf_a)} y11x={len(pf_b)}")
    print(f"[smoke] recall: v8m_only_tracks={rec['a_only_tracks']} (FAR={rec['a_only_tracks_far']})  "
          f"y11x_only_tracks={rec['b_only_tracks']} (FAR={rec['b_only_tracks_far']})")
    print(f"[smoke] OCR: v8m {ocr_a[1]}/{ocr_a[0]} plates  y11x {ocr_b[1]}/{ocr_b[0]} plates")
    print(f"\n[smoke] {'PASS -- chain healthy; run the FULL BATCH cell.' if all_ok else 'CHECK the MISSING items above.'}")
    print(f"[smoke] comparison CSVs in {smoke_dir} (delete this dir anytime; batch is separate).")


# ============================================================================ #
# 8b. main -- full batch
# ============================================================================ #
def main():
    setup_colab()
    if not os.path.isfile(os.path.join(DRIVE_ROOT, "main.py")):
        print(f"[abort] no main.py in {DRIVE_ROOT}")
        return
    if not os.path.isfile(LANE_WEIGHTS):
        print(f"[abort] lane weights not found: {LANE_WEIGHTS}")
        return
    dist_fn = get_distance_fn()
    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **k): return x

    videos = list_videos()
    keys = list(DETECTORS)
    writers = Writers(OUTPUT_DIR, keys)
    done = writers.load_done()
    todo = [(folder, v) for folder, v in videos
            if os.path.splitext(os.path.basename(v))[0] not in done]
    print(f"\n[run] {len(videos)} videos, {len(done)} done, {len(todo)} to go")
    print(f"[run] detectors={DETECTORS} imgsz={IMGSZ} -> {OUTPUT_DIR}")

    t0 = time.time()
    bar = tqdm(todo, desc="videos", unit="vid")
    for k, (folder, video_path) in enumerate(bar, 1):
        name = os.path.splitext(os.path.basename(video_path))[0]
        try:
            bar.set_postfix_str(f"{folder}/{name[:22]}")
        except Exception:
            pass
        v_t0 = time.time()
        outs = {}
        for det_key, det_weights in DETECTORS.items():
            outs[det_key] = run_main(video_path, name, det_key, det_weights)
        if not all(outs.values()):
            print(f"  [skip] {name}: a detector run failed (see run.log)")
            continue

        ka, kb = keys
        pf_a, pf_b = read_perframe(outs[ka], name), read_perframe(outs[kb], name)
        fa, ha = read_tracks(outs[ka], name)
        fb, hb = read_tracks(outs[kb], name)
        rec = recall_tally(fa, ha, fb, hb, dist_fn)
        ocr_a, ocr_b = read_violations(outs[ka], name), read_violations(outs[kb], name)

        writers.write_video(name, pf_a, pf_b, rec, ocr_a, ocr_b)   # incremental -> Drive
        done.add(name)
        writers.mark(done)

        elapsed = time.time() - t0
        eta = (elapsed / k) * (len(todo) - k)
        print(f"  [saved] {name}  ({time.time()-v_t0:.0f}s; ~{eta/60:.1f} min left for batch)")

    print(f"\n[done] {len(done)}/{len(videos)} videos in {(time.time()-t0)/60:.1f} min")
    print("[done] outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
