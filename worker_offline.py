"""
worker_offline.py -- run the Road Guard worker WITHOUT the backend, Cloudflare R2 or any network.

This is the offline twin of `worker.py`. The online worker claims a job from the Node API,
downloads the session files from R2, runs the CV pipeline, uploads the evidence clips back to R2
and POSTs one violation record per finding. This script does exactly the same processing, but
both ends are the LOCAL DISK:

    claim a job          ->  walk a local folder of sessions
    download from R2     ->  read the files that are already there (never copied, never moved)
    run the pipeline     ->  main.process_video_with_models(...)   [the same call, same flags]
    upload evidence      ->  write every artefact into <session>/output/
    POST /violation      ->  write the exact JSON payloads into <session>/output/violations.json
    POST /drive/complete ->  write <session>/output/drive.json

NOTHING IS DELETED. The source video and any sensor CSVs are opened read-only and stay exactly
where they were; every product of the run lands under a separate `output/` directory.

TWO KINDS OF INPUT
------------------
1. A full session (what the Android app records) -- the video plus the six sensor artefacts:

       videos/sessions/session_1782773102655/
           roadguard_1782773102655.mp4
           frames.csv  gyro.csv  gravity.csv  gps.csv     <- required for speed
           linacc.csv  intrinsics.json                     <- optional

   Everything runs: speed, speeding, lane rules, plates, speed plots, GPS on every violation.

2. A VIDEO ON ITS OWN -- a folder holding just an .mp4, or loose .mp4 files. The pipeline has no
   ego pose, so it cannot compute anyone's speed. It still detects solid-line crossings and
   yellow-line/shoulder driving, still reads plates, and still cuts a clip per violation. Speed
   plots are not produced, `calculatedSpeed` is 0 and lat/lon are 0 -- each violation record says
   so explicitly via "speedKnown": false / "locationKnown": false.

3. LANE MODE (`--lanes-only`, or the `run_lanes.py` shortcut) -- run the lane rules and NOTHING
   speed-related, whatever files the session carries. Solid-line crossing, yellow-line/shoulder
   driving, plates, clips and the annotated video all run; the speed stage and the Overpass
   lookup are switched off.

In every mode the detected lane markings are DRAWN into the annotated video and into each
violation clip (`--no-lane-overlay` turns that off).

TIMING
------
Every run ends with a per-stage report (run_timing.py): how long the YOLO vehicle model
spent on the video, the lane-segmentation model, the license-plate model, the Stage-2 tire
model, the speed stage, the rendering -- plus the total wall time from launching this script
to the last line printed. It is measurement only (stopwatches wrapped around the existing
calls), so the outputs are identical with or without it; `--no-timing` hides the report.

Usage:

    python worker_offline.py videos/sessions                  # every session under the root
    python worker_offline.py videos/sessions/session_123      # one session
    python worker_offline.py videos/clips/drive1.mp4          # one bare video

    python worker_offline.py videos/clips --lanes-only                # lane rules, no speed
    python run_lanes.py videos/clips                                  # the same thing, shorter
    python worker_offline.py videos/sessions --fixed-speed-limit 90   # speeding with no internet
    python worker_offline.py videos/sessions --force                  # redo finished sessions
    python worker_offline.py videos/sessions --list                   # just show what was found

Any flag this script does not recognise is left in sys.argv, so every main.py tuning flag still
works unchanged, e.g.:

    python worker_offline.py videos/sessions --lateral-ref near_edge --smoother kalman

Exit code is 0 when every discovered session processed, 1 when at least one failed.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from collections import namedtuple
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                       # so `import main` works from any cwd
    sys.path.insert(0, _HERE)

import worker_common as wc
import run_timing                                # per-stage stopwatch (report at the end)

# One unit of work. `dir` is where the source files live, `out_dir` where everything we make goes.
Session = namedtuple("Session", "id dir video out_dir")

# --lanes-only: reported as its own mode, because it is a deliberate choice rather than a
# consequence of which files the session happens to carry.
LANES_ONLY = "lanes-only"
LANES_ONLY_DESC = ("lanes only -> solid-line crossing + yellow-line/shoulder rules and plate "
                   "reading; speed estimation SKIPPED (no speeding, no speed plots, no network)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="worker_offline.py",
        description="Run the Road Guard CV worker fully offline against local videos/sessions.",
        epilog="Unrecognised flags are passed through to main.py (put them at the END of the line).")
    p.add_argument("path", help="a sessions ROOT (e.g. videos/sessions), one session DIR, or a video file")
    p.add_argument("--out-root", default=None,
                   help="write outputs under OUT_ROOT/<sessionId>/ instead of beside the input")
    p.add_argument("--out-name", default="output",
                   help="name of the output folder (default: output)")
    p.add_argument("--force", action="store_true",
                   help="reprocess sessions already marked processed")
    p.add_argument("--list", action="store_true", dest="list_only",
                   help="list the sessions that would be processed, then exit")
    p.add_argument("--fixed-speed-limit", type=float, default=None, metavar="KMH",
                   help="use this speed limit for the whole clip instead of the OpenStreetMap "
                        "Overpass lookup (the only network call the pipeline makes). Needs GPS, "
                        "so it does nothing for a video-only session")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (smoke test)")
    p.add_argument("--lane-conf", type=float, default=0.25,
                   help="lane-segmentation confidence threshold (default 0.25)")
    p.add_argument("--simulation", action="store_true",
                   help="CARLA clip: read telemetry.csv beside the video instead of the sensor CSVs")
    p.add_argument("--lanes-only", action="store_true",
                   help="LANE MODE: detect solid-line crossing + yellow-line/shoulder driving and "
                        "read plates, with NO speed estimation at all (no speeding, no speed "
                        "plots, no Overpass lookup) -- even when the session has sensor CSVs")
    p.add_argument("--no-lane-overlay", action="store_true",
                   help="do NOT draw the detected lane markings into the annotated video and the "
                        "violation clips (the overlay is on whenever the lane model runs)")
    p.add_argument("--no-yellow", action="store_true", help="skip the lane-seg model (no yellow/solid rules)")
    p.add_argument("--no-evidence", action="store_true", help="skip license-plate reading (FastALPR)")
    p.add_argument("--no-stage2", action="store_true", help="skip the tire-model crossing confirmation")
    p.add_argument("--benchmark", action="store_true",
                   help="CSV-only fast mode: NO annotated video and NO speed plots")
    p.add_argument("--no-bundle", action="store_true",
                   help="delete the .tar.gz once it is unpacked into violations/ -- it is a "
                        "byte-for-byte duplicate of that folder and roughly doubles disk use")
    p.add_argument("--worker-id", default=None, help="name recorded in drive.json (default: host name)")
    p.add_argument("--no-timing", action="store_true",
                   help="do NOT print the end-of-run timing report (which model spent how long)")
    return p.parse_known_args(argv)


# --------------------------------------------------------------------------- #
# Session discovery
# --------------------------------------------------------------------------- #
def _out_dir_for(args, session_id: str, own_dir: str | None, parent_dir: str) -> str:
    """Where this session's outputs go.

    * `--out-root DIR`      -> DIR/<sessionId>            (always, so a read-only input tree works)
    * session in its own    -> <session_dir>/output/
      folder
    * loose video sharing a -> <folder>/output/<video_stem>/   (one video's outputs can never
      folder with others       collide with another's)
    """
    if args.out_root:
        return os.path.join(os.path.abspath(args.out_root), session_id)
    if own_dir is not None:
        return os.path.join(own_dir, args.out_name)
    return os.path.join(parent_dir, args.out_name, session_id)


def _session_for_dir(args, session_dir: str) -> Session | None:
    video = wc.find_video(session_dir)
    if not video:
        return None
    session_id = os.path.basename(os.path.normpath(session_dir))
    return Session(session_id, session_dir, video,
                   _out_dir_for(args, session_id, session_dir, os.path.dirname(session_dir)))


def _session_for_loose_video(args, video: str) -> Session:
    """A bare video file: its folder may hold other videos, so key the output by the file name."""
    parent = os.path.dirname(os.path.abspath(video))
    session_id = os.path.splitext(os.path.basename(video))[0]
    return Session(session_id, parent, video, _out_dir_for(args, session_id, None, parent))


def discover_sessions(args) -> list[Session]:
    """A video file, a session folder, or a root holding session folders and/or loose videos."""
    path = os.path.abspath(args.path)

    if os.path.isfile(path):
        parent = os.path.dirname(path)
        # Pointed at the video inside a real session -> treat the folder as the session.
        if wc.has_sensor_csvs(parent) or (args.simulation and
                                          os.path.isfile(os.path.join(parent, "telemetry.csv"))):
            session_id = os.path.basename(os.path.normpath(parent))
            return [Session(session_id, parent, path,
                            _out_dir_for(args, session_id, parent, os.path.dirname(parent)))]
        return [_session_for_loose_video(args, path)]

    if not os.path.isdir(path):
        sys.exit(f"[offline] no such path: {path}")

    videos_here = [os.path.join(path, n) for n in sorted(os.listdir(path))
                   if os.path.isfile(os.path.join(path, n))
                   and os.path.splitext(n)[1].lower() in wc.VIDEO_EXTS
                   and not wc.VIOLATION_CLIP_RE.match(n)
                   and not n.endswith("_annotated.mp4")]
    subdirs = [os.path.join(path, n) for n in sorted(os.listdir(path))
               if os.path.isdir(os.path.join(path, n)) and n != args.out_name]

    # The folder IS one session only when its own files say so -- sensor CSVs (or telemetry.csv)
    # sitting beside the video. A folder holding bare videos is a ROOT, however many it holds, so
    # the id and the output path of a given video never change when a second one is dropped in.
    if videos_here and (wc.has_sensor_csvs(path)
                        or (args.simulation and os.path.isfile(os.path.join(path, "telemetry.csv")))):
        session = _session_for_dir(args, path)
        return [session] if session else []

    # Otherwise it is a root: every sub-folder with a video, plus every loose video in it.
    sessions = [s for s in (_session_for_dir(args, d) for d in subdirs) if s]
    sessions += [_session_for_loose_video(args, v) for v in videos_here]
    return sessions


def _if_exists(out_dir: str, name: str) -> str | None:
    """Index an output only when the run actually produced it (the speed CSVs and the overspeed
    report are absent whenever there is no ego pose, or under --lanes-only)."""
    return name if os.path.isfile(os.path.join(out_dir, name)) else None


def already_processed(out_dir: str) -> bool:
    """True only for a session that finished SUCCESSFULLY: a failed one is retried on the next
    run without needing --force."""
    path = os.path.join(out_dir, "drive.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh).get("status") == "processed"
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# One session = one job
# --------------------------------------------------------------------------- #
def process_session(main, args, models, session: Session, worker_id: str,
                    lanes=None) -> dict:
    video_name = os.path.splitext(os.path.basename(session.video))[0]
    out_dir = session.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # --lanes-only overrides what the files would otherwise allow: no speed is computed even
    # for a full Android session.
    mode = (LANES_ONLY if args.lanes_only
            else wc.session_mode(session.dir, simulation=args.simulation))
    speed_available = mode not in (LANES_ONLY, wc.VIDEO_ONLY)
    started = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    print(f"\n[offline] ===== {session.id} =====\n[offline] video   : {session.video}"
          f"\n[offline] mode    : {LANES_ONLY_DESC if args.lanes_only else wc.describe_mode(mode)}"
          f"\n[offline] outputs : {out_dir}")

    if lanes is not None:
        lanes.reset()                        # forget the previous clip's lane polygons
    wc.attach_models(main, models)
    wc.reset_speed_limit_lookup()            # re-arm the Overpass circuit breaker for this session

    # `job` is what turns the export stage on. It carries no upload_url, so the bundle is built
    # and written locally instead of being PUT to Cloudflare -- exactly the offline contract.
    # This is the same call worker.py makes for a real job.
    result = main.process_video_with_models(
        session.video, models.yolo, models.lane, models.tire,
        out_dir=out_dir,
        max_frames=args.max_frames,
        is_simulation=args.simulation,
        lane_conf=args.lane_conf,
        benchmark=args.benchmark,
        sim_speed=False,
        job={"job_id": session.id},
        undistorter=None,
    )

    # --- deliver: unpack the bundle, gather the clips and the speed images ---------
    violations_dir = os.path.join(out_dir, "violations")
    bundle_path = os.path.join(out_dir, f"{video_name}_violations_bundle.tar.gz")
    manifest = wc.unpack_bundle(bundle_path, violations_dir)
    clips = wc.move_clips(manifest, out_dir, violations_dir) if manifest else {}
    if manifest and args.no_bundle and os.path.isfile(bundle_path):
        os.remove(bundle_path)                  # unpacked already; the tar.gz is pure duplication
        print(f"[offline] removed {os.path.basename(bundle_path)} (--no-bundle); "
              f"its contents are in violations/")
    plots = (wc.collect_speed_plots(out_dir, video_name) if not args.benchmark
             else {"vehicles": {}, "ego": None})

    detected_at = datetime.now(timezone.utc).isoformat()
    payloads = wc.build_violation_payloads(
        manifest, clips, drive_id=f"offline:{session.id}", session_id=session.id,
        session_dir=session.dir, detected_at=detected_at)

    violations_json = os.path.join(out_dir, "violations.json")
    with open(violations_json, "w", encoding="utf-8") as fh:
        json.dump({"sessionId": session.id, "video": os.path.basename(session.video),
                   "mode": mode, "speedAvailable": speed_available,
                   "laneOverlay": not args.no_lane_overlay,
                   "generatedAt": detected_at, "violationCount": len(payloads),
                   "violations": payloads}, fh, ensure_ascii=False, indent=2)

    files = wc.session_files(session.dir, session.video)
    drive = {
        # POST /api/internal/drive/:id/complete
        "status": "processed",
        "error": None,
        # the Drive document the backend keeps for this session
        "sessionId": session.id,
        "driveId": f"offline:{session.id}",
        "workerId": worker_id,
        "mode": mode,
        "speedAvailable": speed_available,
        "laneOverlay": not args.no_lane_overlay,
        "files": {k: (os.path.relpath(v, session.dir).replace("\\", "/") if v else None)
                  for k, v in files.items()},
        "startedAt": started_iso,
        "finishedAt": datetime.now(timezone.utc).isoformat(),
        "durationSec": round(time.time() - started, 1),
        "violationCount": len(payloads),
        "rawInputsDeleted": False,          # offline: originals are always kept
        "outputs": {
            "violations": os.path.relpath(violations_json, out_dir).replace("\\", "/"),
            "manifest": ("violations/manifest.json" if manifest else None),
            "bundle": (os.path.basename(bundle_path) if os.path.isfile(bundle_path) else None),
            "annotatedVideo": (os.path.relpath(result["annotated_video"], out_dir).replace("\\", "/")
                               if result.get("annotated_video") else None),
            "vehiclesCsv": f"{video_name}_vehicles.csv",
            "perFrameCsv": f"{video_name}_perframe.csv",
            "tracksCsv": f"{video_name}_tracks.csv",
            "violationsCsv": f"{video_name}_violations.csv",
            "vehicleSpeedsCsv": _if_exists(out_dir, f"{video_name}_vehicle_speeds.csv"),
            "overspeedTxt": _if_exists(out_dir, f"{video_name}_overspeed.txt"),
            "overspeedCsv": _if_exists(out_dir, f"{video_name}_overspeed.csv"),
            "evidenceDir": f"{video_name}_evidence",
            "speedPlots": plots,
            "clips": clips,
        },
    }
    with open(os.path.join(out_dir, "drive.json"), "w", encoding="utf-8") as fh:
        json.dump(drive, fh, ensure_ascii=False, indent=2)

    print(f"[offline] {session.id}: {len(payloads)} violation(s), "
          f"{len(plots['vehicles'])} vehicle speed plot(s), "
          f"ego plot {'yes' if plots['ego'] else 'no'}, {drive['durationSec']:.0f}s -> {out_dir}")
    return drive


def write_failure(out_dir, session_id, worker_id, started_iso, error) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    drive = {"status": "failed", "error": str(error)[:500], "sessionId": session_id,
             "driveId": f"offline:{session_id}", "workerId": worker_id,
             "startedAt": started_iso, "finishedAt": datetime.now(timezone.utc).isoformat(),
             "violationCount": 0, "rawInputsDeleted": False}
    with open(os.path.join(out_dir, "drive.json"), "w", encoding="utf-8") as fh:
        json.dump(drive, fh, ensure_ascii=False, indent=2)
    return drive


# --------------------------------------------------------------------------- #
def main_cli():
    args, passthrough = parse_args(sys.argv[1:])
    worker_id = args.worker_id or socket.gethostname()
    if args.lanes_only and args.no_yellow:
        sys.exit("[offline] --lanes-only needs the lane model; drop --no-yellow (it turns the "
                 "lane rules off, which would leave nothing to detect)")

    sessions = discover_sessions(args)
    if not sessions:
        sys.exit(f"[offline] no video found under {os.path.abspath(args.path)}")

    pending = []
    for s in sessions:
        if already_processed(s.out_dir) and not args.force:
            print(f"[offline] skipping {s.id} (already processed; --force to redo)")
            continue
        pending.append(s)

    print(f"[offline] {len(sessions)} session(s) found, {len(pending)} to process")
    for s in pending:
        mode = LANES_ONLY if args.lanes_only else wc.session_mode(s.dir, simulation=args.simulation)
        print(f"           - {s.id}: {os.path.basename(s.video)}  [{mode}]")
    if args.list_only or not pending:
        return 0
    if passthrough:
        print(f"[offline] passing through to main.py: {' '.join(passthrough)}")

    # The stopwatch starts here, before the CV stack is imported, so the report can separate
    # "torch/ultralytics took 12 s to import" from "the models took 4 s to load".
    timer = run_timing.RunTimer()
    with timer.measure("load.imports"):
        import cloud_env
        cloud_env.init()
        import main

    if args.fixed_speed_limit is not None:
        wc.patch_speed_limit(args.fixed_speed_limit)

    # Patch the model loaders BEFORE the weights are read, or their load time is invisible.
    if not args.no_timing:
        timer.install_models(main)
    try:
        with timer.measure("load.total"):
            models = wc.load_models(main, yellow=not args.no_yellow, stage2=not args.no_stage2,
                                    evidence=not args.no_evidence, repo_root=_HERE)
    except RuntimeError as e:
        sys.exit(f"[offline] {e}; nothing can run")

    # Draw the lane model's polygons into the videos, and (with --lanes-only) switch the speed
    # stage off. Both are patches on the imported `main` module -- no pipeline file is modified.
    lanes = None
    overlay = (models.lane is not None) and not args.no_lane_overlay
    if overlay or args.lanes_only:
        import lane_render
        lanes = lane_render.install(main, overlay=overlay, lanes_only=args.lanes_only)
    elif not args.no_lane_overlay:
        print("[lanes] no lane model loaded -> no lane overlay and no lane rules")

    # LAST patch layer, so the stopwatch measures what actually runs: the lane-overlay
    # renderers where lane_render replaced the originals, the stock ones where it did not.
    if not args.no_timing:
        timer.install_pipeline(main)
        timer.install_worker(wc)

    results, failures = [], 0
    for s in pending:
        started_iso = datetime.now(timezone.utc).isoformat()
        with timer.session(s.id):
            try:
                results.append(process_session(main, args, models, s, worker_id, lanes=lanes))
            except Exception as e:                   # one bad clip must not kill the batch
                traceback.print_exc()
                results.append(write_failure(s.out_dir, s.id, worker_id, started_iso, e))
                failures += 1
                print(f"[offline] {s.id} FAILED: {e}")

    print("\n[offline] ===== summary =====")
    for d in results:
        print(f"  {d['status']:>9}  {d['sessionId']}  "
              f"{d.get('violationCount', 0)} violation(s)  {d.get('durationSec', 0)}s")

    root = os.path.abspath(args.path)
    if os.path.isdir(root) and len(sessions) > 1:
        summary_path = os.path.join(root, "offline_run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump({"generatedAt": datetime.now(timezone.utc).isoformat(),
                       "workerId": worker_id, "drives": results}, fh, ensure_ascii=False, indent=2)
        print(f"[offline] batch summary -> {summary_path}")

    if not args.no_timing:
        timer.report(title="RUN TIMING -- worker_offline.py")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main_cli())
