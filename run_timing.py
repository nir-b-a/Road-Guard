"""
run_timing.py -- where the time went, without changing what the run does.

At the end of an offline run this prints one table per session: how long the YOLO vehicle
model spent on the video, how long the lane-segmentation model spent, the license-plate
model, the Stage-2 tire model, the speed stage, the clip/video rendering -- plus the total
wall time from the moment python started until the worker finished.

HOW IT ATTACHES (why no pipeline file changed)
----------------------------------------------
Exactly like `lane_render.py`: it swaps functions on the already-imported modules with
wrappers that start a clock, call the ORIGINAL, and add the elapsed time to a bucket. No
argument, no return value and no branch is touched, so a timed run produces byte-identical
outputs to an untimed one. `restore()` puts everything back.

Two install steps, because the model weights are loaded before the rest is patched:

    timer = RunTimer()
    timer.install_models(main)        # BEFORE worker_common.load_models(...)
    ...
    timer.install_pipeline(main)      # AFTER lane_render.install(...), so the wrappers
    timer.install_worker(wc)          # stack on top of the lane overlay rather than
                                      # being overwritten by it
    with timer.session("session_123"):
        ...run one video...
    timer.report()

READING THE TABLE
-----------------
Indented rows are INSIDE the row above them (the tire-model inference is part of the
Stage-2 pass, which is part of the crossing rule), so only the depth-0 rows add up.
"calls" is how many times that stage ran -- 1800 for a per-frame model on a 1800-frame
clip, 1 for a post-loop stage -- and "avg" is therefore the per-frame cost.
"""
from __future__ import annotations

import functools
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager

# --------------------------------------------------------------------------- #
# When did this process really start?
# --------------------------------------------------------------------------- #
# Importing torch + ultralytics costs seconds before a single line of the worker runs, and
# the user asked for the time "from the moment we ran worker.py". psutil (a torch/ultralytics
# dependency in this env) knows the real process start; without it we fall back to the moment
# this module was imported, which the workers do at the top of the file.
_IMPORT_TS = time.time()
try:                                                # pragma: no cover - env dependent
    import psutil
    PROCESS_START_TS: float | None = psutil.Process(os.getpid()).create_time()
except Exception:                                   # pragma: no cover
    PROCESS_START_TS = None


def process_elapsed() -> float:
    """Seconds since the python process started (or since this module was imported)."""
    return time.time() - (PROCESS_START_TS or _IMPORT_TS)


# --------------------------------------------------------------------------- #
# The stages, in report order.  key -> (label, group, depth)
# --------------------------------------------------------------------------- #
G_SETUP    = "start-up: loading the models"
G_LOOP     = "frame loop: the models that run on every frame"
G_ANALYSIS = "post-loop analysis"
G_OUTPUT   = "evidence, export and rendering"
G_WORKER   = "worker book-keeping"
G_TOTAL    = "totals"

STAGES: "OrderedDict[str, tuple[str, str, int]]" = OrderedDict([
    # -- start-up ----------------------------------------------------------- #
    ("load.imports",   ("importing the CV stack (torch, ultralytics)",     G_SETUP, 0)),
    ("load.total",     ("model loading TOTAL",                              G_SETUP, 0)),
    ("load.yolo",      ("YOLO vehicle model (weights)",                     G_SETUP, 1)),
    ("load.lane",      ("lane-segmentation model (weights)",                G_SETUP, 1)),
    ("load.tire",      ("Stage-2 tire model (init, lazy weights)",          G_SETUP, 1)),
    ("load.plate",     ("license-plate model / FastALPR (weights)",         G_SETUP, 1)),
    # -- frame loop --------------------------------------------------------- #
    # loop.open is OUTSIDE loop.total: the handler is built (and the first frame decoded)
    # before the pipeline starts its own loop clock.
    ("loop.open",      ("video open + first frame decode",                  G_LOOP, 0)),
    ("loop.total",     ("frame loop TOTAL (decode + per-frame models)",     G_LOOP, 0)),
    ("loop.decode",    ("video decode (reading frames)",                    G_LOOP, 1)),
    ("loop.yolo",      ("YOLO vehicle model (detect + track)",              G_LOOP, 1)),
    ("loop.post",      ("detection post-process + crop buffer",             G_LOOP, 1)),
    ("loop.lane",      ("lane model (segmentation)",                        G_LOOP, 1)),
    ("loop.flow",      ("optical-flow ego shift",                           G_LOOP, 1)),
    # -- post-loop analysis ------------------------------------------------- #
    ("an.interp",      ("bbox gap interpolation",                           G_ANALYSIS, 0)),
    ("an.distance",    ("distance estimation (+ smoothing)",                G_ANALYSIS, 0)),
    ("an.speed",       ("speed estimation TOTAL",                           G_ANALYSIS, 0)),
    ("an.ego",         ("ego pose from the sensor CSVs",                    G_ANALYSIS, 1)),
    ("an.relevance",   ("relevance rejection",                              G_ANALYSIS, 1)),
    ("an.world",       ("world-frame vehicle speeds",                       G_ANALYSIS, 1)),
    ("an.overspeed",   ("speeding rule",                                    G_ANALYSIS, 0)),
    ("an.speedlimit",  ("OpenStreetMap speed-limit lookup (network)",       G_ANALYSIS, 1)),
    ("an.yellow",      ("yellow-line / shoulder rule",                      G_ANALYSIS, 0)),
    ("an.crossing",    ("solid-line crossing rule (Stage 1 + 2)",           G_ANALYSIS, 0)),
    ("an.stage2",      ("Stage-2 pass (re-reads the video)",                G_ANALYSIS, 1)),
    ("an.tire",        ("tire model (inference)",                           G_ANALYSIS, 2)),
    ("an.ycross",      ("yellow solid-line crossing rule (v3)",             G_ANALYSIS, 0)),
    # -- evidence / export / render ----------------------------------------- #
    ("out.evidence",   ("evidence stage (plate + best crops)",              G_OUTPUT, 0)),
    ("out.plate",      ("license-plate model (FastALPR detect + OCR)",      G_OUTPUT, 1)),
    ("out.export",     ("violation export bundle",                          G_OUTPUT, 0)),
    ("out.clip",       ("per-violation clip render",                        G_OUTPUT, 1)),
    ("out.csv",        ("CSV writers (per-frame, tracks, speeds)",     G_OUTPUT, 0)),
    ("out.plots",      ("speed plots (PNG)",                                G_OUTPUT, 0)),
    ("out.annotated",  ("annotated video render",                           G_OUTPUT, 0)),
    # -- worker book-keeping ------------------------------------------------ #
    ("worker.unpack",  ("unpack the evidence bundle",                       G_WORKER, 0)),
    ("worker.clips",   ("collect clips / plate pictures / speed plots",     G_WORKER, 0)),
    ("worker.payloads",("build the violation records",                      G_WORKER, 0)),
    ("worker.download",("download the session from R2",                     G_WORKER, 0)),
    ("worker.upload",  ("transcode + upload the evidence clips",            G_WORKER, 0)),
    # -- totals ------------------------------------------------------------- #
    ("pipeline.total", ("pipeline TOTAL (process_video_with_models)",       G_TOTAL, 0)),
])

GROUP_ORDER = (G_SETUP, G_LOOP, G_ANALYSIS, G_OUTPUT, G_WORKER, G_TOTAL)

# Rows the report DERIVES rather than measures: parent key -> (children, label).
LOOP_CHILDREN = ("loop.decode", "loop.yolo", "loop.post", "loop.lane", "loop.flow")

LABEL_W = 48


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def fmt_dur(seconds: float) -> str:
    """Human duration: 1h 02m 03s / 4m 05.1s / 12.34 s / 123.4 ms."""
    if seconds >= 3600:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m):02d}m {int(s):02d}s"
    if seconds >= 60:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:04.1f}s"
    if seconds >= 1:
        return f"{seconds:.2f} s"
    return f"{seconds * 1000:.1f} ms"


# --------------------------------------------------------------------------- #
# One bucket of measurements (the start-up phase, or one video)
# --------------------------------------------------------------------------- #
class Bucket:
    def __init__(self, name: str, kind: str = "session"):
        self.name = name
        self.kind = kind                      # "setup" | "session"
        self.totals: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.frames = 0                       # frames the pipeline actually processed
        self.wall = 0.0                       # wall time of the whole bucket

    def add(self, key: str, seconds: float, calls: int = 1) -> None:
        self.totals[key] = self.totals.get(key, 0.0) + seconds
        self.calls[key] = self.calls.get(key, 0) + calls

    def merge_into(self, other: "Bucket") -> None:
        for key, secs in self.totals.items():
            other.add(key, secs, self.calls.get(key, 0))
        other.frames += self.frames
        other.wall += self.wall


# --------------------------------------------------------------------------- #
# The timer itself
# --------------------------------------------------------------------------- #
class RunTimer:
    """Collects per-stage times and prints the end-of-run report."""

    def __init__(self):
        self.created = time.perf_counter()
        self.setup = Bucket("start-up", kind="setup")
        self.sessions: list[Bucket] = []
        self._cur = self.setup
        self._setup_closed = False
        self._patched: list[tuple] = []       # (obj, attr, original) for restore()
        self._owner = threading.get_ident()   # only this thread may record -- see add()

    # -- recording ---------------------------------------------------------- #
    def add(self, key: str, seconds: float, calls: int = 1) -> None:
        """Record `seconds` against `key` in the CURRENT session bucket.

        Ignored when called from any thread other than the one that built the timer.
        `_cur` is a single attribute swapped by session(), so a wrapped function running on a
        background thread (worker.py ships each drive's evidence off-thread) would otherwise
        land its time in whatever session happens to be open -- usually the NEXT drive's --
        and silently corrupt that report. Dropping the sample keeps every table honest; the
        background work is timed and printed by whoever started the thread."""
        if threading.get_ident() != self._owner:
            return
        self._cur.add(key, seconds, calls)

    @contextmanager
    def measure(self, key: str):
        """Time a block of the caller's own code: `with timer.measure("load.total"): ...`"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - t0)

    @contextmanager
    def session(self, name: str):
        """Everything timed inside the block is attributed to this video."""
        if not self._setup_closed:            # the start-up bucket ends at the first session
            self.setup.wall = time.perf_counter() - self.created
            self._setup_closed = True
        bucket = Bucket(name)
        self.sessions.append(bucket)
        previous, self._cur = self._cur, bucket
        t0 = time.perf_counter()
        try:
            yield bucket
        finally:
            bucket.wall = time.perf_counter() - t0
            self._cur = previous

    # -- patching ----------------------------------------------------------- #
    def _wrap(self, obj, attr: str, key: str) -> bool:
        """Replace obj.attr with a timed passthrough. Missing attribute -> no-op."""
        original = getattr(obj, attr, None)
        if original is None or not callable(original):
            return False

        @functools.wraps(original)
        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.add(key, time.perf_counter() - t0)

        setattr(obj, attr, timed)
        self._patched.append((obj, attr, original))
        return True

    def _wrap_all(self, obj, attrs, key: str) -> None:
        for attr in attrs:
            self._wrap(obj, attr, key)

    def wrap(self, obj, attr: str, key: str) -> bool:
        """Public form of _wrap, for a caller that wants to time one of its own functions."""
        return self._wrap(obj, attr, key)

    def restore(self) -> None:
        """Put every patched function back (leaves the collected numbers intact)."""
        for obj, attr, original in reversed(self._patched):
            setattr(obj, attr, original)
        self._patched.clear()

    # -- step 1: the model loaders (patch BEFORE the weights are loaded) ----- #
    def install_models(self, main) -> None:
        self._wrap(main, "loadYoloModel", "load.yolo")
        self._wrap(main, "loadLaneModel", "load.lane")

        # Both of these are imported INSIDE worker_common.load_models, so patching the class
        # here is enough: the `from ... import X` there fetches this very object.
        try:
            from violations.cascade import tire_model as _tm
            self._wrap(_tm.TireModel, "__init__", "load.tire")
            self._wrap(_tm.TireModel, "detect_in_crop", "an.tire")
        except Exception:
            pass
        try:
            from lpr import reader as _reader
            self._wrap(_reader.FastALPRReader, "__init__", "load.plate")
            # The three methods that actually run the plate detector + OCR. read_plate() is
            # left alone on purpose: it delegates to read_plate_with_conf, so wrapping it too
            # would count the same inference twice.
            self._wrap_all(_reader.FastALPRReader,
                           ("read_plate_with_conf", "crop_plate", "locate_plate"), "out.plate")
        except Exception:
            pass

    # -- step 2: the pipeline (patch AFTER lane_render.install) -------------- #
    def install_pipeline(self, main) -> None:
        self._wrap(main, "process_video_with_models", "pipeline.total")

        # --- frame loop ---------------------------------------------------- #
        self._install_video_handler(main)
        self._install_process_frame(main)
        self._install_print_time(main)
        self._wrap(main, "seg_lanes", "loop.lane")
        self._wrap(main, "estimate_ego_shift", "loop.flow")

        # --- post-loop analysis -------------------------------------------- #
        self._wrap(main, "interpolate_bbox_gaps", "an.interp")
        self._wrap_all(main, ("estimateDistance", "smooth_distances"), "an.distance")
        self._wrap(main, "run_speed_estimation", "an.speed")
        self._wrap(main, "estimate_world_speeds", "an.world")
        ego_yaw = getattr(main, "ego_yaw", None)
        if ego_yaw is not None:
            self._wrap_all(ego_yaw, ("ego_heading_from_android", "ego_position_from_android",
                                     "ego_heading_from_telemetry", "ego_position_from_telemetry"),
                           "an.ego")
        relevance = getattr(main, "relevance_flags", None)
        if relevance is not None:
            self._wrap(relevance, "compute_relevance_flags", "an.relevance")
        overspeed = getattr(main, "overspeed", None)
        if overspeed is not None:
            self._wrap_all(overspeed, ("flag_overspeed_vehicles", "flag_speeding_fixed_limit"),
                           "an.overspeed")
            self._wrap(overspeed, "get_track_speed_limits", "an.speedlimit")
        self._wrap(main, "evaluate_yellow_line", "an.yellow")
        self._wrap(main, "evaluate_crossing", "an.crossing")
        self._wrap(main, "_stage2_confirm", "an.stage2")
        self._wrap(main, "evaluate_yellow_crossing", "an.ycross")

        # --- evidence / export / render ------------------------------------ #
        self._wrap(main, "run_evidence_and_report", "out.evidence")
        self._wrap(main, "export_and_push_violations", "out.export")
        self._wrap(main, "annotate_clip", "out.clip")
        self._wrap(main, "write_perframe_and_tracks", "out.csv")
        logger = getattr(main, "distanceLogger", None)
        if logger is not None:
            self._wrap_all(logger, ("export_vehicle_summary", "export_vehicle_speed_series"),
                           "out.csv")
        plots = getattr(main, "draw_vehicle_plots", None)
        if plots is not None:
            self._wrap_all(plots, ("plot_all_vehicles", "plot_ego_speed"), "out.plots")
        video = getattr(main, "annotated_video", None)
        if video is not None:
            self._wrap(video, "render_annotated_video", "out.annotated")

    # -- step 3: the worker's own delivery helpers -------------------------- #
    def install_worker(self, wc) -> None:
        self._wrap(wc, "unpack_bundle", "worker.unpack")
        self._wrap_all(wc, ("move_clips", "collect_plate_images", "collect_speed_plots"),
                       "worker.clips")
        self._wrap(wc, "build_violation_payloads", "worker.payloads")

    # -- the three hooks that need more than a stopwatch --------------------- #
    def _install_video_handler(self, main):
        """Time the decoder by subclassing it: reading a frame happens in __init__ (opening
        the file + the first frame) and in read_next(); get_frame() only hands back what is
        already decoded. The constructor runs BEFORE the pipeline starts its loop clock, so
        it gets its own row instead of inflating the in-loop decode time."""
        base = getattr(main, "VideoHandler", None)
        if base is None:
            return
        timer = self

        class TimedVideoHandler(base):
            def __init__(self, *args, **kwargs):
                t0 = time.perf_counter()
                try:
                    super().__init__(*args, **kwargs)
                finally:
                    timer.add("loop.open", time.perf_counter() - t0)

            def read_next(self):
                t0 = time.perf_counter()
                try:
                    return super().read_next()
                finally:
                    timer.add("loop.decode", time.perf_counter() - t0)

        self._patched.append((main, "VideoHandler", base))
        main.VideoHandler = TimedVideoHandler

    def _install_process_frame(self, main):
        """processFrame already measures its own YOLO and post-process split and returns it --
        so we read those numbers instead of adding a second, coarser clock around them."""
        original = getattr(main, "processFrame", None)
        if original is None:
            return

        @functools.wraps(original)
        def timed(*args, **kwargs):
            yolo_time, post_time, frame_vehicles = original(*args, **kwargs)
            self.add("loop.yolo", yolo_time)
            self.add("loop.post", post_time)
            return yolo_time, post_time, frame_vehicles

        self._patched.append((main, "processFrame", original))
        main.processFrame = timed

    def _install_print_time(self, main):
        """print_time is called once, right after the frame loop, with the loop's start time
        and the frame count -- exactly the two numbers we want for the loop total."""
        original = getattr(main, "print_time", None)
        if original is None:
            return

        @functools.wraps(original)
        def timed(start_time, read_times, yolo_times, postprocess_times, frame_id):
            self.add("loop.total", time.time() - start_time)
            self._cur.frames += int(frame_id)
            return original(start_time, read_times, yolo_times, postprocess_times, frame_id)

        self._patched.append((main, "print_time", original))
        main.print_time = timed

    # -- reporting ---------------------------------------------------------- #
    def _rows(self, bucket: Bucket):
        """(group, label, depth, calls, total, key) for every stage that actually ran, plus
        the derived 'other' row inside the frame loop."""
        rows = []
        for key, (label, group, depth) in STAGES.items():
            total = bucket.totals.get(key)
            if total is None:
                continue
            rows.append((group, label, depth, bucket.calls.get(key, 0), total))
        # What is left of the loop once every per-frame stage is subtracted: the grayscale
        # conversion for the optical flow, and plain python overhead. Listed last, under the
        # other loop children (rows are printed grouped, in the order they are collected).
        loop_total = bucket.totals.get("loop.total")
        if loop_total is not None:
            other = loop_total - sum(bucket.totals.get(k, 0.0) for k in LOOP_CHILDREN)
            if other > 0.05:
                rows.append((G_LOOP, "other (colour convert, loop overhead)", 1, 0, other))
        return rows

    def _print_table(self, bucket: Bucket, denominator: float) -> None:
        rows = self._rows(bucket)
        if not rows:
            print("  (nothing measured)")
            return
        print(f"  {'stage':<{LABEL_W}}{'calls':>7}{'total':>12}{'avg':>12}{'share':>8}")
        for group in GROUP_ORDER:
            group_rows = [r for r in rows if r[0] == group]
            if not group_rows:
                continue
            print(f"  -- {group} " + "-" * max(0, LABEL_W + 39 - len(group) - 4))
            for _, label, depth, calls, total in group_rows:
                name = ("  " * depth) + label
                if len(name) > LABEL_W - 1:
                    name = name[:LABEL_W - 2] + "~"
                avg = fmt_dur(total / calls) if calls else "-"
                share = f"{100.0 * total / denominator:5.1f}%" if denominator > 0 else "    -"
                print(f"  {name:<{LABEL_W}}{(calls or '-'):>7}{fmt_dur(total):>12}"
                      f"{avg:>12}{share:>8}")

    def _print_bucket(self, bucket: Bucket, title: str) -> None:
        width = LABEL_W + 39
        print("\n" + "=" * width)
        print(f" {title}")
        if bucket.frames:
            fps = bucket.frames / bucket.wall if bucket.wall else 0.0
            print(f" {bucket.frames} frame(s) processed in {fmt_dur(bucket.wall)}"
                  f"  ({fps:.1f} frames/s end to end)")
        else:
            print(f" wall time {fmt_dur(bucket.wall)}")
        print("=" * width)
        self._print_table(bucket, bucket.wall)

        # What the tables do NOT cover: the pipeline call and the worker helpers are the only
        # top-level stages, so whatever is left is plain I/O (writing JSON, walking folders).
        if bucket.kind == "session":
            accounted = (bucket.totals.get("pipeline.total", 0.0)
                         + sum(bucket.totals.get(k, 0.0) for k in
                               ("worker.unpack", "worker.clips", "worker.payloads",
                                "worker.download", "worker.upload")))
            rest = bucket.wall - accounted
            if rest > 0.05:
                print(f"  {'unattributed (file I/O, JSON, book-keeping)':<{LABEL_W}}"
                      f"{'-':>7}{fmt_dur(rest):>12}{'-':>12}"
                      f"{100.0 * rest / bucket.wall:5.1f}%")

    def report_last_session(self, *, title: str | None = None) -> None:
        """One table for the session that just finished -- for the online worker, which polls
        forever and so never reaches an end-of-run report."""
        if not self.sessions:
            return
        bucket = self.sessions[-1]
        self._print_bucket(bucket, title or f"RUN TIMING -- {bucket.name}")
        print(f"  {'worker uptime (since launch)':<{LABEL_W}}{fmt_dur(process_elapsed()):>12}")

    def report(self, *, title: str = "RUN TIMING") -> None:
        """The end-of-run report: start-up, every session, the batch total, the wall clock."""
        if not self._setup_closed:
            self.setup.wall = time.perf_counter() - self.created
            self._setup_closed = True

        width = LABEL_W + 39
        print("\n" + "#" * width)
        print(f"# {title}")
        print("#" * width)

        self._print_bucket(self.setup, "START-UP (before the first video)")
        for bucket in self.sessions:
            self._print_bucket(bucket, f"SESSION  {bucket.name}")

        if len(self.sessions) > 1:
            combined = Bucket("all sessions")
            for bucket in self.sessions:
                bucket.merge_into(combined)
            self._print_bucket(combined, f"ALL {len(self.sessions)} SESSIONS COMBINED")

        # -- the wall clock ------------------------------------------------- #
        total = process_elapsed()
        worker_span = time.perf_counter() - self.created
        startup = total - worker_span              # interpreter + torch/ultralytics imports
        print("\n" + "=" * width)
        print(" TOTAL RUN TIME")
        print("=" * width)
        if PROCESS_START_TS is not None and startup > 0:
            print(f"  {'python start-up + imports (torch, ultralytics)':<{LABEL_W}}"
                  f"{fmt_dur(startup):>12}")
        print(f"  {'worker start-up (imports + model loading)':<{LABEL_W}}"
              f"{fmt_dur(self.setup.wall):>12}")
        for bucket in self.sessions:
            name = f"session {bucket.name}"
            if len(name) > LABEL_W - 1:
                name = name[:LABEL_W - 2] + "~"
            print(f"  {name:<{LABEL_W}}{fmt_dur(bucket.wall):>12}")
        label = ("TOTAL (from launching the worker to now)" if PROCESS_START_TS is not None
                 else "TOTAL (from the worker's first import to now)")
        print("  " + "-" * (LABEL_W + 10))
        print(f"  {label:<{LABEL_W}}{fmt_dur(total):>12}")
        print("=" * width)


# --------------------------------------------------------------------------- #
def install(main, wc=None) -> RunTimer:
    """One-shot install for a caller that loads its models itself (models are timed only if
    install_models ran before they were loaded -- see the module docstring)."""
    timer = RunTimer()
    timer.install_models(main)
    timer.install_pipeline(main)
    if wc is not None:
        timer.install_worker(wc)
    return timer
