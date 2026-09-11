"""
worker_stop.py -- stop worker.py cleanly, from any terminal, instead of Ctrl+C.

    python worker.py --stop       finish the drive in progress (pipeline AND its evidence upload),
                                  then exit
    python worker.py --stop-now   abandon the drive in progress and hand it straight back to the
                                  server's queue (POST /api/internal/drive/<id>/release), let the
                                  evidence uploads already underway finish, then exit

Inside Docker:  docker exec roadguard-worker-gpu python worker.py --stop
                docker stop roadguard-worker-gpu        (SIGTERM = --stop-now)

Both commands only write a one-word request file, STOP_FILE (default <WORK_DIR>/worker.stop), which
the running worker checks every second. Every worker sharing that WORK_DIR obeys it, and a worker
clears a leftover file when it starts. An empty file means --stop.

Signals mean "stop now" too: Ctrl+C, Ctrl+Break and SIGTERM (what `docker stop` sends). A SECOND
Ctrl+C forces the exit even while evidence is still uploading -- the one unsafe way out.

Why "stop now" is safe while the pipeline runs: until its ship tail, a drive has no side effects
(violations are POSTed and the drive marked complete only after processing), and the server keeps
the raw R2 files until the drive is 'processed'. Aborting it loses GPU time and nothing else; the
drive is simply processed again. The ship tail is NOT safe to cut (half its violations posted ->
duplicates on the rerun), so every stop level waits for it.

Ctrl+C typed into the console is also delivered to the worker's child processes (the ffmpeg clip
encoder); --stop / --stop-now never touch the children.
"""
import _thread
import contextlib
import os
import signal
import threading
import time

GRACEFUL = "graceful"
NOW = "now"
_RANK = {None: 0, GRACEFUL: 1, NOW: 2}

_MESSAGES = {
    GRACEFUL: "[worker] stop requested: finishing the current drive (if any), then exiting",
    NOW: "[worker] stop-now requested: the current drive (if any) goes back to the server's queue; "
         "evidence uploads already underway are finished, then the worker exits",
}


def read_request(stop_file):
    """The stop level written to `stop_file`, or None when there is no (readable) request."""
    try:
        with open(stop_file, "r", encoding="utf-8") as fh:
            text = fh.read().strip().lower()
    except OSError:                        # absent, or mid-replace on Windows -> next poll retries
        return None
    return NOW if text == NOW else GRACEFUL


def request_stop(stop_file, level):
    """Ask every worker watching `stop_file` to stop. Never downgrades a pending stop-now.
    -> the level actually written."""
    if _RANK[read_request(stop_file)] > _RANK[level]:
        level = NOW
    os.makedirs(os.path.dirname(os.path.abspath(stop_file)), exist_ok=True)
    tmp = stop_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(level + "\n")
    os.replace(tmp, stop_file)             # atomic: a worker never reads a half-written request
    return level


class StopControl:
    """One worker process's stop state: fed by the request file and by signals, read by the loop.

    Each attribute below has exactly one writer, so there is no lock -- and a signal handler must
    never wait on one: the watcher thread owns `_file_level`, the signal handler `_signal_level`."""

    def __init__(self, stop_file, *, poll_seconds=1.0, log=print):
        self.stop_file = stop_file
        self.poll_seconds = poll_seconds
        self._log = log
        self._file_level = None
        self._signal_level = None
        self._signals = 0
        self._in_drive = False        # main thread is inside drive(): a stop-now aborts the pipeline
        self._nudged = False          # the watcher interrupted the main thread itself (no keypress)
        self._announced = None

    @property
    def level(self):
        a, b = self._file_level, self._signal_level
        return a if _RANK[a] >= _RANK[b] else b

    @property
    def requested(self):
        return self.level is not None

    @property
    def now(self):
        return self.level == NOW

    def install(self):
        """Clear a leftover request, take over Ctrl+C / SIGTERM, start watching the file. Main thread only."""
        try:
            os.remove(self.stop_file)
        except FileNotFoundError:
            pass
        except OSError as e:
            self._log(f"[worker] could not clear an old stop request {self.stop_file} ({e}) -- "
                      f"this worker may stop right away")
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):          # SIGBREAK = Ctrl+Break (Windows)
            sig = getattr(signal, name, None)
            if sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, self._on_signal)
        threading.Thread(target=self._watch, name="stop-watch", daemon=True).start()
        return self

    def idle(self, seconds):
        """Sleep up to `seconds`, returning as soon as a stop is requested."""
        deadline = time.monotonic() + seconds
        while not self.requested:
            left = deadline - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(0.5, left))

    @contextlib.contextmanager
    def drive(self):
        """Wrap one claimed drive's pipeline: a stop-now raises KeyboardInterrupt inside it."""
        self._in_drive = True
        try:
            if self.now:
                raise KeyboardInterrupt
            yield
        finally:
            self._in_drive = False

    def _on_signal(self, signum, frame):
        if self._nudged:                   # the watcher's own interrupt_main(), not a keypress
            self._nudged = False
            if self._in_drive:
                raise KeyboardInterrupt
            return
        self._signals += 1
        self._signal_level = NOW
        # Mid-pipeline: abort it now. Anywhere else (idle, claiming, uploading evidence) the loop
        # stops at the next safe point on its own -- unless this is the second signal: force it.
        if self._in_drive or self._signals > 1:
            raise KeyboardInterrupt

    def _watch(self):
        while True:
            level = read_request(self.stop_file)
            if _RANK[level] > _RANK[self._file_level]:
                self._file_level = level
            current = self.level
            if current != self._announced:
                self._announced = current
                self._log(_MESSAGES[current])
            # A stop-now from the file must break into the pipeline the way Ctrl+C does. A signal
            # already raised in the main thread on its own, so only nudge for a file request.
            if (self._file_level == NOW and self._signal_level is None
                    and self._in_drive and not self._nudged):
                self._nudged = True
                _thread.interrupt_main()
            time.sleep(self.poll_seconds)
