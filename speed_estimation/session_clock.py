"""
session_clock -- anchor one capture's monotonic sensor clock to ABSOLUTE UTC, so
two independently started recordings can be compared sample by sample.

Every stream in a capture (frames.csv, gyro.csv, gravity.csv, linacc.csv, gps.csv)
is stamped on ONE monotonic clock -- Android's elapsedRealtimeNanos. That clock
starts at device boot, so its zero is arbitrary and its values are meaningless
ACROSS devices. Two phones that were not started at the same moment can only be
compared after each capture is mapped onto a shared absolute timeline:

    utc_ms = clock_epoch_unix_ms + timestamp_ns / 1e6

`clock_epoch_unix_ms` is that mapping's ONLY parameter: the UTC instant at which
the phone's monotonic clock read zero. The recorder writes it to session_meta.json
as the MEDIAN of (Location.getTime() - fix.elapsedRealtimeNanos / 1e6) over every
GNSS fix in the drive. Satellite time is derived from the constellation's atomic
clocks, so two phones that both had a sky view agree on it to within milliseconds
-- unlike their own system clocks, which drift independently by 0.1-2 s and cannot
be cross-checked after the fact.

Deliberately stdlib-only: the diagnostic tools that import this must run without
torch / ultralytics / numpy installed.
"""

import csv
import json
import os
from dataclasses import dataclass

MPS_TO_KMH = 3.6

# A GNSS anchor should be repeatable to a few tens of ms across the drive. Beyond
# this the fixes disagree badly enough that sub-second cross-phone sync is not
# credible and the caller is warned.
MAD_WARN_MS = 250.0
# Fewer fixes than this and the median has little to average over.
MIN_FIXES_WARN = 10
# session_meta.json's stored anchor vs. the one recomputed from gps.csv. They are
# the same statistic over the same data, so they only diverge if the CSV was cut
# (a subset median) or the files come from different drives.
ANCHOR_DISAGREE_WARN_MS = 500.0


class SessionClockError(Exception):
    """No usable UTC anchor for a capture folder."""


@dataclass
class GpsFix:
    """One row of gps.csv. `unix_time_ms` is 0 on captures recorded before the
    GNSS-time columns existed."""
    timestamp_ns: int
    lat: float
    lon: float
    speed_mps: float
    bearing_deg: float
    accuracy_m: float
    unix_time_ms: int = 0
    provider: str = ""


@dataclass
class SessionClock:
    """The monotonic-clock -> UTC mapping for ONE capture folder."""
    session_dir: str
    epoch_ms: float                     # UTC ms at which timestamp_ns == 0
    time_source: str                    # "gnss" | "system" | "gnss(gps.csv)"
    fix_count: int                      # GNSS fixes the anchor was averaged over
    mad_ms: float | None                # median abs deviation of those fixes
    recording_start_unix_ms: float | None
    system_clock_error_ms: float | None  # this phone's own clock error vs GNSS
    anchor_file: str                    # where epoch_ms came from

    def to_utc_ms(self, timestamp_ns: int | float) -> float:
        """Sensor-clock nanoseconds -> absolute UTC milliseconds."""
        return self.epoch_ms + timestamp_ns / 1e6

    def to_timestamp_ns(self, utc_ms: float) -> int:
        """Absolute UTC milliseconds -> this capture's sensor clock."""
        return int(round((utc_ms - self.epoch_ms) * 1e6))

    @property
    def is_gnss(self) -> bool:
        return self.time_source.startswith("gnss")

    def warnings(self) -> list[str]:
        """Everything that makes this anchor less trustworthy than it looks.
        Empty list == a clean, satellite-derived anchor."""
        out: list[str] = []
        if not self.is_gnss:
            out.append(
                f"time_source={self.time_source!r} -- no satellite time in this capture, "
                "the anchor is this phone's own clock (expect 0.1-2 s of unmeasurable error)")
        if self.mad_ms is not None and self.mad_ms > MAD_WARN_MS:
            out.append(f"gnss anchor spread is high (MAD {self.mad_ms:.0f} ms > {MAD_WARN_MS:.0f} ms)")
        if self.is_gnss and self.fix_count < MIN_FIXES_WARN:
            out.append(f"only {self.fix_count} GNSS fix(es) behind the anchor "
                       f"(< {MIN_FIXES_WARN}); poor sky view?")
        if self.is_gnss and self.fix_count > 50 and self.system_clock_error_ms == 0:
            out.append("system_clock_error_ms is exactly 0 over many fixes -- Location.getTime() "
                       "was probably echoing the system clock (network-derived fix), NOT satellite time")
        return out

    def describe(self) -> str:
        start = (f"{utc_ms_to_iso(self.recording_start_unix_ms)}"
                 if self.recording_start_unix_ms else "unknown")
        mad = f"{self.mad_ms:.0f} ms" if self.mad_ms is not None else "n/a"
        err = (f"{self.system_clock_error_ms:+.0f} ms"
               if self.system_clock_error_ms is not None else "n/a")
        return (f"{os.path.basename(self.session_dir.rstrip(os.sep))}: "
                f"source={self.time_source} fixes={self.fix_count} anchor_mad={mad} "
                f"phone_clock_error={err} start={start}")


# --------------------------------------------------------------------------- #
# readers
# --------------------------------------------------------------------------- #
def load_frame_times(frames_csv: str) -> list[tuple[int, int]]:
    """[(frame, timestamp_ns), ...] sorted by frame index.

    Kept separate from ego_yaw.load_frame_timestamps (which returns a dict and
    drags in numpy/scipy) so the sync tools stay import-light.
    """
    rows: list[tuple[int, int]] = []
    with open(frames_csv, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((int(r["frame"]), int(r["timestamp_ns"])))
            except (KeyError, ValueError, TypeError):
                continue
    rows.sort()
    return rows


def load_gps(gps_csv: str) -> list[GpsFix]:
    """Every row of gps.csv, sorted by sensor time. Columns added later
    (unix_time_ms, provider) default to 0/"" so old captures still load."""
    fixes: list[GpsFix] = []
    with open(gps_csv, newline="") as f:
        for r in csv.DictReader(f):
            try:
                fixes.append(GpsFix(
                    timestamp_ns=int(r["timestamp_ns"]),
                    lat=float(r["lat"]),
                    lon=float(r["lon"]),
                    speed_mps=float(r["speed_mps"]),
                    bearing_deg=float(r.get("bearing_deg") or 0.0),
                    accuracy_m=float(r.get("accuracy_m") or 0.0),
                    unix_time_ms=int(float(r.get("unix_time_ms") or 0)),
                    provider=(r.get("provider") or "").strip(),
                ))
            except (KeyError, ValueError, TypeError):
                continue
    fixes.sort(key=lambda x: x.timestamp_ns)
    return fixes


def gnss_anchor_from_fixes(fixes: list[GpsFix]) -> tuple[float, float, int] | None:
    """(epoch_ms, mad_ms, n) from the fixes that carry satellite time, or None.

    The same statistic the recorder computes on-device: the median of
    (utc - monotonic) over every timed fix. Hundreds of independent estimates of
    one constant, so a few bad fixes cannot move it.
    """
    offsets = sorted(f.unix_time_ms - f.timestamp_ns / 1e6
                     for f in fixes if f.unix_time_ms > 0)
    if not offsets:
        return None
    epoch = offsets[len(offsets) // 2]
    dev = sorted(abs(o - epoch) for o in offsets)
    return epoch, dev[len(dev) // 2], len(offsets)


def load_session_clock(session_dir: str, *, quiet: bool = False) -> SessionClock:
    """Build the UTC anchor for a capture folder.

    Prefers session_meta.json's `clock_epoch_unix_ms` -- it was computed on-device
    over the WHOLE drive, which still holds for a cut clip (cut_clip.py keeps
    timestamp_ns absolute and copies the meta verbatim), whereas re-deriving from a
    cut gps.csv would only average the fixes that survived the cut.

    Falls back to deriving the anchor from gps.csv when the meta file is missing.
    Raises SessionClockError when neither is available -- i.e. the capture predates
    the GNSS-time recording and CANNOT be placed on an absolute timeline.
    """
    session_dir = os.path.abspath(session_dir)
    meta_path = os.path.join(session_dir, "session_meta.json")
    gps_path = os.path.join(session_dir, "gps.csv")

    meta = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as e:
            if not quiet:
                print(f"[clock] {meta_path} unreadable ({e}); falling back to gps.csv")

    derived = None
    if os.path.exists(gps_path):
        derived = gnss_anchor_from_fixes(load_gps(gps_path))

    if meta and meta.get("clock_epoch_unix_ms") is not None:
        epoch = float(meta["clock_epoch_unix_ms"])
        clock = SessionClock(
            session_dir=session_dir,
            epoch_ms=epoch,
            time_source=str(meta.get("time_source") or "system"),
            fix_count=int(meta.get("gnss_fix_count") or 0),
            mad_ms=(float(meta["gnss_epoch_mad_ms"])
                    if meta.get("gnss_epoch_mad_ms") is not None else None),
            recording_start_unix_ms=(float(meta["recording_start_unix_ms"])
                                     if meta.get("recording_start_unix_ms") is not None else None),
            system_clock_error_ms=(float(meta["system_clock_error_ms"])
                                   if meta.get("system_clock_error_ms") is not None else None),
            anchor_file="session_meta.json",
        )
        if derived and not quiet:
            gap = abs(derived[0] - epoch)
            if gap > ANCHOR_DISAGREE_WARN_MS:
                print(f"[clock] WARNING {os.path.basename(session_dir)}: session_meta.json anchor and "
                      f"the one recomputed from gps.csv differ by {gap:.0f} ms -- are these files "
                      "from the same drive?")
        return clock

    if derived:
        epoch, mad, n = derived
        if not quiet:
            print(f"[clock] {os.path.basename(session_dir)}: no session_meta.json; "
                  f"anchor derived from gps.csv ({n} timed fixes)")
        return SessionClock(
            session_dir=session_dir, epoch_ms=epoch, time_source="gnss(gps.csv)",
            fix_count=n, mad_ms=mad, recording_start_unix_ms=None,
            system_clock_error_ms=None, anchor_file="gps.csv",
        )

    raise SessionClockError(
        f"{session_dir}: no UTC anchor. Needs session_meta.json (clock_epoch_unix_ms) or a "
        "gps.csv with a non-zero unix_time_ms column. Captures recorded before the GNSS-clock "
        "build have neither and cannot be placed on an absolute timeline.")


def utc_ms_to_iso(utc_ms: float | None) -> str:
    """UTC milliseconds -> '2026-08-20T09:15:30.118Z' (empty string for None)."""
    if utc_ms is None:
        return ""
    import datetime
    dt = datetime.datetime.fromtimestamp(utc_ms / 1000.0, datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"
