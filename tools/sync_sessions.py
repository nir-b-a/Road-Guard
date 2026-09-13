#!/usr/bin/env python3
"""
sync_sessions.py -- put TWO independently started captures on one absolute
timeline, using the GNSS clock each phone recorded, and report how well they line up.

Two phones filming the same drive start at different moments and stamp their data
on their own monotonic clock (elapsedRealtimeNanos), whose zero is device boot.
The recorder writes session_meta.json with `clock_epoch_unix_ms` -- the UTC instant
that clock read zero, taken as the median of (satellite UTC - monotonic) over every
GNSS fix. With it, EVERY row of EVERY stream becomes absolute:

    utc_ms = clock_epoch_unix_ms + timestamp_ns / 1e6

and no further per-file offset is needed. This script verifies that anchor on both
captures, prints the overlap, and writes:

    sync_report.json   the two anchors, spans, overlap, the start offset
    frame_map.csv      frame_a <-> frame_b, matched by NEAREST UTC

frame_map.csv is the honest way to pair video frames: a single constant offset
drifts, because two phones never run at exactly the same frame rate (29.97 vs
30.00 fps is 1.8 s of drift over half an hour). Every frame carries its own
timestamp, so each is matched on its own.

Run from the malshinon/ directory:
    python tools/sync_sessions.py SESSION_A SESSION_B [--out DIR] [--no-frame-map]
"""

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # malshinon/
sys.path.insert(0, ROOT)

from speed_estimation.session_clock import (  # noqa: E402
    SessionClockError, load_frame_times, load_session_clock, utc_ms_to_iso,
)

# Streams that carry a timestamp_ns column; each is spanned for the report.
STREAMS = ("gyro.csv", "gravity.csv", "linacc.csv", "gps.csv")


def stream_span_ns(path: str) -> tuple[int, int, int] | None:
    """(first_ts, last_ts, rows) of a timestamp_ns stream, or None if unusable."""
    if not os.path.exists(path):
        return None
    first = last = None
    rows = 0
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header or "timestamp_ns" not in header:
            return None
        idx = header.index("timestamp_ns")
        for row in reader:
            try:
                ts = int(row[idx])
            except (IndexError, ValueError):
                continue
            if first is None or ts < first:
                first = ts
            if last is None or ts > last:
                last = ts
            rows += 1
    return None if first is None else (first, last, rows)


def video_of(session_dir: str) -> str:
    """The capture's source .mp4 (pipeline outputs excluded), for the ffmpeg hint."""
    import glob
    cands = [p for p in glob.glob(os.path.join(session_dir, "*.mp4"))
             if not p.endswith("_annotated.mp4")]
    return cands[0] if cands else os.path.join(session_dir, "<video>.mp4")


def fps_from(frame_times: list[tuple[int, int]]) -> float | None:
    """Effective fps = 1e9 / median inter-frame gap. The nominal fps in the
    container is rounded; this is what the sensor clock actually saw."""
    if len(frame_times) < 3:
        return None
    gaps = sorted(b[1] - a[1] for a, b in zip(frame_times, frame_times[1:]) if b[1] > a[1])
    if not gaps:
        return None
    return 1e9 / gaps[len(gaps) // 2]


def summarize(session_dir: str) -> dict:
    """Anchor + every stream's absolute UTC span for one capture."""
    clock = load_session_clock(session_dir)
    frames_csv = os.path.join(session_dir, "frames.csv")
    if not os.path.exists(frames_csv):
        raise SessionClockError(f"{session_dir}: no frames.csv (not a capture folder?)")
    ft = load_frame_times(frames_csv)
    if not ft:
        raise SessionClockError(f"{session_dir}: frames.csv has no usable rows")

    streams = {}
    for name in STREAMS:
        span = stream_span_ns(os.path.join(session_dir, name))
        if span is None:
            continue
        lo, hi, rows = span
        streams[name] = {"rows": rows,
                         "start_utc_ms": clock.to_utc_ms(lo), "end_utc_ms": clock.to_utc_ms(hi),
                         "start_utc": utc_ms_to_iso(clock.to_utc_ms(lo)),
                         "end_utc": utc_ms_to_iso(clock.to_utc_ms(hi))}

    v_start, v_end = clock.to_utc_ms(ft[0][1]), clock.to_utc_ms(ft[-1][1])
    return {
        "session_dir": session_dir,
        "clock": {
            "epoch_unix_ms": clock.epoch_ms, "time_source": clock.time_source,
            "gnss_fix_count": clock.fix_count, "gnss_epoch_mad_ms": clock.mad_ms,
            "system_clock_error_ms": clock.system_clock_error_ms,
            "anchor_file": clock.anchor_file,
            "warnings": clock.warnings(),
        },
        "video": {
            "frames": len(ft), "fps": fps_from(ft),
            "start_utc_ms": v_start, "end_utc_ms": v_end,
            "start_utc": utc_ms_to_iso(v_start), "end_utc": utc_ms_to_iso(v_end),
            "duration_s": (v_end - v_start) / 1000.0,
        },
        "streams": streams,
        "_clock_obj": clock, "_frame_times": ft,
    }


def build_frame_map(a: dict, b: dict) -> list[tuple[int, float, int, float]]:
    """(frame_a, utc_a, frame_b, dt_ms) for every A frame inside B's span, matching
    each to B's NEAREST frame in UTC. |dt| <= half a frame interval when both
    cameras ran normally; a growing |dt| means one camera dropped frames."""
    import bisect
    ca, cb = a["_clock_obj"], b["_clock_obj"]
    tb = [cb.to_utc_ms(ts) for _, ts in b["_frame_times"]]
    fb = [fr for fr, _ in b["_frame_times"]]
    out = []
    for fr_a, ts_a in a["_frame_times"]:
        t = ca.to_utc_ms(ts_a)
        if not tb or t < tb[0] or t > tb[-1]:
            continue
        i = bisect.bisect_left(tb, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(tb)]
        j = min(cands, key=lambda k: abs(tb[k] - t))
        out.append((fr_a, t, fb[j], tb[j] - t))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Align two captures on absolute UTC via their recorded GNSS clock.")
    ap.add_argument("session_a", help="capture folder of phone A (the chase / camera car)")
    ap.add_argument("session_b", help="capture folder of phone B (the lead car)")
    ap.add_argument("--out", default=".", help="where to write sync_report.json / frame_map.csv")
    ap.add_argument("--no-frame-map", action="store_true", help="skip writing frame_map.csv")
    args = ap.parse_args()

    try:
        a = summarize(os.path.abspath(args.session_a))
        b = summarize(os.path.abspath(args.session_b))
    except SessionClockError as e:
        sys.exit(f"[sync] {e}")

    os.makedirs(args.out, exist_ok=True)

    # ── anchors ──────────────────────────────────────────────────────────────
    print("\n=== clock anchors ===")
    for tag, s in (("A", a), ("B", b)):
        c = s["clock"]
        mad = f"{c['gnss_epoch_mad_ms']:.0f} ms" if c["gnss_epoch_mad_ms"] is not None else "n/a"
        err = (f"{c['system_clock_error_ms']:+.0f} ms"
               if c["system_clock_error_ms"] is not None else "n/a")
        print(f"  {tag} {os.path.basename(s['session_dir'])}")
        print(f"      time_source={c['time_source']}  fixes={c['gnss_fix_count']}  "
              f"anchor MAD={mad}  phone clock error={err}  (from {c['anchor_file']})")
        for w in c["warnings"]:
            print(f"      !! {w}")

    # ── spans / overlap ──────────────────────────────────────────────────────
    print("\n=== video spans (absolute UTC) ===")
    for tag, s in (("A", a), ("B", b)):
        v = s["video"]
        fps = f"{v['fps']:.3f}" if v["fps"] else "n/a"
        print(f"  {tag} {v['start_utc']} -> {v['end_utc']}  "
              f"({v['duration_s']:.1f} s, {v['frames']} frames, {fps} fps)")

    start_offset_s = (b["video"]["start_utc_ms"] - a["video"]["start_utc_ms"]) / 1000.0
    ov_start = max(a["video"]["start_utc_ms"], b["video"]["start_utc_ms"])
    ov_end = min(a["video"]["end_utc_ms"], b["video"]["end_utc_ms"])
    overlap_s = (ov_end - ov_start) / 1000.0

    print(f"\n  B started {abs(start_offset_s):.3f} s "
          f"{'AFTER' if start_offset_s >= 0 else 'BEFORE'} A")
    if overlap_s <= 0:
        print("  !! the two recordings DO NOT OVERLAP in time -- nothing to compare")
    else:
        print(f"  overlap: {utc_ms_to_iso(ov_start)} -> {utc_ms_to_iso(ov_end)}  ({overlap_s:.1f} s)")

    print("\n=== sensor streams (absolute UTC) ===")
    for tag, s in (("A", a), ("B", b)):
        for name, st in s["streams"].items():
            print(f"  {tag} {name:12s} {st['rows']:7d} rows  "
                  f"{st['start_utc']} -> {st['end_utc']}")

    # ── frame map ────────────────────────────────────────────────────────────
    fmap_path = None
    if not args.no_frame_map and overlap_s > 0:
        fmap = build_frame_map(a, b)
        fmap_path = os.path.join(args.out, "frame_map.csv")
        with open(fmap_path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["frame_a", "utc_ms_a", "utc_a", "frame_b", "match_error_ms"])
            for fr_a, t_a, fr_b, dt in fmap:
                wr.writerow([fr_a, f"{t_a:.1f}", utc_ms_to_iso(t_a), fr_b, f"{dt:.1f}"])
        if fmap:
            worst = max(abs(d) for *_, d in fmap)
            print(f"\n[sync] frame_map.csv: {len(fmap)} paired frames, "
                  f"worst match error {worst:.0f} ms -> {fmap_path}")

    # ── report ───────────────────────────────────────────────────────────────
    report = {
        "session_a": {k: v for k, v in a.items() if not k.startswith("_")},
        "session_b": {k: v for k, v in b.items() if not k.startswith("_")},
        "start_offset_s": start_offset_s,
        "overlap_start_utc_ms": ov_start, "overlap_end_utc_ms": ov_end,
        "overlap_start_utc": utc_ms_to_iso(ov_start), "overlap_end_utc": utc_ms_to_iso(ov_end),
        "overlap_s": overlap_s,
        "formula": "utc_ms = clock_epoch_unix_ms + timestamp_ns / 1e6",
    }
    rep_path = os.path.join(args.out, "sync_report.json")
    with open(rep_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[sync] report -> {rep_path}")

    # A constant offset is fine for eyeballing a short side-by-side; it is NOT how
    # the data is compared (that is per-frame, above -- a constant offset drifts
    # because the two cameras never run at exactly the same rate).
    if overlap_s > 0:
        first, second = (a, b) if start_offset_s >= 0 else (b, a)
        print(f"\n  side-by-side playback (trims the head of the one that started first):\n"
              f"    ffmpeg -ss {abs(start_offset_s):.3f} -i {video_of(first['session_dir'])} "
              f"-i {video_of(second['session_dir'])} "
              f'-filter_complex "[0:v][1:v]hstack=inputs=2" -c:v libx264 -crf 18 sync.mp4')


if __name__ == "__main__":
    main()
