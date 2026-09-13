"""Model-free self-test for crossing2.

    python -m line_crossing.crossing2.selftest       (from the malshinon/ root)

Builds synthetic clips - perspective lane lines plus scripted bounding boxes - and
asserts the event counts. No YOLO, no CLRerNet, no video, no GPU: it runs in about a
second, so the geometry and the event logic can be re-validated after every threshold
change without waiting on a real clip.

Each vehicle gets its own scene: the Stage 1 occlusion gate is real, and two boxes
placed carelessly in one frame will (correctly) invalidate each other's samples.

Cases
-----
  A1 vehicle crosses a solid line and stays          -> 1 crossing
  A2 vehicle stays in lane, well clear               -> nothing
  B  vehicle leans a third of its body over, returns -> 1 strafe, 0 crossings
  C1 single-frame teleport across the line           -> nothing  (the legacy
                                                        detector latches forever here)
  C2 fast jitter centred on the line                 -> nothing  (also covers a
                                                        vehicle hugging the line)
  C3 distant vehicle sitting on the line             -> nothing  (near-field gate)
  C4 double solid line crossed once                  -> 1 crossing, not 2
  C5 slow deliberate weave, out and back             -> 2 crossings
"""
from __future__ import annotations

import math
import sys

from ..lane_types import Lane
from .detector import detect_crossings, format_events
from .types import FrameObservation, VehicleBox

W, H, FPS, N = 1920, 1080, 30.0, 150
X_VP, Y_VP, Y_BOT = 950.0, 450.0, 1080.0
DASHED_X, SOLID_X = 600.0, 1300.0


def _line_x(x_bottom: float, y: float) -> float:
    return X_VP + (x_bottom - X_VP) * (y - Y_VP) / (Y_BOT - Y_VP)


def _points(x_bottom: float):
    return [(int(round(_line_x(x_bottom, y))), int(y)) for y in range(1079, 459, -20)]


def _box(cx: float, y2: float, w: float = 200, h: float = 150):
    return (int(cx - w / 2), int(y2 - h), int(cx + w / 2), int(y2))


def _lanes(f: int, extra=()):
    ln = [Lane(points=_points(DASHED_X), lane_type="dashed", score=0.9),
          Lane(points=_points(SOLID_X), lane_type="solid_white", score=0.9)]
    if 55 <= f < 58:                 # lane-type flicker: the track vote must absorb it
        ln[1].lane_type = "dashed"
    if f in (60, 61):                # dropout: the offline stitch must bridge it
        ln = [ln[0]]
    return ln + [Lane(points=_points(x), lane_type=t, score=0.9) for t, x in extra]


def _run(name: str, cx_of, expect: tuple[int, int], *, w=200, h=150, y2=900, extra=()):
    obs = [FrameObservation(f, _lanes(f, extra),
                            [VehicleBox(7, _box(cx_of(f), y2, w, h), "car")])
           for f in range(N)]
    res = detect_crossings(obs, W, H, FPS)
    got = (sum(1 for e in res.events if e.kind == "crossing"),
           sum(1 for e in res.events if e.kind == "strafe"))
    ok = got == expect
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: {got[0]} crossing, {got[1]} strafe"
          f"   (expected {expect[0]}, {expect[1]})")
    if not ok or "-v" in sys.argv:
        tags = " ".join(f"L{t.track_id}:{t.lane_type}"
                        f"{'/SUPP' if t.suppressed else ''}"
                        f"{'/DBL' if t.is_double else ''}"
                        f"{'/GORE' if t.is_gore else ''}" for t in res.tracks)
        print(f"       tracks: {tags}")
        if res.events:
            print(format_events(res.events))
    return ok


def _weave(f: int) -> float:
    if f < 35:
        return 1050.0
    if f < 55:
        return 1050.0 + (1380.0 - 1050.0) * (f - 35) / 20.0
    if f < 95:
        return 1380.0
    if f < 115:
        return 1380.0 - (1380.0 - 1050.0) * (f - 95) / 20.0
    return 1050.0


def _lean(f: int) -> float:
    if f < 50:
        return 1120.0
    if f < 65:
        return 1120.0 + (1275.0 - 1120.0) * (f - 50) / 15.0
    if f < 100:
        return 1275.0
    if f < 115:
        return 1275.0 - (1275.0 - 1120.0) * (f - 100) / 15.0
    return 1120.0


def main() -> int:
    print(f"solid line at y=900: x={_line_x(SOLID_X, 900):.0f}, "
          f"lane width {_line_x(SOLID_X, 900) - _line_x(DASHED_X, 900):.0f} px\n")
    r = [
        _run("A1 crosses the solid line and stays",
             lambda f: 1050.0 if f < 40 else (1380.0 if f > 70 else
                                              1050.0 + 330.0 * (f - 40) / 30.0), (1, 0)),
        _run("A2 stays in lane", lambda f: 820.0, (0, 0), y2=860),
        _run("B  leans over the line and returns", _lean, (0, 1), w=220, h=160, y2=1000),
        _run("C1 single-frame teleport across",
             lambda f: 1400.0 if f == 75 else 1050.0, (0, 0)),
        _run("C2 jitter / hugging the line",
             lambda f: 1200.0 + 150.0 * math.sin(2 * math.pi * f / 6.0), (0, 0)),
        _run("C3 distant vehicle on the line", lambda f: 1256.0, (0, 0), w=40, h=30, y2=700),
        _run("C4 double solid line, crossed once",
             lambda f: _weave(f) if f < 95 else 1380.0, (1, 0),
             extra=(("solid_white", 1332.0),)),
        _run("C5 slow weave out and back", _weave, (2, 0)),
    ]
    print("\nRESULT:", "ALL OK" if all(r) else f"{r.count(False)} FAILED")
    return 0 if all(r) else 1


if __name__ == "__main__":
    raise SystemExit(main())
