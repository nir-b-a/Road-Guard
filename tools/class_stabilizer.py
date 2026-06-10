"""
Temporal CLASS stabilizer for lane-line detections (no retrain).

WHY: the detector classifies each frame independently, so one physical line
strobes between classes (dashed<->solid_white, white<->yellow) as lighting and
dash-gap geometry change. The class is a property of the TRACKED line over TIME,
not of a single frame. This layer associates detections to tracks, votes the
class over a recency-weighted window, and only switches the displayed label
under hysteresis (margin + K consecutive frames) plus a confidence gate.

REUSE / DIFFERENCE vs tools/tracking_grid_diagnostic.py + tools/island_temporal_grid.py:
  - Same IoU greedy association + recency-weighted accumulation pattern
    (w_t = lambda**age, age=0 newest).
  - KEY DIFFERENCE: association here is GEOMETRY-ONLY (class-agnostic). Those
    trackers skip matches across classes; we must NOT, because the entire purpose
    is to stabilize a class that *changes* on the same track.

API:
    stab = ClassStabilizer(window=15, lam=0.85, switch_margin=0.20, switch_k=5,
                           iou_assoc=0.30, conf_gate=0.40, track_death_frames=10)
    out = stab.update([(cls_name, conf, (x1,y1,x2,y2)), ...])  # one frame
    # out -> list[StableDet(box, stable_class, conf, track_id)]
    stats = stab.flicker_stats()  # raw vs stabilized class-changes-per-track
"""
from __future__ import annotations

from dataclasses import dataclass


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@dataclass
class StableDet:
    box: tuple
    stable_class: str
    conf: float
    track_id: int


@dataclass
class _Track:
    tid: int
    box: tuple
    win: list           # newest-first list of (cls_name|None, conf); (None, 0.0) = miss
    stable_class: str
    last_conf: float = 0.0
    miss_streak: int = 0
    # hysteresis challenger state
    challenger: str | None = None
    challenger_streak: int = 0
    # flicker bookkeeping (per-track class changes)
    matched_frames: int = 0
    last_raw_class: str | None = None
    raw_changes: int = 0          # how often the per-frame argmax class flipped
    stable_changes: int = 0       # how often the stabilized label flipped


class ClassStabilizer:
    def __init__(self, window: int = 15, lam: float = 0.85, switch_margin: float = 0.20,
                 switch_k: int = 5, iou_assoc: float = 0.30, conf_gate: float = 0.40,
                 track_death_frames: int = 10):
        self.N = window
        self.lam = lam
        self.switch_margin = switch_margin
        self.switch_k = switch_k
        self.iou_assoc = iou_assoc
        self.conf_gate = conf_gate
        self.death = track_death_frames
        self.tracks: list[_Track] = []
        self._next_id = 0
        # flicker totals folded in from reaped tracks (so long-clip stats stay correct)
        self._closed_raw = 0
        self._closed_stable = 0
        self._closed_tracks = 0

    def _vote(self, tr: _Track) -> dict:
        """Recency-weighted class vote over the track's window. age=0 is newest."""
        v: dict = {}
        for age, (cls, cf) in enumerate(tr.win):
            if cls is None:
                continue
            v[cls] = v.get(cls, 0.0) + (self.lam ** age) * cf
        return v

    def update(self, dets) -> list[StableDet]:
        """dets: iterable of (cls_name, conf, box_xyxy) for ONE frame."""
        live = [(c, float(cf), tuple(b)) for (c, cf, b) in dets]
        used: set[int] = set()

        # Greedy IoU association, GEOMETRY-ONLY (class-agnostic).
        for tr in self.tracks:
            best_j, best = -1, self.iou_assoc
            for j, (c, cf, b) in enumerate(live):
                if j in used:
                    continue
                v = iou(tr.box, b)
                if v >= best:
                    best, best_j = v, j
            if best_j >= 0:
                c, cf, b = live[best_j]
                used.add(best_j)
                tr.box = b
                tr.win.insert(0, (c, cf))
                tr.last_conf = cf
                tr.miss_streak = 0
                tr.matched_frames += 1
                if tr.last_raw_class is not None and c != tr.last_raw_class:
                    tr.raw_changes += 1
                tr.last_raw_class = c
            else:
                tr.win.insert(0, (None, 0.0))
                tr.miss_streak += 1
            tr.win = tr.win[:self.N]

        # Unmatched detections -> new tracks.
        for j, (c, cf, b) in enumerate(live):
            if j not in used:
                self.tracks.append(_Track(
                    tid=self._next_id, box=b, win=[(c, cf)], stable_class=c,
                    last_conf=cf, matched_frames=1, last_raw_class=c))
                self._next_id += 1

        # Vote + hysteresis switch, then emit alive tracks.
        out: list[StableDet] = []
        for tr in self.tracks:
            vote = self._vote(tr)
            if vote:
                total = sum(vote.values()) or 1.0
                leader = max(vote, key=vote.get)
                if leader == tr.stable_class:
                    tr.challenger, tr.challenger_streak = None, 0
                else:
                    lead_v = vote[leader]
                    cur_v = vote.get(tr.stable_class, 0.0)
                    margin_ok = (lead_v - cur_v) >= self.switch_margin * total
                    # CONF GATE: a flip needs recent evidence for the leader above conf_gate.
                    conf_ok = any(cls == leader and cf >= self.conf_gate for (cls, cf) in tr.win)
                    if margin_ok and conf_ok:
                        if tr.challenger == leader:
                            tr.challenger_streak += 1
                        else:
                            tr.challenger, tr.challenger_streak = leader, 1
                        if tr.challenger_streak >= self.switch_k:
                            tr.stable_class = leader
                            tr.stable_changes += 1
                            tr.challenger, tr.challenger_streak = None, 0
                    else:
                        tr.challenger, tr.challenger_streak = None, 0
            if tr.miss_streak == 0:
                out.append(StableDet(tr.box, tr.stable_class, tr.last_conf, tr.tid))

        # Reap dead tracks, folding their flicker counts into the running totals.
        alive: list[_Track] = []
        for tr in self.tracks:
            dead = tr.miss_streak > self.death or all(c is None for c, _ in tr.win)
            if dead:
                if tr.matched_frames >= 2:
                    self._closed_raw += tr.raw_changes
                    self._closed_stable += tr.stable_changes
                    self._closed_tracks += 1
            else:
                alive.append(tr)
        self.tracks = alive
        return out

    def flicker_stats(self) -> dict:
        """Avg class-changes-per-track: raw (per-frame argmax) vs stabilized.
        Counts only tracks seen in >=2 frames (a 1-frame track cannot flicker)."""
        raw, stable, n = self._closed_raw, self._closed_stable, self._closed_tracks
        for tr in self.tracks:
            if tr.matched_frames >= 2:
                raw += tr.raw_changes
                stable += tr.stable_changes
                n += 1
        return {
            "n_tracks": n,
            "raw_changes": raw,
            "stable_changes": stable,
            "raw_per_track": round(raw / n, 3) if n else 0.0,
            "stable_per_track": round(stable / n, 3) if n else 0.0,
        }
