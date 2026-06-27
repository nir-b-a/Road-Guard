"""
Module F -- Cross-Track Re-ID / temporal plate memory.

BoT-SORT's track_buffer (default 30 frames) expires when a car is occluded, leaves the frame, or
changes appearance, so the SAME physical vehicle can re-appear under a NEW track id. The 3-5 frame
ego-compensated handshake (joiner.py) cannot bridge a multi-second gap. This module links tracks
across LONG gaps by APPEARANCE (an injected embedder) plus temporal plausibility, then propagates
the best plate across each linked group -- so a violation committed under a plate-less re-acquired
id still carries the plate read earlier under the original id.

HITL recall-first ([[hitl-business-model]]): thresholds lean toward LINKING. A wrong link is a
2-second human reject; a MISSED link silently loses a violator's plate. Inherited plates are tagged
plate_source="reid_inherited:<tid>" (never silently withheld) and carry both tracks for the
reviewer to confirm.

HARD GUARD: tracks that OVERLAP IN TIME are different physical objects (two boxes on screen at
once) and are never linked -- this is what keeps simultaneously-alive ids apart (e.g. on
0fr98nSc3CA, ID 1 and ID 6 coexist frames 256-445).

The embedder is INJECTED (crop -> 1-D vector), so this core is model-free and CPU-testable.
Cheap default = reid_embedders.HistogramEmbedder; production = reid_embedders.OSNetEmbedder.
"""
from __future__ import annotations

import math

# Recall-first defaults (loose: prefer linking). Tune against gt_links once labelled.
DEFAULT_SIM_THRESHOLD = 0.55      # cosine similarity above which two crops are "the same vehicle"
DEFAULT_MAX_GAP_SEC = 20.0        # re-acquisition allowed up to this many seconds after a track died


# --------------------------------------------------------------------------- #
# small math + union-find
# --------------------------------------------------------------------------- #
def cosine_sim(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return (num / (na * nb)) if na > 0 and nb > 0 else 0.0


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)   # keep the smaller (older) id as canonical


def spans_overlap(a, b) -> bool:
    """True if frame spans (first, last) intersect (>=1 shared frame) -> different objects."""
    return a[0] <= b[1] and b[0] <= a[1]


# --------------------------------------------------------------------------- #
# linking
# --------------------------------------------------------------------------- #
def candidate_links(spans: dict, embeddings: dict, *, sim_threshold: float,
                    max_gap_frames: int) -> list:
    """All (old_tid, new_tid, sim) pairs eligible to merge: new born AFTER old died, within the
    gap window, no temporal overlap, appearance similar. Sorted by similarity (best first)."""
    tids = [t for t in spans if t in embeddings]
    pairs = []
    for new in tids:
        for old in tids:
            if old == new:
                continue
            so, sn = spans[old], spans[new]
            if spans_overlap(so, sn):                 # HARD GUARD: coexisting -> different cars
                continue
            gap = sn[0] - so[1]                        # frames between old's death and new's birth
            if gap <= 0 or gap > max_gap_frames:       # new must come strictly after old, within window
                continue
            sim = cosine_sim(embeddings[old], embeddings[new])
            if sim >= sim_threshold:
                pairs.append((old, new, sim))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def candidate_predecessors(spans: dict, target_tids, *, max_gap_frames: int,
                           exclude=None) -> set:
    """Tracks that could be the SAME car re-acquired as one of `target_tids`: died strictly before
    the target was born, within the gap window, and not time-overlapping it. Appearance is checked
    later -- this only narrows WHICH crops are worth embedding/OCR-ing (perf bound)."""
    exclude = set(exclude or [])
    preds = set()
    for t in target_tids:
        st = spans.get(t)
        if st is None:
            continue
        for o, so in spans.items():
            if o == t or o in exclude:
                continue
            gap = st[0] - so[1]
            if 0 < gap <= max_gap_frames and not spans_overlap(so, st):
                preds.add(o)
    return preds


def link_tracks(spans: dict, embeddings: dict, *, sim_threshold: float = DEFAULT_SIM_THRESHOLD,
                max_gap_frames: int = 600) -> dict:
    """Return {track_id: canonical_id}. Union-find over candidate links; the smallest (oldest)
    id in a group is the canonical id. Tracks without an embedding stay singletons."""
    uf = _UnionFind(list(spans))
    for old, new, _sim in candidate_links(spans, embeddings, sim_threshold=sim_threshold,
                                          max_gap_frames=max_gap_frames):
        # never merge two tracks that already belong to time-overlapping groups
        uf.union(old, new)
    return {t: uf.find(t) for t in spans}


# --------------------------------------------------------------------------- #
# plate propagation across linked groups
# --------------------------------------------------------------------------- #
def propagate_plates(canonical: dict, plate_map: dict) -> dict:
    """Within each canonical group, find the best plate read (highest plate_confidence_score) and
    give it to every member that lacks its own plate. Returns an ENRICHED plate_map:
      - members that already had a plate keep it (plate_source unchanged / 'own')
      - plate-less members inherit -> plate_source='reid_inherited:<best_tid>', + reid_group/_iou-style
    Recall-first: we only ADD plates, never remove a track's own read."""
    # group members by canonical id
    groups: dict = {}
    for tid, cid in canonical.items():
        groups.setdefault(cid, []).append(tid)

    enriched = {tid: dict(plate_map.get(tid, {})) for tid in canonical}
    for cid, members in groups.items():
        # the best plate in this group
        best_tid, best = None, None
        for tid in members:
            pm = plate_map.get(tid) or {}
            if pm.get("plate_candidate"):
                score = float(pm.get("plate_confidence_score", 0.0))
                if best is None or score > best:
                    best, best_tid = score, tid
        if best_tid is None:
            continue                                   # no plate anywhere in the group
        for tid in members:
            e = enriched[tid]
            if e.get("plate_candidate"):
                e.setdefault("plate_source", "own")
                e["reid_group"] = sorted(members)
                continue
            src = plate_map[best_tid]
            e["plate_candidate"] = src["plate_candidate"]
            e["plate_confidence_score"] = src.get("plate_confidence_score", 0.0)
            e["plate_source"] = f"reid_inherited:{best_tid}"
            e["reid_inherited_from"] = best_tid
            e["reid_group"] = sorted(members)
    return enriched


def reid_enrich(spans: dict, embeddings: dict, plate_map: dict, *, fps: float = 30.0,
                sim_threshold: float = DEFAULT_SIM_THRESHOLD,
                max_gap_sec: float = DEFAULT_MAX_GAP_SEC) -> "tuple[dict, dict]":
    """Top-level: link tracks then propagate plates. Returns (enriched_plate_map, canonical_map)."""
    max_gap_frames = max(1, round(max_gap_sec * fps))
    canonical = link_tracks(spans, embeddings, sim_threshold=sim_threshold,
                            max_gap_frames=max_gap_frames)
    return propagate_plates(canonical, plate_map), canonical
