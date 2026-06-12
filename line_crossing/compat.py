"""
Legacy compatibility shim (Phase 1, removed in Phase 4).

`to_legacy_dict` collapses the new `list[Lane]` polyline output down to the old
flat dict that `crossing_detector.py` and `main.py` already consume:

    {"solid_left": (x1, y1, x2, y2), "dashed_right": (...), ...}

Keys are "<lane_type>_<side>" where side is derived from the ego-relative
`position` (-1 -> left, +1 -> right). This lets us swap in the DL detector
during Phase 2 WITHOUT touching the crossing logic.

Note: during Phase 2 (pretrained CLRerNet, no type head yet) `lane_type` is
"unknown", so keys are "unknown_left" / "unknown_right" and the existing
`startswith("solid")` violation gate correctly fires nothing. That is expected:
Phase 2 validates plumbing/VRAM, not violations.
"""

from __future__ import annotations

from .lane_types import Lane


def to_legacy_dict(lanes: list[Lane]) -> dict[str, tuple[int, int, int, int]]:
    """
    Reduce the two ego-boundary lanes (position +-1) to their endpoint segments
    in the legacy 4-tuple format. If two lanes contend for the same key, the
    higher-confidence one wins.
    """
    out: dict[str, tuple[int, int, int, int]] = {}
    best_score: dict[str, float] = {}

    for ln in lanes:
        if ln.position not in (-1, 1) or len(ln.points) < 2:
            continue
        side = "left" if ln.position == -1 else "right"
        key = f"{ln.lane_type}_{side}"

        if key in out and ln.score <= best_score[key]:
            continue

        (xb, yb), (xt, yt) = ln.bottom, ln.top
        out[key] = (xb, yb, xt, yt)
        best_score[key] = ln.score

    return out
