"""
Single source of truth for the lane-class taxonomy (V2: 4 classes).

V2 adds `traffic_island` (the separation/gore zone, פס הפרדה) as a first-class,
trained class. The class merge/drop is done at Roboflow version-generation time,
so a clean export should ship these canonical names:

    solid_white_lane   - solid white lines
    yellow_solid_lane  - solid yellow lines
    dashed_lane        - all dashed lines (white & yellow unified; identical
                         traffic meaning in Israel)
    traffic_island     - separation / gore / median zones (2D region)

This module stays in the pipeline as a *defensive validator/normalizer*: both the
trainer and the inference/pseudo-labeler import it so that, even if a raw export
uses legacy or foreign names, the raw->canonical mapping, the ignore set, and the
overlay colours never drift apart. It tolerates a SUBSET of these classes (a 3- or
4-class export both validate), and folds known aliases into the canonical set.

Island aliases folded in -> traffic_island:
  * `neutral_split_zone` / `polygon` : the OLD Israeli gore-zone labels (the class
    already baked into phase3_israeli_head). V1 dropped these as too few; V2 keeps
    them and merges the foreign island polygons (road_guard_v3_merged) to learn the
    region properly.
  * `traffic-island` / `traffic_island` : foreign / canonical island names.
Still IGNORED: `lane` (generic AI pre-labels), `lanes`, `yellowlane`.
"""
from __future__ import annotations

# Raw class name -> canonical pipeline name. Canonical names map to themselves
# (identity); the rest fold legacy/foreign aliases into the canonical set so a
# mixed export (old Israeli gore + foreign island polygons) lines up cleanly.
RAW_TO_CANONICAL: dict[str, str] = {
    # solid white
    "solid_white_lane": "solid_white_lane",
    "solid_white": "solid_white_lane",
    # solid yellow
    "yellow_solid_lane": "yellow_solid_lane",
    "solid_yellow": "yellow_solid_lane",
    # dashed (white & yellow unified)
    "dashed_lane": "dashed_lane",
    "dashed_white": "dashed_lane",
    "dashed_yellow": "dashed_lane",
    # traffic island / separation zone (V2 new) + legacy/foreign aliases
    "traffic_island": "traffic_island",
    "traffic-island": "traffic_island",
    "neutral_split_zone": "traffic_island",
    "polygon": "traffic_island",
}

# Raw class names to drop entirely (never trained on, never drawn, never exported).
# `lane` is the generic AI pre-labels; the rest are legacy junk classes. NOTE:
# `polygon` was here in V1 (gore dropped); in V2 it is REMAPPED to traffic_island.
IGNORE_RAW: set[str] = {"lane", "lanes", "yellowlane"}

# Full canonical taxonomy for V2.
CANONICAL_CLASSES: tuple[str, ...] = (
    "solid_white_lane", "yellow_solid_lane", "dashed_lane", "traffic_island",
)

# BGR overlay colours per canonical class; unknowns fall back to grey.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "solid_white_lane":  (255, 255, 255),  # white
    "yellow_solid_lane": (0, 215, 255),    # gold/yellow
    "dashed_lane":       (0, 255, 0),      # green
    "traffic_island":    (255, 0, 255),    # magenta
}
FALLBACK_COLOR = (128, 128, 128)


def is_ignored(raw_or_canonical: str) -> bool:
    """True if this class should be skipped (legacy junk, or anything unmapped)."""
    name = raw_or_canonical
    if name in IGNORE_RAW:
        return True
    # A name that is neither a known raw class nor a known canonical class is junk.
    return name not in RAW_TO_CANONICAL and name not in CLASS_COLORS


def color_for(canonical_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(canonical_name, FALLBACK_COLOR)


def build_remap(raw_names: list[str]) -> tuple[list[str], dict[int, int]]:
    """
    Given the raw class-name list from a Roboflow data.yaml, return:
      * the kept canonical names in their new contiguous index order
      * a {old_index: new_index} remap (dropped/ignored classes are absent)

    On a clean V1 export this is effectively an identity map; it still reindexes
    survivors compactly if any ignored/unmapped class is present.
    """
    kept_names: list[str] = []
    old_to_new: dict[int, int] = {}
    for old_idx, raw in enumerate(raw_names):
        if raw in IGNORE_RAW:
            continue
        canon = RAW_TO_CANONICAL.get(raw)
        if canon is None:
            continue  # unmapped -> treat as junk
        # A canonical name can be the target of >1 raw class (e.g. polygon +
        # solid_white_lane both -> solid_white_lane); reuse its existing index.
        if canon in kept_names:
            old_to_new[old_idx] = kept_names.index(canon)
        else:
            old_to_new[old_idx] = len(kept_names)
            kept_names.append(canon)
    return kept_names, old_to_new
