"""Tunables for the Phase 4 crossing detector.

Every threshold here is either in SECONDS or in LANE WIDTHS - never in pixels.
That is deliberate: the pipeline normalizes each measurement by the local lane
width at the same image row (see offsets.py), so one set of numbers works on any
resolution, any focal length and at any distance in the frame. The only pixel
quantity left is `row_grid_step`, which is a sampling resolution, not a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Lane width / vehicle width, used ONLY as the fallback ruler when fewer than two
# lane lines are visible at the contact row (offsets.py). Lane ~3.4 m.
_WIDTH_MULT: dict[str, float] = {
    "motorcycle": 4.2,   # ~0.8 m
    "car": 1.9,          # ~1.8 m
    "bus": 1.4,          # ~2.5 m
    "truck": 1.4,
}


@dataclass
class CrossingConfig:
    # ---- Stage 0: lane tracks -------------------------------------------
    row_grid_step: int = 10
    """Vertical sampling pitch (px) of the row grid every lane track is resampled
    onto. Smaller = finer geometry, linearly more memory/compute."""

    ref_row_fracs: tuple[float, ...] = (0.95, 0.85, 0.75)
    """Rows (as a fraction of image height) used for the association signature.
    Near-field rows: best separated, most accurate, least affected by curvature."""

    assoc_gate_lane_frac: float = 0.40
    """A detection may join a track only if its predicted-vs-observed geometry error
    is under this fraction of the local lane spacing."""

    assoc_min_overlap_rows: int = 3
    """Minimum number of grid rows a detection and a track must share before their
    distance is even meaningful. Matching on the whole row grid rather than on a few
    fixed reference rows is what lets a SHORT fragment - one that never reaches the
    near field - join the track it belongs to."""

    frag_merge_tol_frac: float = 0.02
    """Segmentation heads emit one physical marking as several disjoint contours.
    Two fragments merge when their geometry agrees within this fraction of the frame
    width - either where they overlap, or across a gap along the bottom tangent."""

    frag_merge_gap_rows: int = 15
    """Largest vertical gap (in grid rows) that fragment merging will bridge."""

    track_coast_sec: float = 0.5
    """How long an unmatched track survives before it is closed."""

    stitch_gap_sec: float = 1.0
    """Offline second pass: two fragments separated by at most this gap are joined
    into one physical line if the extrapolated signature still matches the gate."""

    row_fill_extrap_rows: int = 25
    """A segmentation head returns a marking as SHORT segments, often far from the
    camera, so a track frequently has no geometry at the row where a vehicle's wheels
    actually are - which silently starves Stage 2 of samples. A lane line is a smooth
    curve, so each frame's observed rows are fitted (quadratic, or linear when there
    are too few) and evaluated up to this many grid rows beyond the observed extent,
    never outside the rows the track has covered at some point in its life.

    Set to 0 to switch the CONTINUATION off entirely: the geometry then stops exactly
    where the model saw paint stop. Interior gaps between observed rows are still
    bridged - that is interpolation between two real observations, not a guess about
    where the marking goes next."""

    row_fill_min_points: int = 4
    """Rows needed before the fit is quadratic rather than linear."""

    smooth_median_half: int = 2
    smooth_mean_half: int = 3
    """Centered (non-causal) smoothing half-windows applied to BOTH the lane
    geometry and the vehicle anchors. They MUST match between stages 0 and 1: the
    shared ego-motion component only cancels in the Stage 2 difference if both
    signals were filtered identically."""

    near_field_frac: float = 0.60
    """Rows below this fraction of H count as "near field" and carry the weight in
    the per-track lane-type vote (dashed vs solid is only separable where the gaps
    are resolvable)."""

    max_line_slope: float = 2.5
    """Reject a track whose median |dx/dy| exceeds this: a lane boundary runs roughly
    along the road, so in the image it is steep. A near-horizontal "line" is either a
    bad fit that has swept across the frame, or a STOP LINE - and a vehicle passing
    over a stop line at a junction is not a lane violation. Both belong out of the
    line set, and out of the lane-width ruler they would otherwise corrupt."""

    double_line_frac: float = 0.25
    """Two solid tracks closer than this fraction of a lane width, over their whole
    shared extent, are one painted boundary: the weaker one is suppressed so a
    single crossing is not reported twice."""

    gore_divergence: float = 1.6
    """A parallel lane pair's separation scales with (y - y_vanishing). A pair whose
    near/far separation ratio exceeds the perspective prediction by this factor is
    diverging - a gore / traffic island - and its events are tagged, not dropped."""

    # ---- Stage 1: vehicle ground anchors --------------------------------
    footprint_shrink: float = 0.20
    """Fraction of the bbox width pulled inward on the FAR side. The bbox bottom
    edge overhangs the real footprint there (the far wheels are farther away, so
    they image higher) and both sides carry body overhang + mirrors."""

    edge_margin_px: int = 2
    """A box within this many px of a frame border is truncated: its y2 is not a
    ground contact, so the sample is dropped (same test ground_distance uses)."""

    occlusion_overlap: float = 0.15
    """Drop the sample when a NEARER box (larger y2) covers more than this fraction
    of the bottom strip - the wheels are hidden and y2 sits on bodywork."""

    min_bbox_lane_frac: float = 0.25
    """Near-field gate: require the bbox to be at least this wide relative to the
    local lane width. Far vehicles are neither well detected nor well typed."""

    min_bbox_width_frac: float = 0.025
    """Absolute floor on bbox width, as a fraction of image width, applied ALWAYS.
    Far down the road the lane width is small too, so the ratio gate above stays
    satisfied by a 40-px box whose position is worth nothing - and those distant
    vehicles are neither detected nor typed reliably, which is exactly where the
    product does not want to be reporting violations."""

    track_edge_guard: int = 3
    """Ignore the first/last N samples of a vehicle track: a track that appears
    already straddling is usually a new detection, not a crossing."""

    max_aspect: float = 4.0
    min_aspect: float = 0.4
    """Sanity bounds on bbox width/height; outside them the box is merged or split."""

    # ---- Stage 2: signed offsets ----------------------------------------
    extrapolate_frac: float = 0.10
    """A close vehicle's contact row is often BELOW the lowest row the lane
    detector produced. Extend the track down its bottom tangent by up to this
    fraction of H rather than discarding the sample."""

    max_pair_offset: float = 1.5
    """Only (vehicle, line) pairs that come within this many lane widths of each
    other are evaluated at all."""

    gap_interp_sec: float = 0.5
    """Interpolate offset gaps up to this long, so a brief occlusion does not split
    one crossing into two half-events."""

    width_mult: dict[str, float] = field(default_factory=lambda: dict(_WIDTH_MULT))
    default_width_mult: float = 1.9

    # ---- Stage 3: events -------------------------------------------------
    traverse_margin: float = 0.15
    """Hysteresis band, in lane widths (~0.5 m). |u| must exceed this for a sample
    to count as committed to one side."""

    traverse_hold_sec: float = 0.40
    """The committed state must hold this long on BOTH sides of the transit."""

    traverse_max_transit_sec: float = 2.0
    """A transit slower than this is drift or a tracking artefact, not a crossing."""

    traverse_min_transit_sec: float = 0.20
    """...and one FASTER than this is impossible. Crossing the +-margin band means
    moving roughly a metre sideways; at any real lane-change rate that takes several
    tenths of a second. A two-frame sign flip is the LINE jumping - a track switch or
    a bad geometry fit - not the vehicle moving, and it shows up as several different
    vehicles "crossing" the same marking in the same instant."""

    traverse_monotonic: float = 0.60
    """net |du| / total |du| over the transit. Rejects jitter that happens to end up
    on the other side."""

    straddle_min_frac: float = 0.25
    """Fraction of the corrected footprint that must be on the far side of the line
    for the secondary (strafing) channel."""

    straddle_hold_sec: float = 0.50

    straddle_require_approach: bool = True
    """Require the vehicle to be committed to ONE side (|u| > traverse_margin) at
    some point before or after the straddle. Without this, a vehicle that simply
    drives alongside the marking with its body overhanging it - a truck hugging an
    edge line, or any vehicle whose smoothed centre sits on the line - reads as a
    permanent strafe. With it, "strafe" means the vehicle MOVED onto the line."""

    max_event_aspect: float = 2.0
    """Reject an event whose vehicle is presenting its FLANK (median bbox width/height
    above this). A vehicle travelling along our road shows its rear - aspect around
    1.0-1.4 - while cross traffic at a junction shows its side at 2.2-3.0. Those
    junction vehicles do change sides of a solid marking, so the geometry is right and
    the semantics are wrong: crossing a stop line at a junction is not a lane
    violation. Set high (e.g. 99) to disable.

    Note the discriminator that does NOT work here: the vehicle's image-space
    direction of travel. A car keeping pace ahead of the ego has almost no
    longitudinal image motion, so while it drifts across a line its velocity is nearly
    pure lateral - indistinguishable from cross traffic."""

    min_valid_samples: int = 6
    """A (vehicle, line) pair with fewer usable samples than this is not scored."""

    def frames(self, seconds: float, fps: float) -> int:
        """Seconds -> whole frames, at least 1."""
        return max(1, int(round(seconds * max(fps, 1e-6))))
