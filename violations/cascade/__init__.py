"""
Stage-2 Cascade for solid-line-crossing detection.

Stage 1 (fast geometric trigger, lives in the existing ghost_mask / crossing harness) casts a
wide net: a vehicle whose contact point is near a solid-line mask is a *candidate*. Stage 2
(this package) is a high-precision FILTER that confirms or rejects each candidate using:

  * temporal persistence  (K-of-M sliding window)        -> temporal.py
  * a custom YOLOv11 tire model on a smart crop          -> tire_model.py
  * axle-vector straddle / relative-position geometry    -> geometry.py
  * a ghost-line buffer for paint occluded under the car -> temporal.py
  * an LPR fallback (with ground-plane parallax fix)     -> stage2.py + geometry.py
  * a near-field horizon gate                            -> geometry.py

Stage 2 can only REMOVE Stage-1 candidates, never add them -> it trades a little recall for a
large precision gain. Tunables are collected in CascadeParams (stage2.py) and swept by
optimize_cascade_parameters.py.
"""
from __future__ import annotations

from . import geometry, temporal  # noqa: F401

__all__ = ["geometry", "temporal"]
