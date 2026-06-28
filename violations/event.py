"""
ViolationEvent -- the single, violation-type-agnostic record every detector emits.

Speeding, solid-line crossing, right-of-yellow-line and any future composite violation
all produce this same shape, keyed on the offending vehicle's track id. Downstream
stages (LPR / evidence collection, alerting, backend upload) consume ViolationEvents
without caring *which* rule fired -- the ``violation_type`` + ``details`` carry the
type-specific payload. Adding a new violation = emit one of these; nothing downstream
changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ViolationType:
    """Known violation types (string-valued so they serialise straight to JSON and can
    carry bilingual labels later). New rules just add a constant here -- a free-form
    string is also accepted, so this list is a convenience, not a hard enum."""
    SPEEDING = "SPEEDING"
    SOLID_LINE_CROSSING = "SOLID_LINE_CROSSING"
    YELLOW_LINE_RIGHT = "YELLOW_LINE_RIGHT"
    TRAFFIC_ISLAND = "TRAFFIC_ISLAND"
    WRONG_WAY = "WRONG_WAY"


@dataclass
class ViolationEvent:
    """One violation committed by one vehicle.

    Attributes:
        vehicle_id:     tracker id of the offending vehicle -- the join key to its tracked
                        bboxes, buffered crops and (after the evidence stage) its plate.
        violation_type: one of ``ViolationType.*`` (free-form string allowed for new rules).
        key_frame:      the representative frame of the incident -- e.g. the peak-exceedance
                        frame for speeding, the crossing frame for a line crossing.
        confidence:     detector confidence in [0, 1] (1.0 for a hard/binary rule).
        details:        type-specific payload, e.g.
                        ``{"est_speed_kmh": 96, "speed_limit_kmh": 50, "over_by_kmh": 46}``.
    """
    vehicle_id: int
    violation_type: str
    key_frame: int
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)
