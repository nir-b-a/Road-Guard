"""
Module E -- Hard-Negative Logger.

Isolates the joined events worth re-examining, so a later job can ingest them for the
project's active-learning / retraining loop without guessing structure.

Triggers (Directive 4):
  * HIGH_CONF_NO_PLATE : confidence > 0.8 AND plate is missing or weak (score < low_plate_score)
                         -> a confident offender we cannot bill: improve LPR / tracking.
  * BORDERLINE_CONF    : confidence in [0.4, 0.6]
                         -> the violation model itself is unsure: a labelling candidate.

Layout written:
  <out_dir>/<reason>/<clip_prefix>__viol<violation_id>.json   # one self-describing record
  <out_dir>/manifest.jsonl                                    # appended index for ingestion
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

HIGH_CONF = 0.8                 # strictly above this == "confident violation"
LOW_PLATE_SCORE = 0.3           # plate score below this == effectively unreadable
BORDERLINE = (0.4, 0.6)         # inclusive confidence band where the model is unsure

REASON_HIGH_CONF_NO_PLATE = "high_conf_no_plate"
REASON_BORDERLINE = "borderline_confidence"

_MANIFEST_KEYS = ("path", "reason", "clip_prefix", "violation_id", "track_id",
                  "confidence", "plate_candidate", "plate_confidence_score")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_event(
    event: dict,
    *,
    high_conf: float = HIGH_CONF,
    low_plate_score: float = LOW_PLATE_SCORE,
    borderline: "tuple[float, float]" = BORDERLINE,
) -> Optional[str]:
    """Return the hard-negative reason for an event, or None if it is not a hard negative."""
    conf = event.get("confidence", 0.0)
    plate = event.get("plate_candidate")
    plate_score = event.get("plate_confidence_score", 0.0)
    if conf > high_conf and (plate is None or plate_score < low_plate_score):
        return REASON_HIGH_CONF_NO_PLATE
    if borderline[0] <= conf <= borderline[1]:
        return REASON_BORDERLINE
    return None


def log_hard_negatives(
    events: list,
    out_dir: str,
    *,
    clip_prefix: str = "clip",
    high_conf: float = HIGH_CONF,
    low_plate_score: float = LOW_PLATE_SCORE,
    borderline: "tuple[float, float]" = BORDERLINE,
) -> list:
    """Write one JSON record per triggered event into <out_dir>/<reason>/ and append a
    manifest line. Returns the list of logged records (each with its written `path`)."""
    logged = []
    for ev in events:
        reason = classify_event(ev, high_conf=high_conf,
                                low_plate_score=low_plate_score, borderline=borderline)
        if reason is None:
            continue
        record = {**ev, "reason": reason, "clip_prefix": clip_prefix, "logged_at": _now_iso()}
        dest_dir = os.path.join(out_dir, reason)
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{clip_prefix}__viol{ev.get('violation_id')}.json")
        with open(path, "w") as fh:
            json.dump(record, fh, indent=2)
        logged.append({"path": path, **record})

    if logged:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "manifest.jsonl"), "a") as fh:
            for r in logged:
                fh.write(json.dumps({k: r[k] for k in _MANIFEST_KEYS if k in r}) + "\n")
    return logged
