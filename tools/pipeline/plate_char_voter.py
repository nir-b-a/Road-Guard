"""
Per-character temporal voting for Israeli plates (length-first, then per-position).

Israeli plates are numeric, EITHER 7 digits (XX-XXX-XX) or 8 (XXX-XX-XXX). Whole-string voting
throws away good partial reads, and a naive fixed-slot vote MISALIGNS 7- vs 8-digit reads (slot 3
of a 7-digit read is a different field than slot 3 of an 8-digit read). So we:

  1. vote the LENGTH (7 vs 8), confidence-weighted,
  2. among reads of the winning length, vote each DIGIT POSITION independently (freq x confidence),
  3. reconstruct the winning digit string and format it.

Pure logic -- no cv2/torch -- so it unit-tests on CPU. Feed it (raw_plate_string, ocr_conf) reads
collected across a plate track's lifespan.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

VALID_LENGTHS = (7, 8)


@dataclass(frozen=True)
class CharRead:
    digits: str       # digits only, no separators (e.g. "1428132")
    conf: float       # OCR confidence in [0,1]


def digits_only(s: str | None) -> str:
    return re.sub(r"\D", "", s or "")


def format_israeli(digits: str) -> str | None:
    """Group bare digits into the Israeli layout, or None if not a valid length."""
    if len(digits) == 7:
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"   # XX-XXX-XX (pre-2017)
    if len(digits) == 8:
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"   # XXX-XX-XXX (post-2017)
    return None


def vote_characters(reads) -> tuple[str | None, float]:
    """reads: iterable of (plate_string_or_digits, confidence).
    Returns (formatted_plate | None, score) where score = mean per-slot agreement * mean conf."""
    norm = [CharRead(digits_only(p), float(c)) for p, c in reads]
    norm = [r for r in norm if len(r.digits) in VALID_LENGTHS]
    if not norm:
        return None, 0.0

    # 1. length vote (confidence-weighted)
    len_weight: dict[int, float] = defaultdict(float)
    for r in norm:
        len_weight[len(r.digits)] += r.conf
    best_len = max(len_weight, key=lambda k: (len_weight[k], k))   # ties -> longer plate
    group = [r for r in norm if len(r.digits) == best_len]

    # 2. per-position vote (confidence-weighted frequency)
    out, agreements = [], []
    for i in range(best_len):
        slot: dict[str, float] = defaultdict(float)
        for r in group:
            slot[r.digits[i]] += r.conf
        ch = max(slot, key=lambda k: (slot[k], k))
        out.append(ch)
        total = sum(slot.values())
        agreements.append(slot[ch] / total if total else 0.0)

    digits = "".join(out)
    mean_agreement = sum(agreements) / len(agreements)
    mean_conf = sum(r.conf for r in group) / len(group)
    return format_israeli(digits), round(mean_agreement * mean_conf, 6)
