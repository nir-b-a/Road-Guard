"""NaN-aware centered filters shared by Stage 0 and Stage 1.

Centered (non-causal) on purpose: this detector is offline, so there is no reason
to pay the lag of a causal filter. The SAME windows are applied to the lane
geometry and to the vehicle anchors - the ego-motion component common to both only
cancels in the Stage 2 difference if both sides were filtered identically.

Every function operates along axis 0 of a 2-D array (time x rows / time x series)
and treats NaN as "no observation" rather than as a value.
"""
from __future__ import annotations

import warnings

import numpy as np


def _windows(a: np.ndarray, half: int) -> np.ndarray:
    """(T, C) -> (T, C, 2*half+1) sliding windows along time, NaN-padded."""
    pad = np.full((half, a.shape[1]), np.nan, dtype=a.dtype)
    padded = np.concatenate([pad, a, pad], axis=0)
    return np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1, axis=0)


def nan_median_filter(a: np.ndarray, half: int) -> np.ndarray:
    """Centered median over time, ignoring NaN. Kills single-frame spikes (bbox
    jitter, a polyline that wobbled one frame) without touching real motion."""
    if half <= 0 or a.shape[0] == 0:
        return a.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN windows -> NaN
        return np.nanmedian(_windows(a, half), axis=-1).astype(a.dtype)


def nan_mean_filter(a: np.ndarray, half: int) -> np.ndarray:
    """Centered mean over time, ignoring NaN. Applied after the median to take the
    remaining high-frequency edge off."""
    if half <= 0 or a.shape[0] == 0:
        return a.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(_windows(a, half), axis=-1).astype(a.dtype)


def smooth(a: np.ndarray, median_half: int, mean_half: int) -> np.ndarray:
    """The standard median-then-mean pass used by both Stage 0 and Stage 1."""
    return nan_mean_filter(nan_median_filter(a, median_half), mean_half)


def interp_gaps(a: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of at most `max_gap` samples along time.

    Longer runs are left as NaN: a marking hidden for two seconds is not something
    to invent geometry for, but a marking hidden for four frames behind a wheel is.
    Never extrapolates beyond the first/last real sample.
    """
    if max_gap <= 0 or a.shape[0] == 0:
        return a.copy()
    out = a.copy()
    for c in range(a.shape[1]):
        col = out[:, c]
        ok = np.isfinite(col)
        n_ok = int(ok.sum())
        if n_ok < 2:
            continue
        idx = np.flatnonzero(ok)
        first, last = idx[0], idx[-1]
        # Walk the gaps between consecutive observations; fill only the short ones.
        for a0, b0 in zip(idx[:-1], idx[1:]):
            gap = b0 - a0 - 1
            if 0 < gap <= max_gap:
                col[a0 + 1:b0] = np.interp(np.arange(a0 + 1, b0), [a0, b0], [col[a0], col[b0]])
        col[:first] = np.nan
        col[last + 1:] = np.nan
    return out


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end] index runs where `mask` is True (end inclusive)."""
    out: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out
