"""Evaluation metrics and the video-level bootstrap.

All decisions are "forced": p >= threshold means FAKE. NaN is returned, never an error,
when a metric is undefined for the data (for example AUC with one class).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating[Any]]


def auc(y: NDArray[np.integer[Any]], score: Array) -> float:
    """ROC AUC (probability a random fake outranks a random real; ties count half)."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    combined = np.concatenate([neg, pos])[order]
    # Average ranks for ties.
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1] == combined[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_pos = ranks[len(neg) :].sum()
    return float((rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def rate(mask: NDArray[np.bool_]) -> float:
    mask = np.asarray(mask, dtype=bool)
    return float(mask.mean()) if mask.size else float("nan")


def false_positive_rate(y: NDArray[np.integer[Any]], p: Array, threshold: float) -> float:
    """Share of real faces (y = 0) predicted FAKE."""
    y, p = np.asarray(y), np.asarray(p)
    return rate(p[y == 0] >= threshold)


def false_negative_rate(y: NDArray[np.integer[Any]], p: Array, threshold: float) -> float:
    """Share of fake faces (y = 1) predicted REAL."""
    y, p = np.asarray(y), np.asarray(p)
    return rate(p[y == 1] < threshold)


def flip_rate(p_clean: Array, p_copy: Array, threshold: float) -> float:
    return rate((np.asarray(p_clean) >= threshold) != (np.asarray(p_copy) >= threshold))


def cluster_bootstrap(
    clusters: NDArray[np.str_] | list[str],
    statistic: Callable[[NDArray[np.int64]], float],
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI that resamples whole clusters (videos), since frames are correlated.

    `statistic` receives the row indices of a resample. Resamples where the statistic is
    undefined (NaN) are skipped; (nan, nan) if fewer than half are defined.
    """
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    rows = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([rows[c] for c in picked])
        value = statistic(idx)
        if np.isfinite(value):
            values.append(value)
    if len(values) < n_resamples / 2:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0
