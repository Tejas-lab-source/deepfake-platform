"""Aggregation and decision rules (pipeline steps 7-9).

Steps 1-6 and 10 (faces, beautified copies, detectors, Grad-CAM) arrive in Phase 2. The
functions here are pure and are shared by the mock path, so mock verdicts follow exactly
the same rules as real ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.head import Calibration, fuse, probability, sigmoid

Verdict = Literal["FAKE", "REAL", "UNCERTAIN"]


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    probability: float
    confident: bool
    uncertain_reason: str | None


def aggregate(variant_logits: dict[str, float]) -> tuple[float, float]:
    """Return (mean logit over copies, spread = max - min)."""
    values = list(variant_logits.values())
    if not values:
        raise ValueError("variant_logits is empty")
    return sum(values) / len(values), max(values) - min(values)


def decide(
    cal: Calibration,
    variant_logits: dict[str, float],
    sbi_logit: float | None,
    beautified_likelihood: float,
) -> Decision:
    mean_logit, spread = aggregate(variant_logits)
    p = probability(cal, fuse(cal, mean_logit, sbi_logit, spread, beautified_likelihood))
    distance = abs(p - cal.threshold)

    votes = {probability(cal, v) >= cal.threshold for v in variant_logits.values()}
    detectors_disagree = False
    if sbi_logit is not None:
        p_head, p_sbi = sigmoid(mean_logit), sigmoid(sbi_logit)
        detectors_disagree = (p_head >= 0.5) != (p_sbi >= 0.5) and abs(
            p_head - p_sbi
        ) > cal.detector_disagree_limit

    reason: str | None = None
    if distance < cal.uncertain_margin:
        reason = "near_threshold"
    elif spread > cal.spread_limit:
        reason = "unstable_under_beautification"
    elif len(votes) > 1:
        reason = "variants_disagree"
    elif detectors_disagree:
        reason = "detectors_disagree"

    if reason is not None:
        return Decision("UNCERTAIN", p, False, reason)
    verdict: Verdict = "FAKE" if p >= cal.threshold else "REAL"
    return Decision(verdict, p, distance >= cal.confident_margin, None)


def top_k_mean(logits: list[float], k: int) -> float:
    """Mean of the k largest frame logits (video aggregation)."""
    if not logits:
        raise ValueError("no frame logits")
    top = sorted(logits, reverse=True)[: max(1, k)]
    return sum(top) / len(top)
