"""Linear head, fusion and calibration.

Calibration values load from model_card.json; with no model card the defaults are identity
fusion, T = 1 and threshold 0.5. The invariance-trained linear head loads from head.pt; when
that file is missing the pipeline uses GenD's own logit.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FusionWeights:
    """Logistic fusion over [head logit, SBI logit, spread, beautifiedLikelihood]."""

    head: float = 1.0
    sbi: float = 0.0
    spread: float = 0.0
    beautified_likelihood: float = 0.0
    bias: float = 0.0


@dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    threshold: float = 0.5
    # UNCERTAIN when |p - threshold| < uncertain_margin.
    uncertain_margin: float = 0.1
    # `confident` needs |p - threshold| >= confident_margin.
    confident_margin: float = 0.25
    # UNCERTAIN when max - min of the variant logits exceeds this.
    spread_limit: float = 1.5
    # UNCERTAIN when the two detectors land on opposite sides and differ by more than this.
    detector_disagree_limit: float = 0.6
    fusion: FusionWeights = field(default_factory=FusionWeights)
    version: str = "defaults-uncalibrated"
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_calibration(path: Path) -> Calibration:
    """Read model_card.json; missing file or missing keys fall back to the defaults."""
    if not path.is_file():
        return Calibration(sources={"all": "built-in defaults (no model_card.json)"})
    data: dict[str, Any] = json.loads(path.read_text())
    defaults = Calibration()
    fusion = FusionWeights(**{**asdict(defaults.fusion), **data.get("fusion", {})})
    scalar_keys = (
        "temperature",
        "threshold",
        "uncertain_margin",
        "confident_margin",
        "spread_limit",
        "detector_disagree_limit",
    )
    scalars = {k: float(data.get(k, getattr(defaults, k))) for k in scalar_keys}
    return Calibration(
        **scalars,
        fusion=fusion,
        version=str(data.get("version", "unversioned")),
        sources=dict(data.get("sources", {})),
    )


@dataclass(frozen=True)
class LinearHead:
    """Linear head on GenD's 1024-d features: logit = w . f(x) + b.

    head.pt is a torch-saved dict {"weight": (D,), "bias": scalar, "l2_normalize": bool},
    written by offline/train_head.py.
    """

    weight: Any  # numpy float32 array, shape (D,)
    bias: float
    l2_normalize: bool

    @staticmethod
    def load(path: Path) -> LinearHead | None:
        if not path.is_file():
            return None
        import numpy as np
        import torch

        data = torch.load(path, map_location="cpu")
        weight = np.asarray(data["weight"], dtype=np.float32).reshape(-1)
        return LinearHead(weight, float(data["bias"]), bool(data.get("l2_normalize", True)))

    def logits(self, features: Any) -> Any:
        """features: (N, D) array -> (N,) logits."""
        import numpy as np

        x = np.asarray(features, dtype=np.float32)
        if self.l2_normalize:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x @ self.weight + self.bias


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def fuse(
    cal: Calibration,
    head_logit: float,
    sbi_logit: float | None,
    spread: float,
    beautified_likelihood: float,
) -> float:
    """Return the fused logit (before temperature)."""
    w = cal.fusion
    return (
        w.bias
        + w.head * head_logit
        + w.sbi * (sbi_logit if sbi_logit is not None else 0.0)
        + w.spread * spread
        + w.beautified_likelihood * beautified_likelihood
    )


def probability(cal: Calibration, fused_logit: float) -> float:
    return sigmoid(fused_logit / cal.temperature)
