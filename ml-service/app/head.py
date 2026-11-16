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
    sources: dict[str, Any] = field(default_factory=dict)
    # "defaults" (no model card), "ok", or "insufficient_validation_data".
    calibration_status: str = "defaults"
    fixture: bool = False
    # SHA-256 of the head.pt these values were fitted for; None when no head is expected.
    head_sha256: str | None = None

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
    head_sha = data.get("head_sha256")
    return Calibration(
        **scalars,
        fusion=fusion,
        version=str(data.get("version", "unversioned")),
        sources=dict(data.get("sources", {})),
        calibration_status=str(data.get("calibration_status", "unknown")),
        fixture=bool(data.get("fixture", False)),
        head_sha256=None if head_sha is None else str(head_sha),
    )


HEAD_FEATURE_DIM = 1024


class ModelFilesError(RuntimeError):
    """head.pt and model_card.json are missing, mismatched or not approved for use."""


@dataclass(frozen=True)
class LinearHead:
    """Linear head on GenD's 1024-d features: logit = w . f(x) + b.

    head.pt is a torch-saved dict {"weight": (1024,), "bias": scalar, "l2_normalize": bool,
    "meta": {...}}, written by offline/train_head.py. It is loaded with weights_only=True,
    so a file carrying Python objects is refused.
    """

    weight: Any  # numpy float32 array, shape (1024,)
    bias: float
    l2_normalize: bool
    sha256: str = ""

    @staticmethod
    def load(path: Path) -> LinearHead | None:
        if not path.is_file():
            return None
        import hashlib

        import numpy as np
        import torch

        data = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(data, dict) or "weight" not in data or "bias" not in data:
            raise ModelFilesError(f"{path} is not a head file: needs 'weight' and 'bias'")
        weight = np.asarray(data["weight"], dtype=np.float32).reshape(-1)
        if weight.shape != (HEAD_FEATURE_DIM,):
            raise ModelFilesError(
                f"{path}: head weight has {weight.size} values, expected {HEAD_FEATURE_DIM}"
            )
        bias = float(data["bias"])
        if not (np.isfinite(weight).all() and np.isfinite(bias)):
            raise ModelFilesError(f"{path}: head contains non-finite values")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return LinearHead(weight, bias, bool(data.get("l2_normalize", True)), sha)

    def logits(self, features: Any) -> Any:
        """features: (N, D) array -> (N,) logits."""
        import numpy as np

        x = np.asarray(features, dtype=np.float32)
        if self.l2_normalize:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x @ self.weight + self.bias


def check_model_files(cal: Calibration, head: LinearHead | None, allow_unvalidated: bool) -> None:
    """Refuse a head/model-card pair that does not belong together or is not approved.

    * A head needs a model card fitted for exactly that head (same SHA-256).
    * A card fitted for a head needs that head to be present.
    * A fixture card, or one whose validation data was insufficient, is refused unless
      `allow_unvalidated` is set (ALLOW_UNVALIDATED_MODEL_CARD=true, for tests only).
    """
    if head is not None:
        if cal.head_sha256 is None:
            raise ModelFilesError(
                "head.pt is present but model_card.json was not fitted for a head: "
                "run offline/calibrate.py for this head"
            )
        if cal.head_sha256 != head.sha256:
            raise ModelFilesError(
                f"head.pt (sha256 {head.sha256[:12]}) does not match the head model_card.json "
                f"was fitted for ({cal.head_sha256[:12]})"
            )
    elif cal.head_sha256 is not None:
        raise ModelFilesError(
            f"model_card.json was fitted for head {cal.head_sha256[:12]}, but head.pt is missing"
        )
    unvalidated = cal.fixture or cal.calibration_status not in ("defaults", "ok")
    if unvalidated and not allow_unvalidated:
        why = "a plumbing fixture" if cal.fixture else f"status {cal.calibration_status!r}"
        raise ModelFilesError(
            f"model_card.json is {why} and is not approved for use; "
            "set ALLOW_UNVALIDATED_MODEL_CARD=true only for tests"
        )


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
