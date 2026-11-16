"""Linear head on GenD features: the model, its initialisation and the training loss.

The head computes logit = w · normalize(z) + b, the same form as GenD's own `LinearNorm`
head, so it can start exactly at GenD's weights (w = W[1] - W[0], b = b[1] - b[0]).
head.pt holds only tensors and plain values, so the service loads it with
torch.load(weights_only=True).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

FEATURE_DIM = 1024


def normalize(features: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
    x = np.asarray(features, dtype=np.float32)
    return (x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


def gend_head_from_checkpoint(gend_dir: Path) -> tuple[NDArray[np.float32], float]:
    """GenD's own head as a single logit: (w = W[1] - W[0], b = b[1] - b[0])."""
    from safetensors import safe_open

    with safe_open(str(gend_dir / "model.safetensors"), "pt") as f:
        weight = f.get_tensor("model.linear.weight").float().numpy()
        bias = f.get_tensor("model.linear.bias").float().numpy()
    if weight.shape != (2, FEATURE_DIM):
        raise ValueError(f"unexpected GenD head shape {weight.shape}")
    return (weight[1] - weight[0]).astype(np.float32), float(bias[1] - bias[0])


def head_logits(
    weight: NDArray[np.floating[Any]], bias: float, normalized: NDArray[np.floating[Any]]
) -> NDArray[np.float32]:
    out: NDArray[np.float32] = (np.asarray(normalized, np.float32) @ weight + bias).astype(
        np.float32
    )
    return out


def invariance_loss(
    clean_logit: torch.Tensor,
    copy_logit: torch.Tensor,
    target: torch.Tensor,
    lambda_p: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """L = BCE(h(z_x), y) + BCE(h(z_e(x)), y) + λp · (h(z_x) - h(z_e(x)))², batch mean.

    `sample_weight` balances the classes; it multiplies every term of a sample.
    """
    bce_clean = F.binary_cross_entropy_with_logits(clean_logit, target, reduction="none")
    bce_copy = F.binary_cross_entropy_with_logits(copy_logit, target, reduction="none")
    per_sample = bce_clean + bce_copy + lambda_p * (clean_logit - copy_logit) ** 2
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
        return per_sample.sum() / sample_weight.sum()
    return per_sample.mean()


def class_weights(y: NDArray[np.integer[Any]]) -> NDArray[np.float32]:
    """Weight each sample by N / (2 · N_class) so both classes count equally."""
    y = np.asarray(y)
    weights = np.ones(len(y), dtype=np.float32)
    for cls in (0, 1):
        count = int((y == cls).sum())
        if count:
            weights[y == cls] = len(y) / (2.0 * count)
    return weights


def save_head(
    path: Path, weight: NDArray[np.floating[Any]], bias: float, meta: dict[str, Any]
) -> str:
    """Write head.pt atomically and return its SHA-256."""
    import io

    from offline.common import _atomic_write_bytes, sha256_file

    weight = np.asarray(weight, dtype=np.float32).reshape(-1)
    if weight.shape != (FEATURE_DIM,):
        raise ValueError(f"head weight must have {FEATURE_DIM} values, got {weight.shape}")
    for key, value in meta.items():
        if not isinstance(value, str | int | float | bool | type(None) | list):
            raise TypeError(f"meta[{key!r}] must be a plain value for weights_only loading")
    buffer = io.BytesIO()
    torch.save(
        {
            "weight": torch.from_numpy(weight.copy()),
            "bias": float(bias),
            "l2_normalize": True,
            "meta": dict(meta),
        },
        buffer,
    )
    _atomic_write_bytes(path, buffer.getvalue())
    return sha256_file(path)
