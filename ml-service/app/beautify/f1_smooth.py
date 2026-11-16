"""F1, skin smoothing: bilateral, guided or Gaussian filtering inside the skin mask."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.faces import FaceGeometry, Image

METHODS = ("bilateral", "guided", "gaussian")


def _guided(img: NDArray[np.float32], radius: int, eps: float) -> NDArray[np.float32]:
    """Self-guided filter (He et al.), applied per channel with box filters."""
    size = (2 * radius + 1, 2 * radius + 1)
    mean = cv2.blur(img, size)
    var = cv2.blur(img * img, size) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    out: NDArray[np.float32] = cv2.blur(a, size) * img + cv2.blur(b, size)
    return out


def apply(
    img: Image,
    geom: FaceGeometry,
    strength: float,
    rng: np.random.Generator,
    method: str | None = None,
) -> Image:
    """Smooth the skin. strength in [0, 1]; 0 returns the image unchanged."""
    if strength <= 0:
        return img.copy()
    strength = min(1.0, strength)
    method = method or str(rng.choice(METHODS))
    unit = min(img.shape[:2]) / 224.0
    if method == "bilateral":
        smooth = cv2.bilateralFilter(img, d=-1, sigmaColor=30 + 45 * strength, sigmaSpace=5 * unit)
    elif method == "gaussian":
        smooth = cv2.GaussianBlur(img, (0, 0), (1.0 + 1.5 * strength) * unit)
    elif method == "guided":
        radius = max(1, round(5 * unit))
        smooth_f = _guided(img.astype(np.float32) / 255.0, radius, (0.05 + 0.1 * strength) ** 2)
        smooth = np.clip(smooth_f * 255.0, 0, 255)
    else:
        raise ValueError(f"unknown smoothing method: {method}")
    alpha = (geom.skin * strength)[..., None]
    out = img.astype(np.float32) * (1.0 - alpha) + smooth.astype(np.float32) * alpha
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
