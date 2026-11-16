"""F3, geometric reshaping: jaw slimming, eye enlargement and nose narrowing.

A landmark-guided piecewise-affine warp (scikit-image) with capped magnitudes. The transform
is returned so that a heatmap computed on the reshaped face can be warped back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skimage.transform import PiecewiseAffineTransform, warp

from app.faces import FaceGeometry, Image, Points


@dataclass(frozen=True)
class ReshapeLimits:
    """Largest change at strength = 1, as fractions."""

    jaw: float = 0.08  # jaw points move this fraction of their distance to the face axis
    eyes: float = 0.15  # eyes scale by 1 + this about their centre
    nose: float = 0.12  # nose points move this fraction toward the nose axis


LIMITS = ReshapeLimits()


def _fit(src: Points, dst: Points) -> Any:
    """Piecewise-affine map taking `src` coordinates to `dst` coordinates."""
    tform = PiecewiseAffineTransform.from_estimate(src.astype(np.float64), dst.astype(np.float64))
    if not tform:
        raise ValueError("could not estimate the piecewise-affine transform")
    return tform


@dataclass(frozen=True)
class ReshapeTransform:
    """Control points before (`src`) and after (`dst`) reshaping, in (x, y) pixels."""

    src: Points
    dst: Points
    height: int
    width: int

    @property
    def max_shift(self) -> float:
        return float(np.linalg.norm(self.dst - self.src, axis=1).max())

    def _warp(self, arr: NDArray[Any], frm: Points, to: Points) -> NDArray[np.float64]:
        # skimage.warp wants the map from output coordinates to input coordinates.
        out: NDArray[np.float64] = warp(
            arr, _fit(to, frm), output_shape=(self.height, self.width), mode="edge"
        )
        return out

    def forward(self, arr: NDArray[Any]) -> NDArray[np.float64]:
        """Warp an array from the original face to the reshaped face."""
        return self._warp(arr, self.src, self.dst)

    def backward(self, arr: NDArray[Any]) -> NDArray[np.float64]:
        """Warp an array (for example a heatmap) from the reshaped face back to the original."""
        return self._warp(arr, self.dst, self.src)


def _control_points(geom: FaceGeometry, strength: float) -> tuple[Points, Points]:
    oval = geom.points("oval")
    eyes = [geom.points("left_eye"), geom.points("right_eye")]
    nose, lips = geom.points("nose"), geom.points("lips")

    eye_y = float(np.mean([e[:, 1].mean() for e in eyes]))
    axis_x = float(np.mean([e[:, 0].mean() for e in eyes]))
    chin_y = float(oval[:, 1].max())

    oval_dst = oval.copy()
    weight = np.clip((oval[:, 1] - eye_y) / max(1e-3, chin_y - eye_y), 0.0, 1.0)
    oval_dst[:, 0] -= (oval[:, 0] - axis_x) * LIMITS.jaw * strength * weight

    eyes_dst = [e.mean(0) + (e - e.mean(0)) * (1.0 + LIMITS.eyes * strength) for e in eyes]

    nose_dst = nose.copy()
    nose_dst[:, 0] -= (nose[:, 0] - nose[:, 0].mean()) * LIMITS.nose * strength

    # Fixed anchors: the lips, and a ring of points on the image border.
    w, h = geom.width - 1.0, geom.height - 1.0
    t = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    border = np.array(
        [[x * w, 0] for x in t]
        + [[x * w, h] for x in t]
        + [[0, y * h] for y in t[1:-1]]
        + [[w, y * h] for y in t[1:-1]],
        dtype=np.float32,
    )
    src = np.concatenate([oval, *eyes, nose, lips, border]).astype(np.float32)
    dst = np.concatenate([oval_dst, *eyes_dst, nose_dst, lips, border]).astype(np.float32)

    # Keep points inside the image and drop duplicates (Delaunay needs distinct points).
    inside = (src[:, 0] >= 0) & (src[:, 0] <= w) & (src[:, 1] >= 0) & (src[:, 1] <= h)
    src, dst = src[inside], np.clip(dst[inside], [0, 0], [w, h]).astype(np.float32)
    _, keep = np.unique(np.round(src).astype(np.int32), axis=0, return_index=True)
    keep.sort()
    return src[keep], dst[keep]


def apply(
    img: Image, geom: FaceGeometry, strength: float, rng: np.random.Generator
) -> tuple[Image, ReshapeTransform]:
    """Reshape the face. strength in [0, 1]; 0 returns the image unchanged.

    Returns the reshaped image and its transform.
    """
    del rng  # F3 has no random choices; the argument keeps the filter signatures uniform
    strength = min(1.0, max(0.0, strength))
    src, dst = _control_points(geom, strength)
    transform = ReshapeTransform(src, dst, geom.height, geom.width)
    if strength <= 0:
        return img.copy(), transform
    out = transform.forward(img)  # float in [0, 1]
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8), transform
