"""F2, makeup and tone: skin brightening, lip tint, blush, warmth and saturation."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.faces import FaceGeometry, Image

LIP_TINTS = ((190, 40, 70), (170, 50, 90), (200, 70, 60))  # RGB
BLUSH = (225, 110, 120)  # RGB


def _cheek_blobs(geom: FaceGeometry) -> NDArray[np.float32]:
    """Two soft blobs on the cheeks, placed from the eye and lip centres."""
    left, right = geom.points("left_eye").mean(0), geom.points("right_eye").mean(0)
    eyes_mid = (left + right) / 2.0
    down = geom.points("lips").mean(0) - eyes_mid
    sigma = 0.14 * float(np.linalg.norm(left - right)) + 1e-3
    yy, xx = np.mgrid[0 : geom.height, 0 : geom.width].astype(np.float32)
    blobs = np.zeros((geom.height, geom.width), dtype=np.float32)
    for eye in (left, right):
        cx, cy = eye + 0.6 * down + 0.2 * (eye - eyes_mid)
        blobs += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (2.2 * sigma) ** 2))
    return np.clip(blobs, 0.0, 1.0) * geom.skin


def apply(img: Image, geom: FaceGeometry, strength: float, rng: np.random.Generator) -> Image:
    """Apply makeup and tone changes. strength in [0, 1]; 0 returns the image unchanged."""
    if strength <= 0:
        return img.copy()
    strength = min(1.0, strength)
    # Soft face mask whose feather falls inside the oval, so the background is not tinted.
    sigma = max(0.5, 0.02 * min(img.shape[:2]))
    k = 2 * round(2 * sigma) + 1
    face_soft = cv2.GaussianBlur(cv2.erode(geom.face, np.ones((k, k), np.uint8)), (0, 0), sigma)

    # Skin brightening: raise L in Lab inside the skin mask.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] += 16.0 * strength * geom.skin
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # Warmth and saturation over the face.
    hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 + 0.25 * strength * face_soft
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    out[..., 0] += 8.0 * strength * face_soft
    out[..., 2] -= 8.0 * strength * face_soft

    # Lip tint and blush: blend toward a colour.
    tint = np.array(LIP_TINTS[int(rng.integers(len(LIP_TINTS)))], dtype=np.float32)
    for mask, colour, amount in (
        (geom.lips, tint, 0.45),
        (_cheek_blobs(geom), np.array(BLUSH, dtype=np.float32), 0.3),
    ):
        alpha = (mask * amount * strength)[..., None]
        out = out * (1.0 - alpha) + colour * alpha
    result = np.clip(np.round(out), 0, 255).astype(np.uint8)
    # The colour-space round trips shift untouched pixels by a level; put those back.
    untouched = np.maximum(face_soft, geom.lips) < 1e-3
    result[untouched] = img[untouched]
    return result
