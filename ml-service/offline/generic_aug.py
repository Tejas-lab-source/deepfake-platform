"""The generic (non-beauty) augmentation used by scheme H-d: colour jitter, then Gaussian
blur, then JPEG re-compression. Parameters are drawn from a seeded generator so the copy
can be regenerated exactly."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def draw_params(rng: np.random.Generator) -> dict[str, float]:
    return {
        "brightness": float(rng.uniform(-0.10, 0.10)),
        "contrast": float(rng.uniform(-0.10, 0.10)),
        "saturation": float(rng.uniform(-0.15, 0.15)),
        "blur_sigma": float(rng.uniform(0.5, 1.5)),
        "jpeg_quality": float(int(rng.integers(40, 91))),
    }


def apply(img: NDArray[np.uint8], params: dict[str, Any]) -> NDArray[np.uint8]:
    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) * (1.0 + params["contrast"]) + 0.5 + params["brightness"]
    hsv = cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + params["saturation"]), 0, 1)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    out = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (0, 0), params["blur_sigma"])
    ok, enc = cv2.imencode(
        ".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(params["jpeg_quality"])]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded: NDArray[np.uint8] = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].copy()
    return decoded
