"""F4, generative restoration with GFPGAN. Test only: never used in training or inference copies.

Optional. It needs the `gfpgan` package and its weights, neither of which is installed by
default. `available()` reports whether it can run; callers skip F4 when it cannot.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from app.faces import Image

WEIGHT_NAME = "GFPGANv1.4.pth"

_restorer: Any = None


class F4Unavailable(RuntimeError):
    """GFPGAN or its weights are not installed."""


def weight_path(weights_dir: Path) -> Path:
    return weights_dir / "gfpgan" / WEIGHT_NAME


def available(weights_dir: Path) -> bool:
    return importlib.util.find_spec("gfpgan") is not None and weight_path(weights_dir).is_file()


def apply(img: Image, strength: float, weights_dir: Path) -> Image:
    """Restore an aligned face crop and blend it with the original by `strength`."""
    global _restorer
    if strength <= 0:
        return img.copy()
    if not available(weights_dir):
        raise F4Unavailable(
            f"F4 needs `pip install gfpgan` and {weight_path(weights_dir)} (see the README)"
        )
    if _restorer is None:
        from gfpgan import GFPGANer

        _restorer = GFPGANer(
            model_path=str(weight_path(weights_dir)),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
    bgr = np.ascontiguousarray(img[..., ::-1])
    _, restored_faces, _ = _restorer.enhance(
        bgr, has_aligned=True, only_center_face=True, paste_back=False
    )
    if not restored_faces:
        raise F4Unavailable("GFPGAN returned no restored face")
    import cv2

    restored = cv2.resize(restored_faces[0], (img.shape[1], img.shape[0]))[..., ::-1]
    alpha = min(1.0, strength)
    out = img.astype(np.float32) * (1.0 - alpha) + restored.astype(np.float32) * alpha
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
