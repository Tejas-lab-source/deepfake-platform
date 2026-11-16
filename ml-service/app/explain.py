"""Grad-CAM for the CNN (SBI) and the ViT (GenD), heatmap IoU, and heatmap images."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

Heatmap = NDArray[np.float32]


class SingleLogitTarget:
    """Grad-CAM target for models that output one fake-vs-real logit per image."""

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output.reshape(-1)[0]


def vit_reshape_transform(tokens: torch.Tensor) -> torch.Tensor:
    """(N, 1 + S*S, C) ViT tokens -> (N, C, S, S): drop the class token, restore the grid."""
    patches = tokens[:, 1:, :]
    side = int(round(patches.shape[1] ** 0.5))
    if side * side != patches.shape[1]:
        raise ValueError(f"cannot arrange {patches.shape[1]} patch tokens in a square")
    return patches.reshape(tokens.shape[0], side, side, tokens.shape[2]).permute(0, 3, 1, 2)


def _run_cam(model: Any, layer: Any, inputs: torch.Tensor, reshape: Any = None) -> Heatmap:
    from pytorch_grad_cam import GradCAM

    params = list(model.parameters())
    was_frozen = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    try:
        with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
            maps = cam(input_tensor=inputs.float(), targets=[SingleLogitTarget()] * len(inputs))
    finally:
        for p, flag in zip(params, was_frozen, strict=True):
            p.requires_grad_(flag)
        model.zero_grad(set_to_none=True)
    out: Heatmap = np.asarray(maps, dtype=np.float32)
    return out


class _HeadOnGenD(torch.nn.Module):
    """GenD's encoder followed by the loaded linear head: the model whose logit the service
    actually reports, so Grad-CAM explains the verdict rather than GenD's original head."""

    def __init__(self, gend_model: Any, head: Any) -> None:
        super().__init__()
        self.gend_model = gend_model
        self.register_buffer("weight", torch.as_tensor(head.weight, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(float(head.bias), dtype=torch.float32))
        self.l2_normalize = bool(head.l2_normalize)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.gend_model.features(pixel_values).float()
        if self.l2_normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        out: torch.Tensor = (x @ self.weight + self.bias).unsqueeze(1)
        return out


def gradcam_gend(gend: Any, crops: list[NDArray[np.uint8]], head: Any = None) -> Heatmap:
    """Grad-CAM on GenD's last transformer block. Returns (N, 224, 224) maps in [0, 1].

    With a loaded head, the target is that head's logit; otherwise GenD's own logit.
    """
    model = gend.model
    half = next(model.parameters()).dtype == torch.float16
    if half:
        model.float()  # gradients in fp16 are unreliable
    try:
        layer = model.feature_extractor.vision_model.encoder.layers[-1].layer_norm1
        target = model if head is None else _HeadOnGenD(model, head)
        return _run_cam(target, layer, gend.preprocess(crops).float(), vit_reshape_transform)
    finally:
        if half:
            model.half()


def gradcam_sbi(sbi: Any, crops: list[NDArray[np.uint8]]) -> Heatmap:
    """Grad-CAM on SBI's conv head. Returns (N, 380, 380) maps in [0, 1]."""
    return _run_cam(sbi.model, sbi.model.net._conv_head, sbi.preprocess(crops))


def top_region_iou(a: Heatmap, b: Heatmap, top: float = 0.2) -> float:
    """IoU of the top `top` fraction of pixels of two heatmaps of the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"heatmap shapes differ: {a.shape} vs {b.shape}")
    if not 0 < top < 1:
        raise ValueError("top must be between 0 and 1")
    count = max(1, int(round(top * a.size)))

    def region(h: Heatmap) -> NDArray[np.bool_]:
        # Stable sort, so ties are broken the same way for both maps.
        idx = np.argsort(-h.reshape(-1), kind="stable")[:count]
        mask = np.zeros(h.size, dtype=bool)
        mask[idx] = True
        return mask

    ra, rb = region(a), region(b)
    return float((ra & rb).sum() / (ra | rb).sum())


def colourise(heatmap: Heatmap) -> NDArray[np.uint8]:
    """Heatmap in [0, 1] -> RGB image (JET colour map)."""
    grey = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    out: NDArray[np.uint8] = cv2.cvtColor(
        cv2.applyColorMap(grey, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
    )
    return out
