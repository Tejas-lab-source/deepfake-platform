"""SBI (Self-Blended Images) EfficientNet-B4 wrapper: returns one logit per face crop.

Follows `src/inference/model.py` and `inference_image.py` in
https://github.com/mapooon/SelfBlendedImages (research-use licence):

* network: `efficientnet_pytorch` EfficientNet-B4 with two outputs, stored under `net.`;
* checkpoint: `torch.load(path)["model"]`;
* input: RGB, 380 x 380, scaled to [0, 1], with no mean/std normalisation;
* class 1 is "fake" (`softmax(1)[:, 1]`), so the single logit is logits[1] - logits[0].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

INPUT_SIZE = 380


class SBIModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        # from_name builds the architecture without downloading ImageNet weights.
        self.net = EfficientNet.from_name("efficientnet-b4", num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single fake-vs-real logit, shape (N, 1)."""
        two = self.net(x)
        out: torch.Tensor = (two[:, 1] - two[:, 0]).unsqueeze(1)
        return out


class SBIDetector:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.model = SBIModel()
        # weights_only=True refuses anything but tensors and plain containers, so a tampered
        # checkpoint cannot run code. The official file holds {"model", "optimizer", "epoch"}.
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or "model" not in loaded:
            raise ValueError(f"{checkpoint} is not an SBI checkpoint: no 'model' entry")
        state = loaded["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)
        self.device = device
        self.model.to(device)

    def preprocess(self, crops: list[NDArray[np.uint8]]) -> torch.Tensor:
        for crop in crops:
            if crop.shape != (INPUT_SIZE, INPUT_SIZE, 3):
                raise ValueError(
                    f"SBI expects {INPUT_SIZE}x{INPUT_SIZE} RGB crops, got {crop.shape}"
                )
        batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float() / 255.0
        return batch.to(self.device)

    @torch.inference_mode()
    def score(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        logits = self.model(self.preprocess(crops))[:, 0]
        return logits.float().cpu().numpy().astype(np.float32)
