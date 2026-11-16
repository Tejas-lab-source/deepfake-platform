"""GenD (CLIP ViT-L/14) wrapper: returns the 1024-d pre-classifier feature and one logit.

Minimal re-implementation of `src/hf/modeling_gend.py` from https://github.com/yermandy/GenD
(MIT licence, Copyright (c) 2025 Andy), for the released `yermandy/GenD_CLIP_L_14` checkpoint.
What was checked against that repository and the checkpoint:

* backbone: the vision tower of `openai/clip-vit-large-patch14`; the feature is its
  `pooler_output` (1024-d);
* head: `LinearProbe(1024, 2)`; the checkpoint config says `LinearNorm`, so features are
  L2-normalised before the linear layer;
* preprocessing: the backbone's own CLIPProcessor (resize to 224 bicubic, centre crop 224,
  CLIP mean and std);
* class 1 is "fake" (`p_fake = probs[1]` in GenD's `app/run.py`), so the single logit is
  logits[1] - logits[0], and sigmoid(logit) equals softmax(logits)[1].

Unlike the original, this builds the vision tower from its config file alone, so the 1.7 GB
OpenAI CLIP weights are not downloaded: GenD's checkpoint already contains the tower.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import nn

FEATURE_DIM = 1024
_BACKBONE = "openai/clip-vit-large-patch14"


class _Encoder(nn.Module):
    def __init__(self, vision_config: Any) -> None:
        super().__init__()
        from transformers import CLIPVisionModel

        self.vision_model = CLIPVisionModel(vision_config).vision_model


class _Probe(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 2)


class GenDModel(nn.Module):
    """Module layout matches the checkpoint's parameter names."""

    def __init__(self, vision_config: Any, normalize: bool) -> None:
        super().__init__()
        self.feature_extractor = _Encoder(vision_config)
        self.model = _Probe(vision_config.hidden_size)
        self.normalize = normalize

    def features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.feature_extractor.vision_model(pixel_values).pooler_output
        return out

    def logit_from_features(self, features: torch.Tensor) -> torch.Tensor:
        x = F.normalize(features, p=2, dim=1) if self.normalize else features
        two = self.model.linear(x)
        return two[:, 1] - two[:, 0]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Single fake-vs-real logit, shape (N, 1). Used as the Grad-CAM model."""
        return self.logit_from_features(self.features(pixel_values)).unsqueeze(1)


class GenDDetector:
    def __init__(self, weights_dir: Path, device: torch.device) -> None:
        from safetensors.torch import load_file
        from transformers import CLIPConfig, CLIPImageProcessor

        gend_dir, clip_dir = weights_dir / "gend", weights_dir / "clip-vit-large-patch14"
        config = json.loads((gend_dir / "config.json").read_text())
        if config.get("backbone") != _BACKBONE:
            raise ValueError(f"unsupported GenD backbone: {config.get('backbone')}")
        if config.get("head") not in ("linear", "LinearNorm"):
            raise ValueError(f"unsupported GenD head: {config.get('head')}")

        vision_config = CLIPConfig.from_pretrained(clip_dir).vision_config
        self.model = GenDModel(vision_config, normalize=config["head"] == "LinearNorm")
        state = load_file(gend_dir / "model.safetensors")
        # The text-image projection is in the checkpoint but GenD never uses it.
        state = {k: v for k, v in state.items() if "visual_projection" not in k}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)

        self.device = device
        self.fp16 = device.type == "cuda"
        self.model.to(device)
        if self.fp16:
            self.model.half()
        self._processor = CLIPImageProcessor.from_pretrained(clip_dir)

    def preprocess(self, crops: list[NDArray[np.uint8]]) -> torch.Tensor:
        """RGB uint8 crops -> normalised (N, 3, 224, 224) tensor on the model's device."""
        pixels: torch.Tensor = self._processor(images=crops, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.device)
        return pixels.half() if self.fp16 else pixels

    @torch.inference_mode()
    def score(
        self, crops: list[NDArray[np.uint8]]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return (features (N, 1024), logits (N,)) for a batch of crops."""
        features = self.model.features(self.preprocess(crops))
        logits = self.model.logit_from_features(features)
        return (
            features.float().cpu().numpy().astype(np.float32),
            logits.float().cpu().numpy().astype(np.float32),
        )
