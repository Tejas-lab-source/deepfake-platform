"""Service configuration. Every tunable lives here or in model_card.json, not in code."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mock_mode: bool = True
    internal_secret: str = "dev-internal-secret-change-me"

    upload_dir: Path = Path("/data/uploads")
    weights_dir: Path = Path("weights")

    # Number of beautified copies per face. 0 is the fast mode.
    k: int = 4
    max_image_mb: int = 10
    max_video_mb: int = 100
    max_frames: int = 32
    # Video verdict = mean of the top-k frame logits.
    video_top_k: int = 5

    # Real-mode pipeline settings.
    device: str = "auto"  # auto, cpu or cuda
    face_min_prob: float = 0.9
    face_min_size: int = 40
    max_faces: int = 5
    # Images larger than this on their long side are downscaled before detection.
    max_image_side: int = 1600
    gend_crop_size: int = 224
    gend_crop_scale: float = 1.3
    # Random strength range for the beautified copies.
    strength_min: float = 0.3
    strength_max: float = 1.0
    # Grad-CAM is run for this many faces per image (the deciding face first).
    max_explain_faces: int = 1
    heatmap_top_fraction: float = 0.2

    log_level: str = "INFO"

    @property
    def head_path(self) -> Path:
        return self.weights_dir / "head.pt"

    @property
    def sbi_path(self) -> Path:
        return self.weights_dir / "sbi" / "FFc23.tar"

    @property
    def landmarker_path(self) -> Path:
        return self.weights_dir / "mediapipe" / "face_landmarker.task"

    @property
    def derived_dir(self) -> Path:
        """Where crops, filtered images and heatmaps are written (inside the upload volume)."""
        return self.upload_dir / "derived"

    @property
    def model_card_path(self) -> Path:
        return self.weights_dir / "model_card.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
