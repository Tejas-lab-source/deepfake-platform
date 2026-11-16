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

    log_level: str = "INFO"

    @property
    def model_card_path(self) -> Path:
        return self.weights_dir / "model_card.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
