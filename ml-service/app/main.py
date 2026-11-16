"""FastAPI routes. Private network only; every route except /health needs X-Internal-Secret."""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app import mock
from app.config import Settings, get_settings
from app.head import Calibration, load_calibration
from app.media import MediaError, resolve_upload, validate


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "time": self.formatTime(record),
        }
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload)


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


settings = get_settings()
_configure_logging(settings.log_level)
log = logging.getLogger("ml-service")

app = FastAPI(title="deepfake ml-service", docs_url=None, redoc_url=None)
calibration: Calibration = load_calibration(settings.model_card_path)
# Inference runs one request at a time.
inference_lock = threading.Lock()

REAL_MODE_MESSAGE = "Real models are not implemented yet (Phase 2). Run with MOCK_MODE=true."


def require_secret(
    cfg: Annotated[Settings, Depends(get_settings)],
    x_internal_secret: Annotated[str | None, Header()] = None,
) -> None:
    if x_internal_secret is None or not hmac.compare_digest(
        x_internal_secret.encode(), cfg.internal_secret.encode()
    ):
        raise HTTPException(status_code=401, detail="missing or invalid X-Internal-Secret")


Secured = Depends(require_secret)


class AnalyzeImageRequest(BaseModel):
    path: str
    k: int | None = Field(default=None, ge=0, le=16)
    explain: bool = True


class AnalyzeVideoRequest(BaseModel):
    path: str
    maxFrames: int | None = Field(default=None, ge=1, le=256)  # noqa: N815
    k: int | None = Field(default=None, ge=0, le=16)


def _models_ready() -> bool:
    return settings.mock_mode


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", dependencies=[Secured])
def ready() -> dict[str, Any]:
    if not _models_ready():
        raise HTTPException(status_code=503, detail=REAL_MODE_MESSAGE)
    return {"ready": True, "mockMode": settings.mock_mode}


@app.get("/model-info", dependencies=[Secured])
def model_info() -> dict[str, Any]:
    return {
        "mockMode": settings.mock_mode,
        "k": settings.k,
        "maxFrames": settings.max_frames,
        "videoTopK": settings.video_top_k,
        "headLoaded": False,
        "calibration": calibration.to_dict(),
    }


@app.post("/analyze/image", dependencies=[Secured])
def analyze_image(req: AnalyzeImageRequest) -> dict[str, Any]:
    if not _models_ready():
        raise HTTPException(status_code=503, detail=REAL_MODE_MESSAGE)
    try:
        file = resolve_upload(settings.upload_dir, req.path)
        validate(file, "image", settings.max_image_mb)
    except MediaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    k = settings.k if req.k is None else req.k
    with inference_lock:
        start = time.perf_counter()
        result = mock.analyze_image(file, calibration, k)
        score_ms = round((time.perf_counter() - start) * 1000)
    # Mock mode runs no detection, beautification or Grad-CAM, so those stages take 0 ms.
    result["timingsMs"] = {"detect": 0, "beautify": 0, "score": score_ms, "explain": 0}
    log.info("analyze_image", extra={"data": {"verdict": result["verdict"], "ms": score_ms}})
    return result


@app.post("/analyze/video", dependencies=[Secured])
def analyze_video(req: AnalyzeVideoRequest) -> dict[str, Any]:
    if not _models_ready():
        raise HTTPException(status_code=503, detail=REAL_MODE_MESSAGE)
    try:
        file = resolve_upload(settings.upload_dir, req.path)
        validate(file, "video", settings.max_video_mb)
    except MediaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    max_frames = settings.max_frames if req.maxFrames is None else req.maxFrames
    with inference_lock:
        start = time.perf_counter()
        result = mock.analyze_video(file, calibration, max_frames, settings.video_top_k)
        score_ms = round((time.perf_counter() - start) * 1000)
    result["timingsMs"] = {"detect": 0, "beautify": 0, "score": score_ms, "explain": 0}
    log.info("analyze_video", extra={"data": {"verdict": result["verdict"], "ms": score_ms}})
    return result
