"""FastAPI routes. Private network only; every route except /health needs X-Internal-Secret."""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from app import mock
from app.config import Settings, get_settings
from app.head import Calibration, load_calibration
from app.media import MediaError, resolve_upload, validate
from app.pipeline import PipelineError, RealPipeline
from app.progress import PROGRESS_ID, ProgressStore


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
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
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

calibration: Calibration = load_calibration(settings.model_card_path)
# Inference runs one request at a time.
inference_lock = threading.Lock()
# Measured progress of running requests, polled by the server (GET /progress/{id}).
progress_store = ProgressStore()


class ModelState:
    """The real pipeline, loaded once at startup in a background thread."""

    def __init__(self) -> None:
        self.pipeline: RealPipeline | None = None
        self.error: str | None = None

    def load(self) -> None:
        start = time.perf_counter()
        try:
            self.pipeline = RealPipeline(settings, calibration)
            log.info(
                "models loaded",
                extra={"data": {"seconds": round(time.perf_counter() - start, 1)}},
            )
        except Exception as exc:  # noqa: BLE001 - reported through /ready
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("model loading failed")


state = ModelState()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if not settings.mock_mode:
        threading.Thread(target=state.load, name="model-loader", daemon=True).start()
    yield
    if state.pipeline is not None:
        state.pipeline.close()


app = FastAPI(title="deepfake ml-service", docs_url=None, redoc_url=None, lifespan=lifespan)


def require_secret(
    cfg: Annotated[Settings, Depends(get_settings)],
    x_internal_secret: Annotated[str | None, Header()] = None,
) -> None:
    if x_internal_secret is None or not hmac.compare_digest(
        x_internal_secret.encode(), cfg.internal_secret.encode()
    ):
        raise HTTPException(status_code=401, detail="missing or invalid X-Internal-Secret")


Secured = Depends(require_secret)
Family = Literal["F1", "F2", "F3", "F4"]


ProgressId = Annotated[str, Field(pattern=PROGRESS_ID.pattern)]


class AnalyzeImageRequest(BaseModel):
    path: str
    k: int | None = Field(default=None, ge=0, le=16)
    explain: bool = True
    progressId: ProgressId | None = None  # noqa: N815


class AnalyzeVideoRequest(BaseModel):
    path: str
    maxFrames: int | None = Field(default=None, ge=1, le=256)  # noqa: N815
    k: int | None = Field(default=None, ge=0, le=16)
    explain: bool = True
    progressId: ProgressId | None = None  # noqa: N815


class FilterRequest(BaseModel):
    path: str
    family: Family
    strength: float = Field(ge=0.0, le=1.0)


def _real() -> RealPipeline:
    """The loaded real pipeline, or a 503 explaining why it is not there."""
    if state.pipeline is not None:
        return state.pipeline
    if state.error is not None:
        detail = f"Models failed to load: {state.error}. Run scripts/download_weights.sh."
        raise HTTPException(status_code=503, detail=detail)
    raise HTTPException(status_code=503, detail="Models are still loading.")


def _file(path: str, kind: Literal["image", "video"]) -> Path:
    try:
        file = resolve_upload(settings.upload_dir, path)
        validate(file, kind, settings.max_image_mb if kind == "image" else settings.max_video_mb)
    except MediaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return file


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", dependencies=[Secured])
def ready() -> dict[str, Any]:
    if not settings.mock_mode:
        _real()
    return {"ready": True, "mockMode": settings.mock_mode}


@app.get("/model-info", dependencies=[Secured])
def model_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "mockMode": settings.mock_mode,
        "k": settings.k,
        "maxFrames": settings.max_frames,
        "videoTopK": settings.video_top_k,
        "headLoaded": False,
        "calibrationStatus": calibration.calibration_status,
        "fixtureModelCard": calibration.fixture,
        "calibration": calibration.to_dict(),
    }
    if state.pipeline is not None:
        info.update(state.pipeline.info())
    elif not settings.mock_mode:
        info["loadError"] = state.error
    return info


@app.post("/analyze/image", dependencies=[Secured])
def analyze_image(req: AnalyzeImageRequest) -> dict[str, Any]:
    pipeline = None if settings.mock_mode else _real()
    file = _file(req.path, "image")
    k = settings.k if req.k is None else req.k
    report = progress_store.reporter(req.progressId)
    report("queued", 0, 1, "images")
    with inference_lock:
        start = time.perf_counter()
        if pipeline is None:
            result = mock.analyze_image(file, calibration, k)
            # Mock mode runs no detection, beautification or Grad-CAM: those take 0 ms.
            score_ms = round((time.perf_counter() - start) * 1000)
            result["timingsMs"] = {"detect": 0, "beautify": 0, "score": score_ms, "explain": 0}
        else:
            result = pipeline.analyze_image(file, k, req.explain, report)
        total_ms = round((time.perf_counter() - start) * 1000)
    log.info("analyze_image", extra={"data": {"verdict": result["verdict"], "ms": total_ms}})
    return result


@app.post("/analyze/video", dependencies=[Secured])
def analyze_video(req: AnalyzeVideoRequest) -> dict[str, Any]:
    pipeline = None if settings.mock_mode else _real()
    file = _file(req.path, "video")
    max_frames = min(settings.max_frames, req.maxFrames or settings.max_frames)
    k = 0 if req.k is None else req.k
    report = progress_store.reporter(req.progressId)
    report("queued", 0, 1, "videos")
    with inference_lock:
        start = time.perf_counter()
        try:
            if pipeline is None:
                result = mock.analyze_video(
                    file,
                    calibration,
                    max_frames,
                    settings.video_top_k,
                    settings.max_video_decode_frames,
                    report,
                )
                ms = round((time.perf_counter() - start) * 1000)
                result["timingsMs"] = {
                    "decode": ms,
                    "detect": 0,
                    "beautify": 0,
                    "score": 0,
                    "explain": 0,
                }
            else:
                result = pipeline.analyze_video(
                    file, k, max_frames, settings.video_top_k, req.explain, report
                )
        except MediaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except PipelineError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        total_ms = round((time.perf_counter() - start) * 1000)
    log.info(
        "analyze_video",
        extra={
            "data": {
                "verdict": result["verdict"],
                "frames": result.get("framesSampled"),
                "scored": result.get("framesScored"),
                "ms": total_ms,
            }
        },
    )
    return result


@app.get("/progress/{progress_id}", dependencies=[Secured])
def get_progress(progress_id: str) -> dict[str, Any]:
    if not PROGRESS_ID.fullmatch(progress_id):
        raise HTTPException(status_code=422, detail="invalid progress id")
    item = progress_store.get(progress_id)
    if item is None:
        raise HTTPException(status_code=404, detail="no progress for this id")
    return item


def _require_real_for_filters() -> RealPipeline:
    if settings.mock_mode:
        raise HTTPException(
            status_code=501, detail="Filters need the real pipeline; set MOCK_MODE=false."
        )
    return _real()


@app.post("/compare", dependencies=[Secured])
def compare(req: FilterRequest) -> dict[str, Any]:
    pipeline = _require_real_for_filters()
    file = _file(req.path, "image")
    with inference_lock:
        try:
            return pipeline.compare(file, req.family, req.strength)
        except PipelineError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@app.post("/beautify", dependencies=[Secured])
def beautify(req: FilterRequest) -> Response:
    pipeline = _require_real_for_filters()
    file = _file(req.path, "image")
    with inference_lock:
        try:
            png = pipeline.beautify(file, req.family, req.strength)
        except PipelineError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return Response(content=png, media_type="image/png")
