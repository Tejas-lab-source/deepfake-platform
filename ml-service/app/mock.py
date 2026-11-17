"""Deterministic mock scores for MOCK_MODE.

Every number is derived from the file's sha256, so the same upload always gives the same
result. The scores carry no information about the image. They exist so the whole app can
run without weights, a GPU or API keys. Verdicts still go through the real decision rules.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from PIL import Image

from app.head import Calibration
from app.pipeline import aggregate, decide
from app.progress import ProgressFn, no_progress
from app.video import VideoInfo, aggregate_frames, probe, sample_indices, timestamp

VARIANT_NAMES = ("F1", "F2", "F3", "mix")


def file_digest(path: Path, salt: bytes = b"") -> bytes:
    h = hashlib.sha256(salt)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def _unit(digest: bytes, i: int) -> float:
    """i-th pseudo-random number in [0, 1) from a 32-byte digest (i < 16)."""
    return int.from_bytes(digest[2 * i : 2 * i + 2], "big") / 65536.0


def _signed(digest: bytes, i: int) -> float:
    return 2.0 * _unit(digest, i) - 1.0


def mock_face_scores(digest: bytes, k: int) -> dict[str, Any]:
    base = 3.0 * _signed(digest, 0)
    # Squared so most files are stable and a few are not.
    instability = 1.5 * _unit(digest, 1) ** 2
    variants = {"clean": base}
    for i, name in enumerate(VARIANT_NAMES[: max(0, k)]):
        variants[name] = base + instability * _signed(digest, 2 + i)
    return {
        "variantLogits": variants,
        "sbi": base + 1.5 * _signed(digest, 7),
        "beautifiedLikelihood": _unit(digest, 8),
    }


def analyze_image(path: Path, cal: Calibration, k: int) -> dict[str, Any]:
    with Image.open(path) as im:
        width, height = im.size
    scores = mock_face_scores(file_digest(path), k)
    variants: dict[str, float] = scores["variantLogits"]
    mean_logit, spread = aggregate(variants)
    decision = decide(cal, variants, scores["sbi"], scores["beautifiedLikelihood"])
    side = int(min(width, height) * 0.6)
    face = {
        # Mock mode does not detect faces: this is a centred placeholder box.
        "box": [(width - side) // 2, (height - side) // 2, side, side],
        "scores": {"gendHead": round(mean_logit, 4), "sbi": round(scores["sbi"], 4)},
        "variantLogits": {name: round(v, 4) for name, v in variants.items()},
        "spread": round(spread, 4),
        "beautifiedLikelihood": round(scores["beautifiedLikelihood"], 4),
        "heatmaps": None,
        "heatmapIoU": None,
    }
    return {
        "verdict": decision.verdict,
        "probability": round(decision.probability, 4),
        "threshold": cal.threshold,
        "confident": decision.confident,
        "uncertainReason": decision.uncertain_reason,
        "faces": [face],
        "modelCardVersion": cal.version,
        "mock": True,
    }


def analyze_video(
    path: Path,
    cal: Calibration,
    max_frames: int,
    top_k: int,
    max_decode_frames: int = 18000,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Mock video result: real frame sampling and timestamps when OpenCV is installed,
    placeholder scores derived from the file hash for every sampled frame."""
    report = progress or no_progress
    base_logit = 3.0 * _signed(file_digest(path), 0)
    info: VideoInfo | None = None
    if importlib.util.find_spec("cv2") is not None:
        report("detecting_faces", 0, 1, "videos")
        info = probe(path, max_decode_frames)
        indices = sample_indices(info.frame_count, max_frames)
    else:
        # Without OpenCV the file is not decoded: synthetic frames without timestamps.
        indices = list(range(max(1, min(max_frames, 16))))
    report("extracting_frames", 0, len(indices), "frames")
    frames: list[dict[str, Any]] = []
    for n, index in enumerate(indices):
        report("scoring", n, len(indices), "frames")
        d = file_digest(path, salt=f"frame-{index}".encode())
        has_face = _unit(d, 0) > 0.15
        frames.append(
            {
                "index": index,
                "timestamp": None if info is None else timestamp(index, info.fps),
                "decoded": info is not None,
                "hasFace": has_face,
                "faceCount": 1 if has_face else 0,
                "logit": round(base_logit + 0.8 * _signed(d, 1), 4) if has_face else None,
                "sbi": None,
                "explained": False,
            }
        )
    report("aggregating", 0, 1, "frames")
    result = aggregate_frames(frames, cal, top_k)
    return {
        **result,
        "kind": "video",
        "video": None if info is None else {**info.to_dict(), "sampledFrames": len(indices)},
        "frames": frames,
        "faces": [],
        "modelCardVersion": cal.version,
        "mock": True,
    }
