"""Shared helpers: atomic writes, hashing, deterministic seeds, run limits, logging and the
split-access guard that keeps the test split away from training and calibration."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Subsets written into the manifest. "val" is split in two halves by identity group.
SUBSETS = ("train", "val-A", "val-B", "test")
SPLIT_OF_SUBSET = {"train": "train", "val-A": "val", "val-B": "val", "test": "test"}
SPLITS = ("train", "val", "test")

# Which cache splits each stage may read. Training and calibration never see test.
ALLOWED_SPLITS = {
    "train_head": frozenset({"train", "val"}),
    "calibrate": frozenset({"val"}),
    "eval_lofo": frozenset({"train", "val", "test"}),
}

PROTOCOL_LABEL = (
    "Internal identity-disjoint split of the Celeb-DF v2 official test list (60/20/20 by "
    "identity component). NOT comparable with published Celeb-DF v2 benchmark results."
)
FIXTURE_LABEL = "PLUMBING FIXTURE: labels are arbitrary; no number here is an accuracy result."


class SplitAccessError(PermissionError):
    """A stage tried to read a split it must never see."""


def check_split_access(stage: str, split: str) -> None:
    allowed = ALLOWED_SPLITS.get(stage)
    if allowed is None:
        raise SplitAccessError(f"unknown stage {stage!r}")
    if split not in allowed:
        raise SplitAccessError(f"stage {stage!r} may not read the {split!r} split")


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("offline")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write to a temporary file in the same folder, then rename: readers never see a
    half-written file, and an interrupted run leaves no partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; outputs should be readable
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _atomic_write_bytes(path, buffer.getvalue())


def atomic_write_png(path: Path, rgb: NDArray[np.uint8]) -> None:
    """Lossless PNG: compression artefacts would interact with the detectors."""
    import cv2

    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(rgb[..., ::-1]))
    if not ok:
        raise OSError(f"could not encode {path}")
    _atomic_write_bytes(path, encoded.tobytes())


def read_png(path: Path) -> NDArray[np.uint8]:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not read {path}")
    return np.ascontiguousarray(bgr[..., ::-1])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def seed_for(*parts: object) -> int:
    """A 64-bit seed derived from the given parts; identical on every platform and run."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rng_for(*parts: object) -> np.random.Generator:
    return np.random.default_rng(seed_for(*parts))


class RunLimiter:
    """Stops a long run cleanly after `limit` items or `max_minutes`; resume by re-running."""

    def __init__(self, limit: int | None = None, max_minutes: float | None = None) -> None:
        self.limit = limit
        self.max_seconds = None if max_minutes is None else max_minutes * 60.0
        self.start = time.monotonic()
        self.done = 0

    def tick(self) -> None:
        self.done += 1

    def should_stop(self) -> bool:
        if self.limit is not None and self.done >= self.limit:
            return True
        return self.max_seconds is not None and time.monotonic() - self.start >= self.max_seconds

    def progress(self, remaining: int) -> str:
        elapsed = time.monotonic() - self.start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = remaining / rate if rate > 0 else float("nan")
        return f"{self.done} done, {remaining} left, {rate * 60:.1f}/min, ETA {eta / 60:.1f} min"


def git_commit(repo: Path) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
