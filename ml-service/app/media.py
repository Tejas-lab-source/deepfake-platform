"""Upload validation: path containment, size limits and magic bytes (pipeline step 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

Kind = Literal["image", "video"]


class MediaError(ValueError):
    """The file is missing, too large or not an accepted type."""


def sniff(header: bytes) -> tuple[Kind, str] | None:
    """Identify a file from its first bytes. Returns (kind, format) or None."""
    if header.startswith(b"\xff\xd8\xff"):
        return "image", "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "png"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image", "webp"
    if header[4:8] == b"ftyp":
        return "video", "mp4"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "video", "webm"
    return None


def resolve_upload(upload_dir: Path, path: str) -> Path:
    """Resolve `path` inside upload_dir; refuse anything that escapes it."""
    root = upload_dir.resolve()
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root):
        raise MediaError("path is outside the upload directory")
    if not candidate.is_file():
        raise MediaError("file not found")
    return candidate


def validate(file: Path, expected: Kind, max_mb: int) -> str:
    """Check size and magic bytes. Returns the detected format."""
    if file.stat().st_size > max_mb * 1024 * 1024:
        raise MediaError(f"file exceeds the {max_mb} MB limit")
    with file.open("rb") as fh:
        found = sniff(fh.read(16))
    if found is None or found[0] != expected:
        raise MediaError(f"file is not a supported {expected}")
    return found[1]
