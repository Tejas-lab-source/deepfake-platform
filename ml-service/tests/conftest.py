"""Shared fixtures. Tests that need the ML stack or downloaded weights skip when it is absent."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
LANDMARKER = WEIGHTS / "mediapipe" / "face_landmarker.task"
GEND = WEIGHTS / "gend" / "model.safetensors"
SBI = WEIGHTS / "sbi" / "FFc23.tar"

HAS_ML = all(
    importlib.util.find_spec(m) is not None
    for m in ("torch", "cv2", "mediapipe", "skimage", "facenet_pytorch", "pytorch_grad_cam")
)

needs_ml = pytest.mark.skipif(not HAS_ML, reason="ML stack not installed (requirements-ml.txt)")
needs_landmarker = pytest.mark.skipif(
    not (HAS_ML and LANDMARKER.is_file()), reason="needs the ML stack and the MediaPipe model"
)
needs_gend = pytest.mark.skipif(
    not (HAS_ML and LANDMARKER.is_file() and GEND.is_file()),
    reason="needs the ML stack and downloaded weights (scripts/download_weights.sh)",
)
needs_sbi = pytest.mark.skipif(
    not (HAS_ML and SBI.is_file()), reason="needs the ML stack and the SBI checkpoint (see README)"
)


@pytest.fixture(scope="session")
def photo():
    """A public-domain portrait that ships with scikit-image (NASA, Eileen Collins)."""
    from skimage import data

    return data.astronaut()


@pytest.fixture(scope="session")
def face(photo):
    import torch

    from app.faces import FaceDetector

    faces = FaceDetector(torch.device("cpu"), 0.9, 40).detect(photo)
    assert len(faces) == 1
    return faces[0]


@pytest.fixture(scope="session")
def crop(photo, face):
    from app.faces import gend_crop

    return gend_crop(photo, face.points)


@pytest.fixture(scope="session")
def landmarker():
    from app.faces import Landmarker

    lm = Landmarker(LANDMARKER)
    yield lm
    lm.close()


@pytest.fixture(scope="session")
def geom(crop, landmarker):
    g = landmarker.geometry(crop)
    assert g is not None
    return g
