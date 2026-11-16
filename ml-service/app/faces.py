"""Face detection (MTCNN), the two detector crops, and MediaPipe landmarks.

Crop geometry follows each detector's own repository:

* GenD aligns the face with a similarity transform from five landmarks to a fixed template,
  with scale 1.3 (`align_face` in yermandy/GenD `detector.py`, MIT licence). GenD uses
  RetinaFace landmarks; MTCNN returns the same five points in the same order.
* SBI crops the detection box plus one eighth of its width and height on each side, then
  resizes to 380 x 380 (`crop_face(..., crop_by_bbox=True, phase='test')` in
  mapooon/SelfBlendedImages `src/inference/preprocess.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Points = NDArray[np.float32]

# GenD's five-point template (eyes, nose, mouth corners) in unit coordinates.
_GEND_TEMPLATE = np.array(
    [[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]], dtype=np.float32
)


@dataclass(frozen=True)
class DetectedFace:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1
    prob: float
    points: Points  # (5, 2): left eye, right eye, nose, mouth left, mouth right

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def xywh(self, scale: float = 1.0) -> list[int]:
        x0, y0, x1, y1 = (v * scale for v in self.box)
        return [round(x0), round(y0), round(x1 - x0), round(y1 - y0)]


class FaceDetector:
    """MTCNN from facenet-pytorch. Returns faces sorted by area, largest first."""

    def __init__(self, device: Any, min_prob: float, min_face_size: int) -> None:
        from facenet_pytorch import MTCNN

        self._mtcnn = MTCNN(keep_all=True, device=device, min_face_size=min_face_size)
        self._min_prob = min_prob

    def detect(self, image: Image) -> list[DetectedFace]:
        from PIL import Image as PILImage

        boxes, probs, points = self._mtcnn.detect(PILImage.fromarray(image), landmarks=True)
        if boxes is None:
            return []
        faces = [
            DetectedFace(
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                prob=float(p),
                points=np.asarray(pts, dtype=np.float32),
            )
            for b, p, pts in zip(boxes, probs, points, strict=True)
            if p is not None and p >= self._min_prob
        ]
        return sorted(faces, key=lambda f: f.area, reverse=True)


def gend_crop(image: Image, points: Points, size: int = 224, scale: float = 1.3) -> Image:
    """Aligned face crop for GenD. Port of GenD's `align_face`."""
    dst = _GEND_TEMPLATE * size
    margin = size * (scale - 1) / 2.0
    dst = (dst + margin) * (size / (size + 2 * margin))
    matrix = cv2.estimateAffinePartial2D(points.astype(np.float32), dst, method=cv2.LMEDS)[0]
    if matrix is None:
        raise ValueError("could not estimate the face alignment transform")
    out: Image = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)
    return out


def sbi_crop(image: Image, box: tuple[float, float, float, float], size: int = 380) -> Image:
    """Box crop for SBI: the box plus w/8 and h/8 on each side, resized to size x size."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) / 8.0, (y1 - y0) / 8.0
    top, bottom = max(0, int(y0 - my)), min(height, int(y1 + my) + 1)
    left, right = max(0, int(x0 - mx)), min(width, int(x1 + mx) + 1)
    if bottom <= top or right <= left:
        raise ValueError("face box is outside the image")
    out: Image = cv2.resize(image[top:bottom, left:right], dsize=(size, size))
    return out


def _region_indices() -> dict[str, list[int]]:
    """Landmark indices per face region, read from MediaPipe's own connection tables."""
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

    def unique(*groups: Any) -> list[int]:
        return sorted({i for g in groups for c in g for i in (c.start, c.end)})

    # The oval connections form one closed chain; walk it to get polygon order.
    nxt = {c.start: c.end for c in C.FACE_LANDMARKS_FACE_OVAL}
    first = C.FACE_LANDMARKS_FACE_OVAL[0].start
    oval, cur = [first], nxt[first]
    while cur != first:
        oval.append(cur)
        cur = nxt[cur]
    return {
        "oval": oval,
        "left_eye": unique(C.FACE_LANDMARKS_LEFT_EYE),
        "right_eye": unique(C.FACE_LANDMARKS_RIGHT_EYE),
        "left_brow": unique(C.FACE_LANDMARKS_LEFT_EYEBROW),
        "right_brow": unique(C.FACE_LANDMARKS_RIGHT_EYEBROW),
        "lips": unique(C.FACE_LANDMARKS_LIPS),
        "nose": unique(C.FACE_LANDMARKS_NOSE),
    }


_REGIONS: dict[str, list[int]] | None = None


def region_indices() -> dict[str, list[int]]:
    global _REGIONS
    if _REGIONS is None:
        _REGIONS = _region_indices()
    return _REGIONS


class FaceGeometry:
    """Landmarks of one crop plus the masks derived from them. Masks are computed once."""

    def __init__(self, landmarks: Points, height: int, width: int) -> None:
        self.landmarks = landmarks
        self.height = height
        self.width = width

    def points(self, region: str) -> Points:
        pts: Points = self.landmarks[region_indices()[region]]
        return pts

    def _fill(self, pts: Points, convex: bool) -> NDArray[np.float32]:
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        poly = np.round(pts).astype(np.int32)
        if convex:
            cv2.fillConvexPoly(mask, cv2.convexHull(poly), 255)
        else:
            cv2.fillPoly(mask, [poly], 255)
        return mask.astype(np.float32) / 255.0

    def _feather(self, mask: NDArray[np.float32], frac: float) -> NDArray[np.float32]:
        sigma = max(0.5, frac * min(self.height, self.width))
        # The blur can overshoot 1.0 by float rounding; masks must stay in [0, 1].
        out: NDArray[np.float32] = np.clip(cv2.GaussianBlur(mask, (0, 0), sigma), 0.0, 1.0)
        return out

    @cached_property
    def face(self) -> NDArray[np.float32]:
        """Hard mask of the face oval."""
        return self._fill(self.points("oval"), convex=False)

    @cached_property
    def lips(self) -> NDArray[np.float32]:
        return self._feather(self._fill(self.points("lips"), convex=True), 0.008)

    @cached_property
    def skin(self) -> NDArray[np.float32]:
        """Soft mask: face oval minus eyes, brows and lips."""
        holes = np.zeros((self.height, self.width), dtype=np.float32)
        for region in ("left_eye", "right_eye", "left_brow", "right_brow", "lips"):
            holes = np.maximum(holes, self._fill(self.points(region), convex=True))
        k = max(3, int(0.03 * min(self.height, self.width)) | 1)
        holes = cv2.dilate(holes, np.ones((k, k), np.uint8))
        return self._feather(np.clip(self.face - holes, 0.0, 1.0), 0.012)

    @cached_property
    def background(self) -> NDArray[np.float32]:
        """Hard mask of everything well outside the face oval (hair and background)."""
        k = max(3, int(0.08 * min(self.height, self.width)) | 1)
        grown = cv2.dilate(self.face, np.ones((k, k), np.uint8))
        out: NDArray[np.float32] = 1.0 - grown
        return out


class Landmarker:
    """MediaPipe Face Landmarker (478 points) in single-image mode."""

    def __init__(self, model_path: Path) -> None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)), num_faces=1
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def geometry(self, crop: Image) -> FaceGeometry | None:
        """Landmarks for a face crop, or None when MediaPipe finds no face in it."""
        import mediapipe as mp

        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        result = self._landmarker.detect(frame)
        if not result.face_landmarks:
            return None
        height, width = crop.shape[:2]
        pts = np.array(
            [[p.x * width, p.y * height] for p in result.face_landmarks[0]], dtype=np.float32
        )
        return FaceGeometry(pts, height, width)
