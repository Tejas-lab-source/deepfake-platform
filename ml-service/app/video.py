"""Video decoding, frame sampling and frame aggregation (video path of the pipeline).

Decoding uses OpenCV (FFmpeg backend), which reads the formats the upload API accepts: MP4,
WebM and QuickTime (MOV). Frames are sampled evenly: the video is split into `max_frames`
equal segments and the middle frame of each is taken, so a short video gives every frame and
a long one gives an even spread. Each sampled frame keeps its index and its timestamp
(index / fps, in seconds).

The aggregation is pure and shared by the mock and the real path:
  - frames without a detected face are excluded, never scored as real;
  - the video logit is the mean of the top-k frame logits (`top_k_mean`);
  - the verdict goes through the same `decide` rules as an image;
  - the most and least suspicious frames are the highest and lowest frame logits.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.head import Calibration
from app.media import MediaError
from app.pipeline import decide, top_k_mean

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": round(self.fps, 3),
            "frameCount": self.frame_count,
            "durationSec": round(self.duration_s, 3),
            "width": self.width,
            "height": self.height,
        }


def _open(path: Path) -> Any:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise MediaError("the video could not be decoded (corrupt or unsupported codec)")
    return cap


def probe(path: Path, max_decode_frames: int) -> VideoInfo:
    """Read the video's metadata; count frames by decoding when the container does not say.

    Raises MediaError for files that cannot be decoded or have no frames.
    """
    import cv2

    cap = _open(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Containers can omit or misstate the count (WebM often reports nothing): count the
        # frames that actually decode, up to the limit. grab() does not decode pixels.
        counted = 0
        while counted < max_decode_frames + 1 and cap.grab():
            counted += 1
    finally:
        cap.release()
    if counted == 0:
        raise MediaError("the video has no decodable frames")
    if counted > max_decode_frames:
        raise MediaError(f"the video is too long (more than {max_decode_frames} frames)")
    if not (0.0 < fps <= 1000.0):
        fps = 0.0
    del reported  # the decoded count is authoritative
    return VideoInfo(fps=fps, frame_count=counted, width=width, height=height)


def sample_indices(frame_count: int, max_frames: int) -> list[int]:
    """Evenly spaced frame indices: the middle of each of `max_frames` equal segments."""
    if frame_count <= 0 or max_frames <= 0:
        return []
    if frame_count <= max_frames:
        return list(range(frame_count))
    step = frame_count / max_frames
    return sorted({min(frame_count - 1, int((i + 0.5) * step)) for i in range(max_frames)})


def timestamp(index: int, fps: float) -> float | None:
    return round(index / fps, 3) if fps > 0 else None


def read_frames(
    path: Path, indices: list[int], max_side: int
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Decode only the sampled frames, in order, as RGB (downscaled to `max_side`)."""
    import cv2

    wanted = set(indices)
    last = max(indices) if indices else -1
    cap = _open(path)
    try:
        index = 0
        while index <= last and cap.grab():
            if index in wanted:
                ok, bgr = cap.retrieve()
                if ok and bgr is not None:
                    h, w = bgr.shape[:2]
                    longest = max(h, w)
                    if longest > max_side:
                        scale = max_side / longest
                        bgr = cv2.resize(
                            bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
                        )
                    yield index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        cap.release()


def aggregate_frames(frames: list[dict[str, Any]], cal: Calibration, top_k: int) -> dict[str, Any]:
    """Video verdict from per-frame results.

    Each frame dict has `index`, `timestamp`, `hasFace` and, for frames with a face, `logit`
    (the deciding face's head logit) and optionally `sbi` and `beautifiedLikelihood`.
    """
    scored = [f for f in frames if f.get("hasFace") and f.get("logit") is not None]
    base: dict[str, Any] = {
        "threshold": cal.threshold,
        "framesSampled": len(frames),
        "framesScored": len(scored),
    }
    if not scored:
        return {
            **base,
            "verdict": "UNCERTAIN",
            "probability": None,
            "confident": False,
            "uncertainReason": "no_face_detected",
            "aggregate": None,
            "mostSuspicious": None,
            "leastSuspicious": None,
        }
    logits = [float(f["logit"]) for f in scored]
    k = max(1, min(top_k, len(scored)))
    agg = top_k_mean(logits, k)
    # The detectors' disagreement and the smoothing cue use the same top-k frames.
    top = sorted(scored, key=lambda f: float(f["logit"]), reverse=True)[:k]
    sbis = [float(f["sbi"]) for f in top if f.get("sbi") is not None]
    sbi = sum(sbis) / len(sbis) if sbis else None
    likes = [
        float(f["beautifiedLikelihood"]) for f in top if f.get("beautifiedLikelihood") is not None
    ]
    likelihood = sum(likes) / len(likes) if likes else 0.0
    decision = decide(cal, {"clean": agg}, sbi, likelihood)
    most = max(scored, key=lambda f: float(f["logit"]))
    least = min(scored, key=lambda f: float(f["logit"]))

    def ref(f: dict[str, Any]) -> dict[str, Any]:
        return {"index": f["index"], "timestamp": f.get("timestamp"), "logit": f["logit"]}

    return {
        **base,
        "verdict": decision.verdict,
        "probability": round(decision.probability, 4),
        "confident": decision.confident,
        "uncertainReason": decision.uncertain_reason,
        "aggregate": {
            "method": "top_k_mean",
            "k": k,
            "logit": round(agg, 4),
            "sbi": None if sbi is None else round(sbi, 4),
        },
        "mostSuspicious": ref(most),
        "leastSuspicious": ref(least),
    }
