"""Video decoding, sampling, timestamps and aggregation, on synthetic videos made here.

The real-model video tests (actual face frames, both detectors, Grad-CAM) are in
test_real_pipeline.py, which loads the weights once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402
from app.head import Calibration  # noqa: E402
from app.media import MediaError, validate  # noqa: E402
from app.pipeline import decide, top_k_mean  # noqa: E402
from app.progress import ProgressStore  # noqa: E402
from app.video import aggregate_frames, probe, read_frames, sample_indices, timestamp  # noqa: E402

HEADERS = {"X-Internal-Secret": main.settings.internal_secret}
CAL = Calibration()

FORMATS = {"mp4": ("mp4v", ".mp4"), "webm": ("VP80", ".webm"), "mov": ("mp4v", ".mov")}


def write_video(
    path: Path, frames: list[np.ndarray], fps: float = 10.0, fourcc: str = "mp4v"
) -> Path:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    assert writer.isOpened(), f"OpenCV cannot write {fourcc}"
    for rgb in frames:
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    return path


BARS = 8


def numbered_frames(n: int, size: tuple[int, int] = (64, 48)) -> list[np.ndarray]:
    """Frame i shows its index in binary as 8 black/white vertical bars, which survive lossy
    compression, so a decoded frame tells which frame it was."""
    w, h = size
    out = []
    for i in range(n):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for bit in range(BARS):
            if i >> bit & 1:
                rgb[:, bit * w // BARS : (bit + 1) * w // BARS] = 255
        out.append(rgb)
    return out


def frame_number(rgb: np.ndarray) -> int:
    w = rgb.shape[1]
    bars = [rgb[:, b * w // BARS + 2 : (b + 1) * w // BARS - 2].mean() > 128 for b in range(BARS)]
    return sum(1 << b for b, on in enumerate(bars) if on)


# ---- sampling and timestamps -----------------------------------------------------------


def test_sample_indices_are_even_and_bounded():
    assert sample_indices(5, 32) == [0, 1, 2, 3, 4]
    assert sample_indices(100, 10) == [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    picked = sample_indices(1000, 32)
    assert len(picked) == 32 and picked == sorted(set(picked))
    assert 0 <= picked[0] and picked[-1] <= 999
    gaps = np.diff(picked)
    assert gaps.max() - gaps.min() <= 1  # even spacing
    assert sample_indices(0, 10) == [] and sample_indices(10, 0) == []
    assert sample_indices(1, 32) == [0]


def test_timestamps_are_index_over_fps():
    assert timestamp(0, 25.0) == 0.0
    assert timestamp(30, 10.0) == 3.0
    assert timestamp(1, 3.0) == 0.333
    assert timestamp(5, 0.0) is None  # unknown frame rate: no invented timestamp


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_formats_decode_with_frame_count_and_fps(tmp_path, fmt):
    fourcc, ext = FORMATS[fmt]
    path = write_video(tmp_path / f"v{ext}", numbered_frames(30), fps=10.0, fourcc=fourcc)
    assert validate(path, "video", 100) in {"mp4", "webm"}  # MOV is ISO-BMFF ("ftyp")
    info = probe(path, max_decode_frames=1000)
    assert info.frame_count == 30
    assert info.fps == pytest.approx(10.0, rel=0.01)
    assert (info.width, info.height) == (64, 48)
    assert info.duration_s == pytest.approx(3.0, rel=0.01)


def test_read_frames_returns_exactly_the_sampled_frames_in_order(tmp_path):
    path = write_video(tmp_path / "v.mp4", numbered_frames(40), fps=20.0)
    indices = sample_indices(40, 8)
    got = list(read_frames(path, indices, max_side=1600))
    assert [i for i, _ in got] == indices
    for index, rgb in got:
        assert rgb.shape == (48, 64, 3) and rgb.dtype == np.uint8
        assert frame_number(rgb) == index  # the decoded pixels are that very frame


def test_read_frames_downscales_large_frames(tmp_path):
    path = write_video(tmp_path / "v.mp4", numbered_frames(3, size=(320, 240)), fps=5.0)
    (_, rgb), *_ = read_frames(path, [0], max_side=160)
    assert max(rgb.shape[:2]) == 160


# ---- invalid videos --------------------------------------------------------------------


def test_corrupt_video_is_refused(tmp_path):
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"\0\0\0\x18ftypmp42" + b"\0" * 4 + np.random.default_rng(0).bytes(4096))
    with pytest.raises(MediaError, match="decod"):
        probe(path, max_decode_frames=1000)


def test_empty_video_is_refused(tmp_path):
    path = tmp_path / "empty.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    writer.release()  # a container with no frames (or no file at all)
    if not path.exists():
        path.write_bytes(b"\0\0\0\x18ftypmp42" + b"\0" * 8)
    with pytest.raises(MediaError):
        probe(path, max_decode_frames=1000)


def test_too_many_frames_is_refused(tmp_path):
    path = write_video(tmp_path / "v.mp4", numbered_frames(30))
    with pytest.raises(MediaError, match="too long"):
        probe(path, max_decode_frames=20)


# ---- aggregation -----------------------------------------------------------------------


def frame(index: int, logit: float | None, sbi: float | None = None) -> dict:
    return {
        "index": index,
        "timestamp": index / 10,
        "hasFace": logit is not None,
        "logit": logit,
        "sbi": sbi,
    }


def test_frames_without_faces_are_excluded_never_real():
    out = aggregate_frames([frame(0, None), frame(1, None)], CAL, top_k=5)
    assert out["verdict"] == "UNCERTAIN" and out["uncertainReason"] == "no_face_detected"
    assert out["probability"] is None and out["framesScored"] == 0 and out["framesSampled"] == 2

    mixed = [frame(0, None), frame(1, 2.5), frame(2, None), frame(3, 3.0)]
    out = aggregate_frames(mixed, CAL, top_k=5)
    assert out["framesScored"] == 2
    assert out["aggregate"]["logit"] == pytest.approx(2.75)  # the faceless frames do not count


def test_top_k_mean_aggregation_and_extreme_frames():
    logits = [-1.0, 4.0, 0.5, 3.0, -2.0, 2.0]
    frames = [frame(i, v, sbi=v / 2) for i, v in enumerate(logits)]
    out = aggregate_frames(frames, CAL, top_k=3)
    assert out["aggregate"] == {
        "method": "top_k_mean",
        "k": 3,
        "logit": pytest.approx(top_k_mean(logits, 3)),
        "sbi": pytest.approx((4.0 + 3.0 + 2.0) / 2 / 3),
    }
    expected = decide(CAL, {"clean": top_k_mean(logits, 3)}, (4.0 + 3.0 + 2.0) / 6, 0.0)
    assert out["verdict"] == expected.verdict
    assert out["probability"] == round(expected.probability, 4)
    assert out["mostSuspicious"] == {"index": 1, "timestamp": 0.1, "logit": 4.0}
    assert out["leastSuspicious"] == {"index": 4, "timestamp": 0.4, "logit": -2.0}
    # k larger than the scored frames uses them all.
    assert aggregate_frames(frames[:2], CAL, top_k=10)["aggregate"]["k"] == 2


# ---- API (mock mode) -------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(main.settings, "mock_mode", True)
    monkeypatch.setattr(main, "progress_store", ProgressStore())
    return TestClient(main.app), tmp_path


def test_mock_video_has_real_sampling_timestamps_and_progress(client):
    c, uploads = client
    write_video(uploads / "clip.mp4", numbered_frames(50), fps=25.0)
    body = {"path": "clip.mp4", "maxFrames": 10, "progressId": "an-1"}
    r = c.post("/analyze/video", json=body, headers=HEADERS)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["mock"] is True and out["kind"] == "video"
    assert out["video"]["frameCount"] == 50 and out["video"]["sampledFrames"] == 10
    assert [f["index"] for f in out["frames"]] == sample_indices(50, 10)
    assert all(f["timestamp"] == round(f["index"] / 25.0, 3) for f in out["frames"])
    assert out["framesSampled"] == 10
    assert out["framesScored"] == sum(1 for f in out["frames"] if f["hasFace"])
    # The same file always gives the same mock result.
    again = c.post("/analyze/video", json=body, headers=HEADERS).json()
    out.pop("timingsMs"), again.pop("timingsMs")
    assert out == again
    progress = c.get("/progress/an-1", headers=HEADERS).json()
    assert progress == {"stage": "aggregating", "done": 0, "total": 1, "unit": "frames"}


def test_max_frames_cannot_exceed_the_service_limit(client, monkeypatch):
    c, uploads = client
    monkeypatch.setattr(main.settings, "max_frames", 4)
    write_video(uploads / "clip.mp4", numbered_frames(20))
    out = c.post("/analyze/video", json={"path": "clip.mp4", "maxFrames": 200}, headers=HEADERS)
    assert out.json()["framesSampled"] == 4


def test_bad_videos_get_clear_422s(client):
    c, uploads = client
    (uploads / "corrupt.mp4").write_bytes(b"\0\0\0\x18ftypmp42" + b"\x07" * 2048)
    (uploads / "notes.mp4").write_text("not a video")
    write_video(uploads / "clip.mp4", numbered_frames(3))
    corrupt = c.post("/analyze/video", json={"path": "corrupt.mp4"}, headers=HEADERS)
    assert corrupt.status_code == 422 and "decod" in corrupt.json()["detail"]
    assert c.post("/analyze/video", json={"path": "notes.mp4"}, headers=HEADERS).status_code == 422
    assert c.post("/analyze/video", json={"path": "../x.mp4"}, headers=HEADERS).status_code == 422
    assert (
        c.post("/analyze/video", json={"path": "missing.mp4"}, headers=HEADERS).status_code == 422
    )
    bad_id = {"path": "clip.mp4", "progressId": "../../etc"}
    assert c.post("/analyze/video", json=bad_id, headers=HEADERS).status_code == 422


def test_oversized_video_is_refused(client, monkeypatch):
    c, uploads = client
    monkeypatch.setattr(main.settings, "max_video_mb", 0)
    write_video(uploads / "clip.mp4", numbered_frames(3))
    r = c.post("/analyze/video", json={"path": "clip.mp4"}, headers=HEADERS)
    assert r.status_code == 422 and "limit" in r.json()["detail"]


def test_progress_endpoint(client):
    c, _ = client
    assert c.get("/progress/unknown", headers=HEADERS).status_code == 404
    assert c.get("/progress/bad%20id", headers=HEADERS).status_code == 422
    assert c.get("/progress/unknown").status_code == 401
    main.progress_store.set("p1", "scoring", 3, 8, "frames")
    assert c.get("/progress/p1", headers=HEADERS).json() == {
        "stage": "scoring",
        "done": 3,
        "total": 8,
        "unit": "frames",
    }


def test_progress_store_expires_and_bounds_entries():
    store = ProgressStore(ttl_s=0.0, max_entries=3)
    store.set("a", "scoring", 1, 2, "frames")
    assert store.get("a") is None  # expired immediately
    store = ProgressStore(ttl_s=3600, max_entries=3)
    for i in range(10):
        store.set(f"id{i}", "scoring", i, 10, "frames")
    assert len(store._items) <= 3 and store.get("id9") is not None
