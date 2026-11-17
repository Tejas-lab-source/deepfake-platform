"""End-to-end tests of the real pipeline on a sample image. Need the downloaded weights."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from tests.conftest import SBI, needs_gend, needs_sbi  # noqa: E402

pytestmark = needs_gend

HEADERS: dict[str, str] = {}


@pytest.fixture(scope="module")
def client(tmp_path_factory, photo):
    from fastapi.testclient import TestClient
    from PIL import Image

    from app import main

    uploads = tmp_path_factory.mktemp("uploads")
    Image.fromarray(photo).save(uploads / "face.png")
    Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8)).save(uploads / "blank.png")

    mp = pytest.MonkeyPatch()
    mp.setattr(main.settings, "upload_dir", uploads)
    mp.setattr(main.settings, "mock_mode", False)
    HEADERS["X-Internal-Secret"] = main.settings.internal_secret
    main.state.load()
    assert main.state.error is None
    yield TestClient(main.app), uploads
    main.state.pipeline.close()
    main.state.pipeline = None
    mp.undo()


def _post(client, route, body):
    return client[0].post(route, json=body, headers=HEADERS)


def test_ready_and_model_info(client):
    assert client[0].get("/ready", headers=HEADERS).json() == {"ready": True, "mockMode": False}
    info = client[0].get("/model-info", headers=HEADERS).json()
    assert info["gend"] == "yermandy/GenD_CLIP_L_14" and info["device"] in {"cpu", "cuda"}
    assert info["sbiLoaded"] == SBI.is_file()


def test_gend_feature_and_logit(client, crop):
    from app import main
    from app.detectors.gend import FEATURE_DIM

    gend = main.state.pipeline.gend
    features, logits = gend.score([crop, crop])
    assert features.shape == (2, FEATURE_DIM) and logits.shape == (2,)
    assert np.allclose(features[0], features[1], atol=1e-4)
    # The logit is exactly the checkpoint's own head applied to the returned feature.
    import torch

    again = gend.model.logit_from_features(torch.from_numpy(features)).numpy()
    assert np.allclose(again, logits, atol=1e-4)


def test_analyze_image_full(client):
    r = _post(client, "/analyze/image", {"path": "face.png"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in {"FAKE", "REAL", "UNCERTAIN"}
    assert 0.0 <= body["probability"] <= 1.0
    assert (body["verdict"] == "UNCERTAIN") == (body["uncertainReason"] is not None)
    assert set(body["timingsMs"]) == {"detect", "beautify", "score", "explain"}
    assert "mock" not in body

    face = body["faces"][0]
    assert len(face["box"]) == 4 and face["landmarksFound"]
    assert set(face["variantLogits"]) == {"clean", "F1", "F2", "F3", "mix"}
    values = list(face["variantLogits"].values())
    assert face["spread"] == pytest.approx(max(values) - min(values), abs=1e-3)
    assert face["scores"]["gendHead"] == pytest.approx(np.mean(values), abs=1e-3)
    assert 0.0 <= face["beautifiedLikelihood"] <= 1.0
    assert 0.0 <= face["heatmapIoU"] <= 1.0
    for key in (face["heatmaps"]["gend"], face["crops"]["gend"], face["crops"]["sbi"]):
        assert (client[1] / key).is_file()

    # Same file, same copies, same scores.
    second = _post(client, "/analyze/image", {"path": "face.png", "explain": False}).json()
    assert second["faces"][0]["variantLogits"] == face["variantLogits"]
    assert second["faces"][0]["heatmaps"] is None


@needs_sbi
def test_both_detectors_score_and_explain(client):
    body = _post(client, "/analyze/image", {"path": "face.png"}).json()
    face = body["faces"][0]
    sbi, gend = face["scores"]["sbi"], face["scores"]["gendHead"]
    assert isinstance(sbi, float) and isinstance(gend, float)
    assert np.isfinite([sbi, gend]).all()
    assert (client[1] / face["heatmaps"]["sbi"]).is_file()
    # With the default identity fusion, the probability is sigmoid(mean GenD logit): SBI can
    # only change the verdict through the detectors_disagree rule.
    assert body["probability"] == pytest.approx(1 / (1 + np.exp(-gend)), abs=1e-3)
    reasons = {None, "near_threshold", "unstable_under_beautification", "variants_disagree"}
    disagree = (gend >= 0) != (sbi >= 0) and abs(
        1 / (1 + np.exp(-gend)) - 1 / (1 + np.exp(-sbi))
    ) > 0.6
    if body["uncertainReason"] not in reasons - {None}:
        assert (body["uncertainReason"] == "detectors_disagree") == disagree

    cmp = _post(client, "/compare", {"path": "face.png", "family": "F1", "strength": 0.8}).json()
    assert isinstance(cmp["clean"]["scores"]["sbi"], float)
    assert isinstance(cmp["filtered"]["scores"]["sbi"], float)
    assert cmp["clean"]["scores"]["sbi"] == pytest.approx(sbi, abs=1e-3)


def test_fast_mode_k0(client):
    body = _post(client, "/analyze/image", {"path": "face.png", "k": 0, "explain": False}).json()
    face = body["faces"][0]
    assert set(face["variantLogits"]) == {"clean"} and face["spread"] == 0


def test_no_face_is_uncertain_not_real(client):
    body = _post(client, "/analyze/image", {"path": "blank.png"}).json()
    assert body["verdict"] == "UNCERTAIN" and body["uncertainReason"] == "no_face_detected"
    assert body["faces"] == [] and body["probability"] is None


@pytest.mark.parametrize("family", ("F1", "F3"))
def test_compare(client, family):
    r = _post(client, "/compare", {"path": "face.png", "family": family, "strength": 0.8})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["flipped"] == (body["clean"]["verdict"] != body["filtered"]["verdict"])
    assert 0.0 <= body["heatmapIoU"] <= 1.0
    for side in ("clean", "filtered"):
        assert (client[1] / body[side]["image"]).is_file()
        assert (client[1] / body[side]["heatmap"]).is_file()


def test_compare_at_zero_strength_changes_nothing(client):
    body = _post(client, "/compare", {"path": "face.png", "family": "F2", "strength": 0}).json()
    assert body["flipped"] is False and body["heatmapIoU"] == 1.0
    assert body["clean"]["probability"] == body["filtered"]["probability"]


def test_beautify_returns_a_png(client):
    import io

    from PIL import Image

    r = _post(client, "/beautify", {"path": "face.png", "family": "F2", "strength": 0.7})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).size == (224, 224)


def test_filter_errors(client):
    assert (
        _post(
            client, "/beautify", {"path": "face.png", "family": "F9", "strength": 0.5}
        ).status_code
        == 422
    )
    assert (
        _post(client, "/beautify", {"path": "face.png", "family": "F1", "strength": 2}).status_code
        == 422
    )
    assert (
        _post(
            client, "/compare", {"path": "blank.png", "family": "F1", "strength": 0.5}
        ).status_code
        == 422
    )
    # F4 is optional and not installed here: a clear 501, not a crash.
    assert (
        _post(
            client, "/beautify", {"path": "face.png", "family": "F4", "strength": 0.5}
        ).status_code
        == 501
    )
    # Real video analysis exists since Phase 5; an image sent as a video is refused clearly.
    not_video = _post(client, "/analyze/video", {"path": "face.png"})
    assert not_video.status_code == 422 and "not a supported video" in not_video.json()["detail"]
    assert Path(client[1] / "derived").is_dir()


# ---- /analyze/video with the real models -------------------------------------------------


def _face_video(uploads: Path, photo, pattern: str, name: str, fps: float = 4.0) -> str:
    """A video whose frames follow `pattern`: "F" = the portrait, "." = a blank grey frame."""
    import cv2

    blank = np.full_like(photo, 128)
    path = uploads / name
    h, w = photo.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    for c in pattern:
        writer.write(cv2.cvtColor(photo if c == "F" else blank, cv2.COLOR_RGB2BGR))
    writer.release()
    return name


def test_analyze_video_real_models(client, photo):
    from app import main

    pattern = "FF.F.FF."
    name = _face_video(client[1], photo, pattern, "clip.mp4")
    body = {"path": name, "k": 0, "maxFrames": 8, "progressId": "real-video"}
    r = _post(client, "/analyze/video", body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert "mock" not in out and out["kind"] == "video"
    assert out["video"]["frameCount"] == len(pattern) and out["framesSampled"] == len(pattern)

    frames = out["frames"]
    assert [f["index"] for f in frames] == list(range(len(pattern)))
    assert [f["timestamp"] for f in frames] == [i / 4.0 for i in range(len(pattern))]
    for f, c in zip(frames, pattern, strict=True):
        # Blank frames have no face: excluded, never scored as real.
        assert f["hasFace"] is (c == "F"), f
        assert (f["logit"] is None) is (c != "F")
    assert out["framesScored"] == pattern.count("F")

    # The verdict is the top-k mean of the face frames' logits, through the decision rules.
    logits = [f["logit"] for f in frames if f["logit"] is not None]
    k = min(main.settings.video_top_k, len(logits))
    assert out["aggregate"]["k"] == k
    assert out["aggregate"]["logit"] == pytest.approx(
        sum(sorted(logits, reverse=True)[:k]) / k, abs=1e-3
    )
    assert out["verdict"] in {"FAKE", "REAL", "UNCERTAIN"}
    most, least = out["mostSuspicious"], out["leastSuspicious"]
    assert most["logit"] == max(logits) and least["logit"] == min(logits)
    assert pattern[most["index"]] == "F" and pattern[least["index"]] == "F"

    # Only the most and least suspicious frames are explained, with heatmaps on disk.
    explained = [f for f in frames if f["explained"]]
    assert {f["index"] for f in explained} == {most["index"], least["index"]}
    for f in explained:
        maps = f["faces"][0]["heatmaps"]
        assert (client[1] / maps["gend"]).is_file()
        if SBI.is_file():
            assert (client[1] / maps["sbi"]).is_file()
    assert out["faces"] and out["faces"][0]["scores"]["gendHead"] == pytest.approx(
        most["logit"], abs=1e-3
    )
    assert set(out["timingsMs"]) == {"decode", "detect", "beautify", "score", "explain"}
    progress = client[0].get("/progress/real-video", headers=HEADERS).json()
    assert progress["stage"] == "aggregating" and progress["done"] == progress["total"]


def test_video_without_faces_is_uncertain_not_real(client, photo):
    name = _face_video(client[1], photo, "....", "blank.mp4")
    out = _post(client, "/analyze/video", {"path": name, "k": 0}).json()
    assert out["verdict"] == "UNCERTAIN" and out["uncertainReason"] == "no_face_detected"
    assert out["probability"] is None and out["framesScored"] == 0
    assert out["faces"] == [] and all(not f["hasFace"] for f in out["frames"])
