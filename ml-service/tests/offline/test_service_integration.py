"""The service with a trained head and model card (Phase 2 integration fixes)."""

import shutil

import numpy as np
import pytest

from tests.conftest import WEIGHTS, needs_gend

pytestmark = needs_gend


@pytest.fixture(scope="module")
def gend():
    import torch

    from app.detectors.gend import GenDDetector

    return GenDDetector(WEIGHTS, torch.device("cpu"))


def test_gradcam_targets_the_loaded_head(gend, crop):
    from app.explain import gradcam_gend
    from app.head import LinearHead
    from offline.heads import gend_head_from_checkpoint

    w0, b0 = gend_head_from_checkpoint(WEIGHTS / "gend")
    own = gradcam_gend(gend, [crop])
    same = gradcam_gend(gend, [crop], LinearHead(w0, b0, True, "x"))
    assert np.allclose(own, same, atol=1e-4)  # GenD's own head through the wrapper: same map
    other = LinearHead(-w0, -b0, True, "x")  # the opposite head: a different explanation
    assert not np.allclose(own, gradcam_gend(gend, [crop], other), atol=1e-2)


def _weights_dir(tmp_path, heads_dir, card_path):
    d = tmp_path / "weights"
    d.mkdir()
    for sub in ("gend", "clip-vit-large-patch14", "mediapipe", "sbi"):
        if (WEIGHTS / sub).exists():
            (d / sub).symlink_to((WEIGHTS / sub).resolve())
    shutil.copy(heads_dir / "weights" / "head.pt", d / "head.pt")
    shutil.copy(card_path, d / "model_card.json")
    return d


def test_service_refuses_the_fixture_card_unless_allowed(tmp_path, heads_dir, card_path):
    from app.config import Settings
    from app.head import ModelFilesError, load_calibration
    from app.pipeline import RealPipeline

    d = _weights_dir(tmp_path, heads_dir, card_path)
    cal = load_calibration(d / "model_card.json")
    with pytest.raises(ModelFilesError, match="plumbing fixture"):
        RealPipeline(Settings(weights_dir=d, upload_dir=tmp_path / "u", mock_mode=False), cal)


def test_service_uses_the_trained_head(tmp_path, heads_dir, card_path, photo):
    from PIL import Image

    from app.config import Settings
    from app.head import load_calibration
    from app.pipeline import RealPipeline

    d = _weights_dir(tmp_path, heads_dir, card_path)
    uploads = tmp_path / "u"
    uploads.mkdir()
    Image.fromarray(photo).save(uploads / "face.png")
    settings = Settings(
        weights_dir=d, upload_dir=uploads, mock_mode=False, allow_unvalidated_model_card=True
    )
    cal = load_calibration(d / "model_card.json")
    pipe = RealPipeline(settings, cal)
    try:
        info = pipe.info()
        assert info["headLoaded"] and info["headSha256"] == cal.head_sha256
        result = pipe.analyze_image(uploads / "face.png", 0, False)
        assert result["modelCardVersion"] == cal.version
        clean = result["faces"][0]["variantLogits"]["clean"]
        # The reported logit is the trained head's, not GenD's own.
        rgb, _ = pipe._load(uploads / "face.png")
        g, _ = pipe._crops(rgb, pipe.detector.detect(rgb)[0])
        features, gend_logit = pipe.gend.score([g])
        assert clean == pytest.approx(float(pipe.head.logits(features)[0]), abs=1e-3)
        assert abs(clean - float(gend_logit[0])) > 1e-3
    finally:
        pipe.close()
