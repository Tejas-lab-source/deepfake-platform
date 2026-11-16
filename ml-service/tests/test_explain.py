import pytest

np = pytest.importorskip("numpy")

from tests.conftest import SBI, needs_ml, needs_sbi  # noqa: E402

pytestmark = needs_ml


def test_iou_identical_disjoint_and_partial():
    from app.explain import top_region_iou

    a = np.zeros((10, 10), dtype=np.float32)
    a[:2, :] = 1.0  # exactly the top 20 %
    assert top_region_iou(a, a) == 1.0
    b = np.zeros((10, 10), dtype=np.float32)
    b[8:, :] = 1.0
    assert top_region_iou(a, b) == 0.0
    c = np.zeros((10, 10), dtype=np.float32)
    c[1:3, :] = 1.0  # shares one of two rows: 10 / 30
    assert top_region_iou(a, c) == pytest.approx(1 / 3)


def test_iou_uses_rank_not_scale_and_checks_inputs():
    from app.explain import top_region_iou

    rng = np.random.default_rng(1)
    a = rng.random((16, 16)).astype(np.float32)
    assert top_region_iou(a, a * 0.1 + 3.0) == 1.0
    with pytest.raises(ValueError):
        top_region_iou(a, a[:8])
    with pytest.raises(ValueError):
        top_region_iou(a, a, top=1.5)


def test_vit_reshape_transform_drops_the_class_token():
    import torch

    from app.explain import vit_reshape_transform

    tokens = torch.arange(2 * 257 * 3, dtype=torch.float32).reshape(2, 257, 3)
    grid = vit_reshape_transform(tokens)
    assert grid.shape == (2, 3, 16, 16)
    assert torch.equal(grid[0, :, 0, 0], tokens[0, 1])
    assert torch.equal(grid[1, :, 15, 15], tokens[1, 256])
    with pytest.raises(ValueError):
        vit_reshape_transform(tokens[:, :200])


def test_single_logit_target_and_colourise():
    import torch

    from app.explain import SingleLogitTarget, colourise

    assert SingleLogitTarget()(torch.tensor([2.5])).item() == 2.5
    rgb = colourise(np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4))
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8


def test_sbi_wrapper_shapes_and_gradcam_on_untrained_net(photo, face):
    """Checks the SBI plumbing only. The weights here are random, so scores mean nothing."""
    import torch

    from app.detectors.sbi import SBIDetector, SBIModel
    from app.explain import gradcam_sbi
    from app.faces import sbi_crop

    torch.manual_seed(0)
    sbi = SBIDetector.__new__(SBIDetector)
    sbi.model = SBIModel().eval().requires_grad_(False)
    sbi.device = torch.device("cpu")
    crop = sbi_crop(photo, face.box)
    assert sbi.score([crop]).shape == (1,)
    with pytest.raises(ValueError):
        sbi.preprocess([crop[:100]])
    cam = gradcam_sbi(sbi, [crop])
    assert cam.shape == (1, 380, 380) and 0.0 <= cam.min() and cam.max() <= 1.0
    assert not any(p.requires_grad for p in sbi.model.parameters())


def test_sbi_checkpoint_format_round_trip(tmp_path):
    """SBIDetector reads the official layout: torch.save({'model': state_dict})."""
    import torch

    from app.detectors.sbi import SBIDetector, SBIModel

    path = tmp_path / "FFc23.tar"
    torch.save({"model": SBIModel().state_dict()}, path)
    assert SBIDetector(path, torch.device("cpu")).model.net._fc.out_features == 2


def test_sbi_loader_rejects_pickled_code_and_wrong_files(tmp_path):
    """weights_only=True: a checkpoint that smuggles a Python object must not load."""
    import pickle

    import torch

    from app.detectors.sbi import SBIDetector

    bad = tmp_path / "evil.tar"
    torch.save({"model": {}, "extra": print}, bad)
    with pytest.raises(pickle.UnpicklingError):
        SBIDetector(bad, torch.device("cpu"))
    plain = tmp_path / "plain.tar"
    torch.save({"weights": {}}, plain)
    with pytest.raises(ValueError, match="not an SBI checkpoint"):
        SBIDetector(plain, torch.device("cpu"))


@needs_sbi
def test_official_sbi_checkpoint_loads_and_scores(photo, face):
    import torch

    from app.detectors.sbi import SBIDetector
    from app.explain import gradcam_sbi
    from app.faces import sbi_crop

    sbi = SBIDetector(SBI, torch.device("cpu"))
    crop = sbi_crop(photo, face.box)
    logits = sbi.score([crop, crop])
    assert logits.shape == (2,) and np.isfinite(logits).all()
    assert logits[0] == pytest.approx(logits[1], abs=1e-4)
    # A plain studio portrait of a real person: SBI should lean "not fake".
    assert logits[0] < 0
    cam = gradcam_sbi(sbi, [crop])
    assert cam.shape == (1, 380, 380) and cam.max() > 0


def test_linear_head_math(tmp_path):
    import torch

    from app.head import LinearHead

    assert LinearHead.load(tmp_path / "head.pt") is None
    torch.save(
        {"weight": torch.tensor([3.0, 4.0]), "bias": 0.5, "l2_normalize": True},
        tmp_path / "head.pt",
    )
    head = LinearHead.load(tmp_path / "head.pt")
    assert head is not None
    # (3, 4) normalised is (0.6, 0.8): 3*0.6 + 4*0.8 + 0.5 = 5.5, whatever the input's length.
    out = head.logits(np.array([[3.0, 4.0], [30.0, 40.0]]))
    assert out == pytest.approx([5.5, 5.5])
