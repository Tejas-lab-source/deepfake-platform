import pytest

np = pytest.importorskip("numpy")

from tests.conftest import needs_landmarker  # noqa: E402

pytestmark = needs_landmarker

FAMILIES = ("F1", "F2", "F3")


def _rng():
    return np.random.default_rng(0)


def _mad(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def test_crops_and_landmarks(photo, face, crop, geom):
    from app.faces import sbi_crop

    assert crop.shape == (224, 224, 3) and crop.dtype == np.uint8
    assert sbi_crop(photo, face.box).shape == (380, 380, 3)
    assert geom.landmarks.shape == (478, 2)
    # Aligned crop: the eyes sit above the lips, and the skin mask is a sensible size.
    assert geom.points("left_eye")[:, 1].mean() < geom.points("lips")[:, 1].mean()
    assert 0.1 < float(geom.skin.mean()) < 0.5
    assert geom.skin.min() >= 0.0 and geom.skin.max() <= 1.0


def test_skin_mask_excludes_eyes_brows_and_lips(geom):
    for region in ("left_eye", "right_eye", "lips"):
        cx, cy = geom.points(region).mean(0).round().astype(int)
        assert geom.skin[cy, cx] < 0.05


@pytest.mark.parametrize("family", FAMILIES)
def test_strength_zero_leaves_the_image_unchanged(family, crop, geom):
    from app import beautify

    out, _ = beautify.apply_family(family, crop, geom, 0.0, _rng())
    assert np.array_equal(out, crop)
    assert out is not crop


@pytest.mark.parametrize("family", FAMILIES)
def test_output_shape_dtype_and_visible_change(family, crop, geom):
    from app import beautify

    out, _ = beautify.apply_family(family, crop, geom, 1.0, _rng())
    assert out.shape == crop.shape and out.dtype == np.uint8
    assert _mad(out, crop) > 0.3


@pytest.mark.parametrize("family", FAMILIES)
def test_identity_preserved(family, crop, geom, landmarker):
    """Even at full strength the change is small and the face stays where it was."""
    from app import beautify

    out, _ = beautify.apply_family(family, crop, geom, 1.0, _rng())
    assert _mad(out, crop) < 10.0
    after = landmarker.geometry(out)
    assert after is not None
    shift = np.linalg.norm(after.landmarks - geom.landmarks, axis=1)
    assert float(shift.mean()) < 0.03 * crop.shape[0]


@pytest.mark.parametrize("family", ("F1", "F2"))
def test_change_grows_with_strength_and_stays_on_the_face(family, crop, geom):
    from app import beautify

    weak, _ = beautify.apply_family(family, crop, geom, 0.3, _rng())
    strong, _ = beautify.apply_family(family, crop, geom, 1.0, _rng())
    assert _mad(weak, crop) < _mad(strong, crop)
    background = geom.background > 0.5
    assert np.array_equal(strong[background], crop[background])


@pytest.mark.parametrize("method", ("bilateral", "guided", "gaussian"))
def test_f1_methods_reduce_skin_detail(method, crop, geom):
    import cv2

    from app.beautify import f1_smooth

    out = f1_smooth.apply(crop, geom, 1.0, _rng(), method=method)

    def detail(img):
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return float((np.abs(cv2.Laplacian(grey, cv2.CV_32F)) * geom.skin).sum())

    assert detail(out) < detail(crop)


def test_f1_rejects_unknown_method(crop, geom):
    from app.beautify import f1_smooth

    with pytest.raises(ValueError):
        f1_smooth.apply(crop, geom, 0.5, _rng(), method="median")


def test_f3_magnitude_is_capped(crop, geom):
    from app.beautify import f3_reshape

    _, t = f3_reshape.apply(crop, geom, 1.0, _rng())
    limit = max(f3_reshape.LIMITS.jaw, f3_reshape.LIMITS.eyes, f3_reshape.LIMITS.nose)
    assert 0.5 < t.max_shift <= limit * crop.shape[0]
    # Strength above 1 is clamped.
    _, over = f3_reshape.apply(crop, geom, 5.0, _rng())
    assert over.max_shift == pytest.approx(t.max_shift)


def test_f3_transform_warps_back(crop, geom):
    """backward(forward(x)) returns close to x, and backward undoes the image warp."""
    from app.beautify import f3_reshape

    out, t = f3_reshape.apply(crop, geom, 1.0, _rng())
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    cx, cy = geom.points("nose").mean(0)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy - 40) ** 2) / (2 * 25.0**2))
    round_trip = t.backward(t.forward(blob))
    assert float(np.abs(round_trip - blob).max()) < 0.05
    restored = t.backward(out.astype(np.float64) / 255.0) * 255.0
    assert _mad(restored, crop) < 0.6 * _mad(out, crop)


def test_make_variants(crop, geom):
    from app import beautify

    variants = beautify.make_variants(crop, geom, 4, _rng(), (0.3, 1.0))
    assert [v.name for v in variants] == ["F1", "F2", "F3", "mix"]
    for v in variants:
        assert v.image.shape == crop.shape
        assert all(0.3 <= s <= 1.0 and f in beautify.TRAIN_FAMILIES for f, s in v.filters)
        assert (v.transform is not None) == any(f == "F3" for f, _ in v.filters)
    assert len(variants[3].filters) == 2
    assert beautify.make_variants(crop, geom, 0, _rng(), (0.3, 1.0)) == []
    again = beautify.make_variants(crop, geom, 4, _rng(), (0.3, 1.0))
    assert all(np.array_equal(a.image, b.image) for a, b in zip(variants, again, strict=True))
    six = beautify.make_variants(crop, geom, 6, _rng(), (0.3, 1.0))
    assert [v.name for v in six][4:] == ["F1_2", "F2_2"]


def test_f4_is_never_an_inference_copy_and_skips_when_missing(crop, tmp_path):
    from app import beautify
    from app.beautify import f4_restore

    assert "F4" not in beautify.TRAIN_FAMILIES
    assert not f4_restore.available(tmp_path)
    assert np.array_equal(f4_restore.apply(crop, 0.0, tmp_path), crop)
    with pytest.raises(f4_restore.F4Unavailable):
        f4_restore.apply(crop, 0.5, tmp_path)


def test_beautified_likelihood_rises_after_smoothing(crop, geom):
    from app.beautify import f1_smooth
    from app.pipeline import beautified_likelihood

    before = beautified_likelihood(crop, geom)
    after = beautified_likelihood(f1_smooth.apply(crop, geom, 1.0, _rng(), "gaussian"), geom)
    assert before is not None and after is not None
    assert 0.0 <= before < after <= 1.0
