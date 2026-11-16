"""Regression tests for the Phase 3 fixes to head loading and model-card validation."""

import json

import pytest

from app.head import (
    Calibration,
    LinearHead,
    ModelFilesError,
    check_model_files,
    load_calibration,
)

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


def _save(path, weight, bias=0.0, **extra):
    torch.save({"weight": weight, "bias": bias, "l2_normalize": True, **extra}, path)


def test_head_loads_safely_with_its_sha(tmp_path):
    path = tmp_path / "head.pt"
    _save(path, torch.ones(1024), 0.5, meta={"scheme": "H-f"})
    head = LinearHead.load(path)
    assert head is not None and head.weight.shape == (1024,) and head.bias == 0.5
    import hashlib

    assert head.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert LinearHead.load(tmp_path / "missing.pt") is None


def test_head_with_python_objects_is_refused(tmp_path):
    import pickle

    path = tmp_path / "head.pt"
    _save(path, torch.ones(1024), extra=print)  # a function cannot be loaded weights-only
    with pytest.raises(pickle.UnpicklingError):
        LinearHead.load(path)


@pytest.mark.parametrize(
    "weight, message",
    [
        (torch.ones(512), "512 values, expected 1024"),
        (torch.ones(2, 1024), "2048 values, expected 1024"),
        (torch.full((1024,), float("nan")), "non-finite"),
    ],
)
def test_head_shape_and_values_are_validated(tmp_path, weight, message):
    path = tmp_path / "head.pt"
    _save(path, weight)
    with pytest.raises(ModelFilesError, match=message):
        LinearHead.load(path)


def test_head_file_without_weight_is_refused(tmp_path):
    path = tmp_path / "head.pt"
    torch.save({"bias": 0.0}, path)
    with pytest.raises(ModelFilesError, match="not a head file"):
        LinearHead.load(path)


def _head(sha="a" * 64):
    return LinearHead(np.zeros(1024, np.float32), 0.0, True, sha)


def test_model_file_pairing_rules():
    check_model_files(Calibration(), None, False)  # no head, no card: Phase 2 behaviour
    check_model_files(Calibration(calibration_status="ok", head_sha256="a" * 64), _head(), False)
    with pytest.raises(ModelFilesError, match="does not match"):
        check_model_files(
            Calibration(calibration_status="ok", head_sha256="b" * 64), _head(), False
        )
    with pytest.raises(ModelFilesError, match="not fitted for a head"):
        check_model_files(Calibration(), _head(), False)
    with pytest.raises(ModelFilesError, match="head.pt is missing"):
        check_model_files(Calibration(calibration_status="ok", head_sha256="a" * 64), None, False)


def test_unvalidated_cards_need_explicit_permission():
    insufficient = Calibration(
        calibration_status="insufficient_validation_data", head_sha256="a" * 64
    )
    fixture = Calibration(calibration_status="ok", fixture=True, head_sha256="a" * 64)
    for cal in (insufficient, fixture):
        with pytest.raises(ModelFilesError, match="not approved"):
            check_model_files(cal, _head(), False)
        check_model_files(cal, _head(), True)


def test_model_card_fields_round_trip(tmp_path):
    assert load_calibration(tmp_path / "none.json").calibration_status == "defaults"
    card = tmp_path / "model_card.json"
    card.write_text(
        json.dumps({"calibration_status": "ok", "fixture": False, "head_sha256": "c" * 64})
    )
    cal = load_calibration(card)
    assert (cal.calibration_status, cal.fixture, cal.head_sha256) == ("ok", False, "c" * 64)
    # An old card without the new fields is "unknown", which the service refuses.
    card.write_text("{}")
    assert load_calibration(card).calibration_status == "unknown"
    with pytest.raises(ModelFilesError):
        check_model_files(load_calibration(card), None, False)


# ---- the original head mathematics, through a real checkpoint (audit item 3) -------------
# test_explain.py::test_linear_head_math used to save a 2-value head and load it. A 2-value
# head is now refused, so the same arithmetic is checked here through a valid 1024-value
# file: save -> load -> logits.


def _padded(*values):
    w = torch.zeros(1024)
    w[: len(values)] = torch.tensor(values)
    return w


def _inputs(*rows):
    x = np.zeros((len(rows), 1024), np.float32)
    for i, row in enumerate(rows):
        x[i, : len(row)] = row
    return x


def test_loaded_head_reproduces_the_original_math(tmp_path):
    path = tmp_path / "head.pt"
    _save(path, _padded(3.0, 4.0), 0.5)
    head = LinearHead.load(path)
    # (3, 4) normalised is (0.6, 0.8): 3*0.6 + 4*0.8 + 0.5 = 5.5, whatever the input's length.
    assert head.logits(_inputs([3.0, 4.0], [30.0, 40.0])) == pytest.approx([5.5, 5.5])


def test_loaded_head_without_normalisation(tmp_path):
    path = tmp_path / "head.pt"
    torch.save({"weight": _padded(3.0, 4.0), "bias": 0.5, "l2_normalize": False}, path)
    head = LinearHead.load(path)
    assert head.l2_normalize is False
    # Raw dot products: 3*3 + 4*4 + 0.5 = 25.5 and 3*30 + 4*40 + 0.5 = 250.5.
    assert head.logits(_inputs([3.0, 4.0], [30.0, 40.0])) == pytest.approx([25.5, 250.5])


def test_missing_normalisation_flag_defaults_to_l2(tmp_path):
    path = tmp_path / "head.pt"
    torch.save({"weight": _padded(3.0, 4.0), "bias": 0.5}, path)
    assert LinearHead.load(path).l2_normalize is True


def test_metadata_does_not_change_the_head(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    _save(a, _padded(1.0, -2.0), 0.25)
    _save(b, _padded(1.0, -2.0), 0.25, meta={"scheme": "H-f", "lambda_p": 0.3, "seed": 0})
    ha, hb = LinearHead.load(a), LinearHead.load(b)
    x = _inputs([1.0, 1.0], [-2.0, 5.0])
    assert np.array_equal(ha.logits(x), hb.logits(x))
    assert ha.sha256 != hb.sha256  # different files, so the model card must name the right one


def test_trained_head_files_load_in_the_service(tmp_path):
    """A head written by offline/heads.save_head loads in the service with the same values."""
    from offline.heads import save_head

    w = np.linspace(-1, 1, 1024, dtype=np.float32)
    sha = save_head(tmp_path / "head.pt", w, -0.75, {"scheme": "H-f"})
    head = LinearHead.load(tmp_path / "head.pt")
    assert np.array_equal(head.weight, w) and head.bias == -0.75 and head.sha256 == sha
