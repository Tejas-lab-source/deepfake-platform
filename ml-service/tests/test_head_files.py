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
