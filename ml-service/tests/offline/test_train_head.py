import json

import numpy as np
import pytest
import torch

from offline import train_head
from offline.heads import normalize
from offline.schemes import SCHEMES

NAMES = ["clean", "F1_lo", "F1_hi", "F2_lo", "F2_hi", "F3_lo", "F3_hi", "F1_bilat", "generic"]
CFG = train_head.TrainConfig(epochs=150, batch_size=32, lr=1e-2, patience=1000)


def _synthetic(n=200, nuisance=1.5, seed=0):
    """Labels depend on direction u; every beautified copy adds a nuisance direction v."""
    rng = np.random.default_rng(seed)
    d = 1024
    u, v = np.zeros(d, np.float32), np.zeros(d, np.float32)
    u[0], v[1] = 1.0, 1.0
    y = rng.integers(0, 2, n).astype(np.int8)
    base = rng.normal(0, 0.3, (n, d)).astype(np.float32)
    base[:, 0] += np.where(y == 1, 2.0, -2.0)
    z = np.repeat(base[:, None], len(NAMES), axis=1)
    z[:, 1:] += nuisance * v
    data = train_head.Data(NAMES, normalize(z), np.ones((n, len(NAMES)), bool), y)
    return data, u, v


def test_h_a_is_gend_untouched():
    data, _, _ = _synthetic(20)
    w0 = np.ones(1024, np.float32)
    w, b, info = train_head.train_one(SCHEMES["H-a"], "F1", 0, 1.0, data, data, w0, 0.5, CFG)
    assert np.array_equal(w, w0) and b == 0.5 and info["epochs"] == 0


def test_training_learns_the_label_direction():
    data, _, _ = _synthetic()
    w0 = np.zeros(1024, np.float32)
    w, b, info = train_head.train_one(SCHEMES["H-c"], "F1", 0, 0.0, data, data, w0, 0.0, CFG)
    auc, _ = train_head.val_a_rule_stats(data, w, b)
    assert auc > 0.95 and info["candidates"] == ["F2_lo", "F2_hi", "F3_lo", "F3_hi"]


def test_invariance_loss_shrinks_the_clean_to_copy_gap():
    data, _, v = _synthetic()
    w0 = np.zeros(1024, np.float32)
    w0[1] = 3.0  # an initial head that reacts strongly to the nuisance direction

    def gap(w, b):
        logits = data.z @ w + b
        return float(np.abs(logits[:, 0] - logits[:, 3]).mean())

    w_c, b_c, _ = train_head.train_one(SCHEMES["H-c"], "F1", 0, 0.0, data, data, w0, 0.0, CFG)
    w_e, b_e, _ = train_head.train_one(SCHEMES["H-e"], "F1", 0, 10.0, data, data, w0, 0.0, CFG)
    assert gap(w_e, b_e) < 0.5 * gap(w_c, b_c)


def test_scheme_candidates_are_respected():
    data, _, _ = _synthetic(40)
    w0 = np.zeros(1024, np.float32)
    for name, expected in [("H-b", ["F1_bilat"]), ("H-d", ["generic"])]:
        _, _, info = train_head.train_one(SCHEMES[name], "F2", 0, 1.0, data, data, w0, 0.0, CFG)
        assert info["candidates"] == expected
    _, _, info_b = train_head.train_one(SCHEMES["H-b"], "F2", 0, 1.0, data, data, w0, 0.0, CFG)
    assert info_b["lambda_p"] == 0.0  # H-b has no invariance loss


def test_single_class_validation_disables_early_stopping_and_says_so():
    data, _, _ = _synthetic(40)
    one_class = train_head.Data(data.names, data.z[:5], data.present[:5], np.ones(5, np.int8))
    cfg = train_head.TrainConfig(epochs=3, batch_size=16, lr=1e-3, patience=1)
    _, _, info = train_head.train_one(
        SCHEMES["H-f"], "F1", 0, 1.0, data, one_class, np.zeros(1024, np.float32), 0.0, cfg
    )
    assert "warning" in info and info["epochs_run"] == 3


def test_fixture_training_outputs(heads_dir, cache_dir, monkeypatch):
    index = json.loads((heads_dir / "index.json").read_text())
    assert len(index) == 6 * 3 * 3
    trained = [e for e in index.values() if e["scheme"] != "H-a"]
    assert len(trained) == 45 and all((heads_dir / e["file"]).is_file() for e in trained)
    assert index["H-b__F1__seed0"]["held_out"] is False
    assert all(e["held_out"] for k, e in index.items() if not k.startswith("H-b__F1"))
    final = json.loads((heads_dir / "final.json").read_text())
    assert final["scheme"] == "H-f" and final["lambda_p"] in train_head.LAMBDA_GRID
    assert final["fixture"] is True and "lambda_rule" in final
    data = torch.load(heads_dir / "weights" / "head.pt", weights_only=True)
    assert data["weight"].shape == (1024,) and data["meta"]["fixture"] is True


def test_training_reads_no_test_and_no_val_b(cache_dir, monkeypatch):
    from offline import cache_features

    requested = []
    real = cache_features.load_split

    def spy(cache, split, stage):
        requested.append((split, stage))
        return real(cache, split, stage)

    monkeypatch.setattr(train_head, "load_split", spy)
    train, val_a, _, _, _ = train_head.load_training_data(cache_dir)
    assert requested == [("train", "train_head"), ("val", "train_head")]
    assert len(val_a.y) == 4  # val-A only: the 4 val-B faces are not passed on


def test_rerun_skips_finished_heads(heads_dir, cache_dir):
    before = {p.name: p.stat().st_mtime_ns for p in heads_dir.glob("*.pt")}
    assert train_head.main(["--cache", str(cache_dir), "--out-dir", str(heads_dir)]) == 0
    after = {p.name: p.stat().st_mtime_ns for p in heads_dir.glob("*.pt")}
    assert before == after


@pytest.mark.parametrize("lam", [0.0, 10.0])
def test_lambda_grid_covers_the_spec_default(lam):
    assert lam in train_head.LAMBDA_GRID and 1.0 in train_head.LAMBDA_GRID
