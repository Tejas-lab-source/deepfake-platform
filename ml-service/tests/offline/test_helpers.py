import json
import os

import numpy as np
import pytest
import torch

from offline import common, heads, metrics, schemes, splits
from offline.celebdf import identity_components, parse_test_list
from offline.fixture import fixture_list_text

# ---- common ---------------------------------------------------------------------------


def test_atomic_writes_leave_no_temporary_files(tmp_path):
    common.atomic_write_json(tmp_path / "a.json", {"b": 1})
    common.atomic_save_npz(tmp_path / "x.npz", v=np.arange(3))
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    common.atomic_write_png(tmp_path / "i.png", img)
    assert json.loads((tmp_path / "a.json").read_text()) == {"b": 1}
    assert np.array_equal(np.load(tmp_path / "x.npz")["v"], np.arange(3))
    assert np.array_equal(common.read_png(tmp_path / "i.png"), img)  # lossless, RGB order
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json", "i.png", "x.npz"]


def test_failed_atomic_write_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "a.json"
    common.atomic_write_json(target, {"v": 1})

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        common.atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text()) == {"v": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_seeds_are_deterministic_and_distinct():
    assert common.seed_for("face", 1) == common.seed_for("face", 1)
    assert common.seed_for("face", 1) != common.seed_for("face", 2)
    assert common.rng_for("a").random() == common.rng_for("a").random()


def test_split_guard_keeps_test_away_from_training_and_calibration():
    common.check_split_access("train_head", "train")
    common.check_split_access("train_head", "val")
    common.check_split_access("calibrate", "val")
    common.check_split_access("eval_lofo", "test")
    for stage, split in [("train_head", "test"), ("calibrate", "test"), ("calibrate", "train")]:
        with pytest.raises(common.SplitAccessError):
            common.check_split_access(stage, split)
    with pytest.raises(common.SplitAccessError):
        common.check_split_access("unknown", "val")


def test_run_limiter():
    limiter = common.RunLimiter(limit=2)
    assert not limiter.should_stop()
    limiter.tick()
    limiter.tick()
    assert limiter.should_stop()
    assert common.RunLimiter(max_minutes=0).should_stop()
    assert "left" in limiter.progress(3)


# ---- splits ---------------------------------------------------------------------------


def _fixture_components():
    return identity_components(parse_test_list(fixture_list_text()))


def test_component_assignment_fills_every_subset_deterministically():
    comps = _fixture_components()
    a = splits.assign_components(comps, splits.DEFAULT_SUBSET_RATIOS, seed=0)
    assert a == splits.assign_components(comps, splits.DEFAULT_SUBSET_RATIOS, seed=0)
    assert set(a.values()) == set(common.SUBSETS)
    assert set(a) == {c.group_id for c in comps}
    with pytest.raises(ValueError):
        splits.assign_components(comps[:3], splits.DEFAULT_SUBSET_RATIOS, seed=0)
    with pytest.raises(ValueError):
        splits.assign_components(comps, {"train": 1.0}, seed=0)


def test_leakage_checks_refuse_crossing_identities_and_videos():
    ok = [
        {"video_id": "v1", "identities": ["id0"], "subset": "train"},
        {"video_id": "v2", "identities": ["id0", "id1"], "subset": "train"},
        {"video_id": "v3", "identities": ["id2"], "subset": "val-A"},
        {"video_id": "v4", "identities": ["id2"], "subset": "val-B"},  # same split: val
    ]
    splits.check_no_leakage(ok)
    leaky = ok + [{"video_id": "v5", "identities": ["id1"], "subset": "test"}]
    with pytest.raises(splits.LeakageError, match="cross splits"):
        splits.check_no_leakage(leaky)
    dup = ok + [{"video_id": "v1", "identities": [], "subset": "test"}]
    with pytest.raises(splits.LeakageError, match="video v1"):
        splits.check_no_leakage(dup)
    with pytest.raises(splits.LeakageError, match="unknown subset"):
        splits.check_no_leakage([{"video_id": "v", "identities": [], "subset": "dev"}])
    faces = [{"face_id": "f1", "video_id": "v1", "subset": "train"}]
    splits.check_face_leakage(faces, {"v1": "train"})
    with pytest.raises(splits.LeakageError, match="duplicate"):
        splits.check_face_leakage(faces * 2, {"v1": "train"})
    with pytest.raises(splits.LeakageError, match="differs"):
        splits.check_face_leakage(faces, {"v1": "test"})


# ---- schemes --------------------------------------------------------------------------


def test_scheme_table_matches_the_spec():
    s = schemes.SCHEMES
    assert list(s) == ["H-a", "H-b", "H-c", "H-d", "H-e", "H-f"]
    assert [x.invariance for x in s.values()] == [False, False, False, True, True, True]
    assert not s["H-a"].trains and all(x.trains for n, x in s.items() if n != "H-a")
    assert s["H-f"].selection == "hardest" and s["H-e"].selection == "random"


def test_candidates_per_fold():
    s = schemes.SCHEMES
    assert schemes.candidate_copies(s["H-c"], "F2") == ("F1_lo", "F1_hi", "F3_lo", "F3_hi")
    assert schemes.candidate_copies(s["H-f"], None) == (
        "F1_lo",
        "F1_hi",
        "F2_lo",
        "F2_hi",
        "F3_lo",
        "F3_hi",
    )
    assert schemes.candidate_copies(s["H-b"], "F2") == ("F1_bilat",)
    assert schemes.candidate_copies(s["H-d"], "F1") == ("generic",)
    assert schemes.candidate_copies(s["H-a"], "F1") == ()
    with pytest.raises(ValueError):
        schemes.training_families("F4")


def test_hb_f1_fold_is_not_held_out():
    s = schemes.SCHEMES
    assert not schemes.is_held_out(s["H-b"], "F1")
    assert schemes.is_held_out(s["H-b"], "F2") and schemes.is_held_out(s["H-f"], "F1")
    assert schemes.lambda_for(s["H-c"], 1.0) == 0.0 and schemes.lambda_for(s["H-f"], 1.0) == 1.0


def test_copy_selection():
    clean = np.array([0.0, 0.0, 0.0])
    cand = np.array([[1.0, -3.0], [0.5, 0.2], [9.0, 9.0]])
    avail = np.array([[True, True], [True, True], [False, False]])
    hard = schemes.select_copies("hardest", clean, cand, avail, np.random.default_rng(0))
    assert hard.tolist() == [1, 0, -1]
    avail2 = np.array([[False, True], [True, False], [True, True]])
    rnd = schemes.select_copies("random", clean, cand, avail2, np.random.default_rng(0))
    assert rnd[0] == 1 and rnd[1] == 0 and rnd[2] in (0, 1)
    with pytest.raises(ValueError):
        schemes.select_copies("worst", clean, cand, avail, np.random.default_rng(0))


# ---- heads ----------------------------------------------------------------------------


def test_invariance_loss_matches_hand_computation():
    clean, copy = torch.tensor([2.0, -1.0]), torch.tensor([1.0, 0.5])
    y = torch.tensor([1.0, 0.0])

    def bce(z, t):
        return -(t * np.log(1 / (1 + np.exp(-z))) + (1 - t) * np.log(1 - 1 / (1 + np.exp(-z))))

    expected = np.mean(
        [bce(2.0, 1) + bce(1.0, 1) + 0.5 * 1.0**2, bce(-1.0, 0) + bce(0.5, 0) + 0.5 * 1.5**2]
    )
    got = heads.invariance_loss(clean, copy, y, lambda_p=0.5).item()
    assert got == pytest.approx(expected, rel=1e-6)
    no_inv = heads.invariance_loss(clean, copy, y, lambda_p=0.0).item()
    assert no_inv < got
    w = torch.tensor([1.0, 3.0])
    weighted = heads.invariance_loss(clean, copy, y, 0.5, sample_weight=w).item()
    parts = [bce(2.0, 1) + bce(1.0, 1) + 0.5, bce(-1.0, 0) + bce(0.5, 0) + 0.5 * 2.25]
    assert weighted == pytest.approx((parts[0] + 3 * parts[1]) / 4, rel=1e-6)


def test_class_weights_and_normalize():
    w = heads.class_weights(np.array([1, 1, 1, 0]))
    assert w[:3].sum() == pytest.approx(w[3])
    x = heads.normalize(np.array([[3.0, 4.0]]))
    assert np.allclose(x, [[0.6, 0.8]])


def test_save_head_round_trip_is_weights_only_safe(tmp_path):
    w = np.linspace(-1, 1, heads.FEATURE_DIM, dtype=np.float32)
    sha = heads.save_head(tmp_path / "head.pt", w, 0.25, {"scheme": "H-f", "seed": 0})
    data = torch.load(tmp_path / "head.pt", weights_only=True)
    assert data["bias"] == 0.25 and data["l2_normalize"] is True
    assert np.allclose(data["weight"].numpy(), w) and data["meta"]["scheme"] == "H-f"
    assert sha == common.sha256_file(tmp_path / "head.pt")
    with pytest.raises(ValueError):
        heads.save_head(tmp_path / "bad.pt", w[:10], 0.0, {})
    with pytest.raises(TypeError):
        heads.save_head(tmp_path / "bad.pt", w, 0.0, {"x": object()})


# ---- metrics --------------------------------------------------------------------------


def test_auc_matches_sklearn_including_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)  # rounding creates ties
    assert metrics.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(metrics.auc(np.ones(3, int), np.zeros(3)))


def test_error_and_flip_rates():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.2, 0.7, 0.9, 0.4])
    assert metrics.false_positive_rate(y, p, 0.5) == 0.5
    assert metrics.false_negative_rate(y, p, 0.5) == 0.5
    assert metrics.flip_rate(np.array([0.2, 0.8]), np.array([0.6, 0.9]), 0.5) == 0.5
    assert np.isnan(metrics.false_positive_rate(np.ones(2, int), np.zeros(2), 0.5))


def test_cluster_bootstrap_is_reproducible_and_resamples_whole_videos():
    clusters = np.array(["a", "a", "b", "b", "c"])
    values = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
    lengths = []

    def stat(idx):
        lengths.append(len(idx))
        return float(values[idx].mean())

    ci = metrics.cluster_bootstrap(clusters, stat, n_resamples=200, seed=1)
    assert ci == metrics.cluster_bootstrap(clusters, stat, n_resamples=200, seed=1)
    assert 0.0 <= ci[0] <= ci[1] <= 1.0
    assert all(n in (3, 4, 5, 6) for n in lengths)  # sums of whole-video sizes
    nan_ci = metrics.cluster_bootstrap(clusters, lambda _i: float("nan"), n_resamples=10)
    assert np.isnan(nan_ci[0])


def test_mean_std():
    assert metrics.mean_std([1.0, 3.0]) == (2.0, pytest.approx(np.sqrt(2)))
    assert metrics.mean_std([2.0, float("nan")]) == (2.0, 0.0)
    assert np.isnan(metrics.mean_std([])[0])
