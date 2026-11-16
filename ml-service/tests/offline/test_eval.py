import json
import shutil

import numpy as np
import pytest

from offline import eval_lofo
from offline.schemes import SCHEMES


def _agg(**values):
    return {s: {m: {"mean": v} for m, v in values.get(s, {}).items()} for s in SCHEMES}


def test_success_criteria_logic():
    met = eval_lofo.criteria(
        _agg(
            **{
                "H-f": {"flip_rate": 0.10, "fpr_beautified_real": 0.05, "clean_auc": 0.905},
                "H-c": {"flip_rate": 0.20, "fpr_beautified_real": 0.10},
                "H-a": {"fpr_beautified_real": 0.12, "clean_auc": 0.90},
            }
        )
    )
    assert [c["result"] for c in met] == ["MET", "MET", "MET"]
    missed = eval_lofo.criteria(
        _agg(
            **{
                "H-f": {"flip_rate": 0.15, "fpr_beautified_real": 0.11, "clean_auc": 0.88},
                "H-c": {"flip_rate": 0.20, "fpr_beautified_real": 0.10},
                "H-a": {"fpr_beautified_real": 0.12, "clean_auc": 0.90},
            }
        )
    )
    # 0.15 is only 25% below 0.20; FPR 0.11 is not below H-c's 0.10; AUC is 2 points lower.
    assert [c["result"] for c in missed] == ["NOT MET", "NOT MET", "NOT MET"]
    assert {c["result"] for c in eval_lofo.criteria(_agg())} == {"NOT EVALUABLE"}


def test_face_level_metrics():
    y = np.array([0, 0, 1, 1], np.int8)
    fl = {
        "logit_clean": np.array([-2.0, -1.0, 1.0, 2.0]),
        "p_clean": np.array([0.1, 0.3, 0.7, 0.9]),
        "p_held": np.array([[0.6, 0.2], [0.4, 0.4], [0.8, 0.3], [0.9, 0.9]]),
        "held_present": np.ones((4, 2), bool),
        "p_f4": None,
        "f4_present": None,
        "uncertain": np.array([True, False, False, False]),
    }
    m = eval_lofo.metrics_for(fl, y, 0.5)
    assert m["clean_auc"] == 1.0 and m["clean_error_forced"] == 0.0
    assert m["fpr_beautified_real"] == 0.25  # 1 of 4 real copies reads FAKE
    assert m["fnr_beautified_fake"] == 0.25  # 1 of 4 fake copies reads REAL
    assert m["flip_rate"] == 0.25 and m["uncertain_rate"] == 0.25
    assert np.isnan(m["f4_fpr_real"])


def test_lofo_calibration_ignores_the_held_out_family(cache_dir):
    """T and threshold must not change when the held-out family's features change."""
    from offline.cache_features import load_split

    val = eval_lofo.Split(load_split(cache_dir, "val", "eval_lofo"))
    with np.load(cache_dir / "gend_head.npz") as g:
        w, b = g["weight"].astype(np.float32), float(g["bias"])
    _, before = eval_lofo.head_calibration(val, w, b, "F2")
    for name in ("F2_lo", "F2_hi"):
        val.z[:, val.col(name)] = np.random.default_rng(0).normal(size=val.z[:, 0].shape)
    _, after = eval_lofo.head_calibration(val, w, b, "F2")
    assert before["temperature"] == after["temperature"]
    assert before["threshold"] == after["threshold"]


def test_fixture_report(eval_report):
    r = eval_report
    assert r["fixture"] is True and "PLUMBING FIXTURE" in r["notice"]
    assert "NOT comparable" in r["protocol"]
    assert len(r["runs"]) == 6 * 3 * 3
    assert r["per_fold"]["H-b__F1"]["held_out"] is False
    assert r["aggregate_over_held_out_folds"]["H-b"]["folds_included"] == ["F2", "F3"]
    assert r["aggregate_over_held_out_folds"]["H-f"]["folds_included"] == ["F1", "F2", "F3"]
    assert len(r["heatmap_iou_seed0"]) == 18
    assert all(v["faces"] == 1 for v in r["heatmap_iou_seed0"].values())
    assert len(r["ablation_shipped_head_clean_test_inputs"]) == 5
    assert all("fixture_only" in c for c in r["success_criteria"])
    assert r["inputs"]["test_faces"] == 4
    for run in r["runs"]:
        assert set(eval_lofo.METRICS) <= set(run["metrics"])
    # H-a has no training: its three seeds are identical.
    ha = [
        x["metrics"]["flip_rate"] for x in r["runs"] if x["scheme"] == "H-a" and x["fold"] == "F1"
    ]
    assert len(set(ha)) == 1


def test_tampered_head_or_mismatched_card_is_refused(
    card_path, heads_dir, cache_dir, faces_dir, tmp_path
):
    from tests.conftest import WEIGHTS

    heads = tmp_path / "heads"
    shutil.copytree(heads_dir, heads)
    base = [
        "--cache",
        str(cache_dir),
        "--heads-dir",
        str(heads),
        "--faces",
        str(faces_dir),
        "--weights-dir",
        str(WEIGHTS),
        "--out",
        str(tmp_path / "o.json"),
        "--iou-faces",
        "0",
        "--bootstrap",
        "0",
    ]
    card = tmp_path / "card.json"
    data = json.loads(card_path.read_text())
    data["head_sha256"] = "0" * 64
    card.write_text(json.dumps(data))
    args = base + ["--final-head", str(heads / "weights" / "head.pt"), "--model-card", str(card)]
    assert eval_lofo.main(args) == 2
    import torch

    index = json.loads((heads / "index.json").read_text())
    f = heads / index["H-f__F1__seed0"]["file"]
    d = torch.load(f, weights_only=True)
    d["bias"] = d["bias"] + 1.0
    torch.save(d, f)
    args = base + [
        "--final-head",
        str(heads / "weights" / "head.pt"),
        "--model-card",
        str(card_path),
    ]
    assert eval_lofo.main(args) == 2
    assert not (tmp_path / "o.json").exists()


@pytest.mark.parametrize("metric", eval_lofo.METRICS)
def test_every_metric_is_documented(metric):
    name = metric.replace("f4_", "f4_*") if metric.startswith("f4_") else metric
    assert name.split("*")[0] in eval_lofo.__doc__
