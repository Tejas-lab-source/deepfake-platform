import json

import numpy as np
import pytest

from offline import calibrate as cb


def _faces(n_videos_per_class=30, frames=4, seed=0, subsets=("val-A", "val-B")):
    """Synthetic FaceInputs: logits shifted by class, small copy noise, both halves."""
    rng = np.random.default_rng(seed)
    vids, subs, ys, variants, sbi, bl = [], [], [], [], [], []
    for half in subsets:
        for cls in (0, 1):
            for v in range(n_videos_per_class):
                for _ in range(frames):
                    base = rng.normal(2.0 if cls else -2.0, 1.0)
                    var = {"clean": base}
                    for slot in ("F1", "F2", "F3", "mix"):
                        var[slot] = base + rng.normal(0, 0.3)
                    vids.append(f"{half}-{cls}-{v}")
                    subs.append(half)
                    ys.append(cls)
                    variants.append(var)
                    sbi.append(float(rng.normal(1.0 if cls else -1.0, 1.0)))
                    bl.append(float(rng.uniform()))
    m1 = np.array([np.mean(list(v.values())) for v in variants])
    spread = np.array([max(v.values()) - min(v.values()) for v in variants])
    return cb.FaceInputs(
        np.array(vids),
        np.array(subs),
        np.array(ys, np.int8),
        variants,
        m1,
        spread,
        sbi,
        np.array(bl),
    )


def test_temperature_recovers_a_known_value():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 4, 20000)
    y = (rng.random(20000) < cb.sigmoid(z / 2.0)).astype(int)
    assert cb.fit_temperature(z, y) == pytest.approx(2.0, rel=0.08)


def test_threshold_rule():
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1])
    t = cb.fit_threshold(p, y)
    assert 0.3 < t <= 0.8 and cb.balanced_accuracy(p, y, t) == 1.0
    # All perfect thresholds lie in (0.3, 0.8]; the one closest to 0.5 is chosen.
    assert t == pytest.approx(0.5)
    assert cb.THRESHOLD_RANGE[0] <= cb.fit_threshold(np.array([0.0, 1.0]), np.array([0, 1]))


def test_spread_limit_is_the_95th_percentile():
    s = np.arange(101, dtype=float)
    assert cb.fit_spread_limit(s) == pytest.approx(95.0)


def test_fusion_learns_the_informative_input():
    faces = _faces()
    fusion = cb.fit_fusion(faces.features(), faces.y)
    assert fusion["head"] > 0.5 and abs(fusion["beautified_likelihood"]) < fusion["head"]


def test_vectorised_fusion_matches_the_service():
    from app.head import Calibration, FusionWeights, fuse

    faces = _faces(3, 2)
    fusion = {"head": 0.7, "sbi": 0.2, "spread": -0.3, "beautified_likelihood": 0.1, "bias": 0.4}
    cal = Calibration(fusion=FusionWeights(**fusion))
    expected = [
        fuse(cal, faces.m1[i], faces.sbi[i], faces.spread[i], faces.bl[i])
        for i in range(len(faces.y))
    ]
    assert np.allclose(cb.fused_logits(fusion, faces), expected)


def test_margin_respects_the_uncertain_cap():
    from app.head import Calibration

    faces = _faces(10, 2).take(np.ones(80, bool))
    margin, rate, cap_met = cb.fit_margin(Calibration(spread_limit=100.0), faces)
    assert cap_met and rate <= cb.MAX_UNCERTAIN_RATE
    if margin < cb.MARGIN_GRID[-1]:
        over = cb.uncertain_rate(
            Calibration(spread_limit=100.0, uncertain_margin=margin + 0.01), faces
        )
        assert over > cb.MAX_UNCERTAIN_RATE
    # A spread limit of 0 makes every face UNCERTAIN: the cap cannot be met, and it says so.
    margin0, rate0, met0 = cb.fit_margin(Calibration(spread_limit=0.0), faces)
    assert (margin0, met0) == (0.0, False) and rate0 == 1.0


def test_sufficient_data_gives_ok_with_no_warnings():
    values, sources, warnings = cb.calibrate(_faces(), n_boot=20)
    assert warnings == []
    assert values["temperature"] > 0 and 0.05 <= values["threshold"] <= 0.95
    for key in ("fusion", "temperature", "threshold", "spread_limit", "uncertain_margin"):
        assert sources[key]["data"] in ("val-A", "val-B")
    assert sources["temperature"]["ci95"] is not None
    assert sources["uncertain_margin"]["uncertain_rate"] <= cb.MAX_UNCERTAIN_RATE


def test_small_or_one_class_validation_is_reported_not_hidden():
    _, _, warnings = cb.calibrate(_faces(n_videos_per_class=3, frames=2), n_boot=0)
    assert any("real videos (< 20)" in w for w in warnings)
    faces = _faces(5, 2)
    one_class_b = faces.take(~((faces.subset == "val-B") & (faces.y == 1)))
    values, sources, warnings = cb.calibrate(one_class_b, n_boot=0)
    assert "NOT fitted" in sources["temperature"]["method"]
    assert values["temperature"] == 1.0 and values["threshold"] == 0.5
    assert any("lacks both classes" in w for w in warnings)


def test_fixture_model_card(card_path, heads_dir):
    from offline.common import sha256_file

    card = json.loads(card_path.read_text())
    assert card["calibration_status"] == "insufficient_validation_data"
    assert card["fixture"] is True and "PLUMBING FIXTURE" in card["warnings"][0]
    assert card["head_sha256"] == sha256_file(heads_dir / "weights" / "head.pt")
    assert card["data"]["test_split_used"] is False and "NOT comparable" in card["protocol"]
    for key in ("fusion", "temperature", "threshold", "spread_limit", "uncertain_margin"):
        assert key in card["sources"]
    # The existing service loader accepts the card.
    from app.head import load_calibration

    cal = load_calibration(card_path)
    assert cal.threshold == pytest.approx(card["threshold"]) and cal.version == card["version"]


def test_calibration_never_reads_test(cache_dir, heads_dir, tmp_path, monkeypatch):
    from offline import cache_features

    requested = []
    real = cache_features.load_split
    monkeypatch.setattr(
        cb, "load_split", lambda c, s, st: requested.append((s, st)) or real(c, s, st)
    )
    args = [
        "--cache",
        str(cache_dir),
        "--head",
        str(heads_dir / "weights" / "head.pt"),
        "--head-report",
        str(heads_dir / "final.json"),
        "--out",
        str(tmp_path / "c.json"),
        "--bootstrap",
        "0",
    ]
    assert cb.main(args) == 0
    assert requested == [("val", "calibrate")]


def test_mismatched_head_report_is_refused(heads_dir, cache_dir, tmp_path):
    report = json.loads((heads_dir / "final.json").read_text())
    report["sha256"] = "0" * 64
    bad = tmp_path / "final.json"
    bad.write_text(json.dumps(report))
    args = [
        "--cache",
        str(cache_dir),
        "--head",
        str(heads_dir / "weights" / "head.pt"),
        "--head-report",
        str(bad),
        "--out",
        str(tmp_path / "c.json"),
    ]
    assert cb.main(args) == 2 and not (tmp_path / "c.json").exists()
