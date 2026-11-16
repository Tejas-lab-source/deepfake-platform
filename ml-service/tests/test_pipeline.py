from app.head import Calibration, FusionWeights, fuse, load_calibration, probability, sigmoid
from app.pipeline import aggregate, decide, top_k_mean

CAL = Calibration()


def test_defaults_are_identity_fusion(tmp_path):
    cal = load_calibration(tmp_path / "missing.json")
    assert cal.temperature == 1.0 and cal.threshold == 0.5
    assert fuse(cal, 1.3, -2.0, 0.9, 0.4) == 1.3


def test_model_card_overrides(tmp_path):
    card = tmp_path / "model_card.json"
    card.write_text(
        '{"temperature": 2.0, "threshold": 0.4, "fusion": {"sbi": 0.5}, "version": "v1"}'
    )
    cal = load_calibration(card)
    assert cal.temperature == 2.0 and cal.threshold == 0.4 and cal.version == "v1"
    assert cal.fusion == FusionWeights(sbi=0.5)
    assert probability(cal, 2.0) == sigmoid(1.0)


def test_aggregate_mean_and_spread():
    mean, spread = aggregate({"clean": 1.0, "F1": 2.0, "F2": 0.0})
    assert mean == 1.0 and spread == 2.0


def test_confident_fake_and_real():
    fake = decide(CAL, {"clean": 3.0, "F1": 3.1}, 2.5, 0.2)
    assert (fake.verdict, fake.confident, fake.uncertain_reason) == ("FAKE", True, None)
    real = decide(CAL, {"clean": -3.0, "F1": -2.9}, -2.5, 0.2)
    assert (real.verdict, real.confident) == ("REAL", True)


def test_not_confident_between_margins():
    # p = sigmoid(0.7) ~ 0.668: outside the 0.1 uncertain margin, inside the 0.25 confident one.
    d = decide(CAL, {"clean": 0.7}, 0.7, 0.0)
    assert (d.verdict, d.confident) == ("FAKE", False)


def test_uncertain_near_threshold():
    d = decide(CAL, {"clean": 0.1}, 0.1, 0.0)
    assert (d.verdict, d.uncertain_reason) == ("UNCERTAIN", "near_threshold")


def test_uncertain_when_spread_too_large():
    d = decide(CAL, {"clean": 4.0, "F1": 2.0}, 3.0, 0.0)
    assert (d.verdict, d.uncertain_reason) == ("UNCERTAIN", "unstable_under_beautification")


def test_uncertain_when_variants_disagree():
    d = decide(CAL, {"clean": 1.4, "F1": 1.4, "F2": 1.4, "F3": -0.1}, 1.0, 0.0)
    assert (d.verdict, d.uncertain_reason) == ("UNCERTAIN", "variants_disagree")


def test_uncertain_when_detectors_disagree():
    d = decide(CAL, {"clean": 3.0}, -3.0, 0.0)
    assert (d.verdict, d.uncertain_reason) == ("UNCERTAIN", "detectors_disagree")


def test_top_k_mean():
    assert top_k_mean([0.0, 1.0, 2.0, 3.0], 2) == 2.5
    assert top_k_mean([1.0], 5) == 1.0


def test_copy_votes_use_the_same_scale_as_the_verdict():
    """Regression (Phase 3): with non-identity fusion, each copy must vote on the fused,
    temperature-scaled probability, not on its raw logit."""
    from app.head import Calibration, FusionWeights

    cal = Calibration(fusion=FusionWeights(bias=3.0), spread_limit=10.0)
    # Raw logits straddle 0, but after the +3 fusion bias both copies are clearly FAKE.
    d = decide(cal, {"clean": 0.5, "F1": -0.5}, None, 0.0)
    assert (d.verdict, d.uncertain_reason) == ("FAKE", None)
    # Under identity fusion the copies really do disagree, as before.
    plain = decide(Calibration(spread_limit=10.0), {"clean": 1.5, "F1": -0.2}, None, 0.0)
    assert plain.uncertain_reason == "variants_disagree"
