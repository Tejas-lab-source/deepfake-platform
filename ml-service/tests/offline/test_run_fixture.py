"""The Phase 3 done criterion: every stage runs end to end on the 20-face fixture and
produces head.pt and model_card.json, which the service then loads."""

import json

import pytest
import torch

from tests.conftest import GEND, WEIGHTS

pytestmark = pytest.mark.skipif(not GEND.is_file(), reason="needs downloaded weights")


def test_end_to_end_fixture_run(tmp_path):
    from offline import run_fixture

    work = tmp_path / "run"
    args = [
        "--workdir",
        str(work),
        "--weights-dir",
        str(WEIGHTS),
        "--iou-faces",
        "1",
        "--bootstrap",
        "20",
    ]
    assert run_fixture.main(args) == 0
    summary = json.loads((work / "summary.json").read_text())
    head = torch.load(work / "weights" / "head.pt", weights_only=True)
    card = json.loads((work / "weights" / "model_card.json").read_text())
    assert head["weight"].shape == (1024,)
    assert card["fixture"] is True and card["calibration_status"] == "insufficient_validation_data"
    s = summary["service_check"]
    assert s["refused_without_permission"] and s["head_loaded"] and s["head_sha256_matches_card"]
    assert s["copies_scored"] == ["F1", "F2", "F3", "clean", "mix"]
    assert "PLUMBING FIXTURE" in summary["notice"]
    report = json.loads((work / "eval" / "lofo.json").read_text())
    assert report["fixture"] is True and len(report["runs"]) == 54


def test_runner_refuses_the_real_weights_folder(tmp_path):
    from offline import run_fixture

    assert (
        run_fixture.main(["--workdir", str(tmp_path), "--weights-dir", str(tmp_path / "weights")])
        == 2
    )
