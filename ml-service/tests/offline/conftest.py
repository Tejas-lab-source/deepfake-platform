"""Session fixtures: the plumbing fixture dataset and its extracted faces, built once."""

from __future__ import annotations

import pytest

from tests.conftest import HAS_ML, LANDMARKER


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory):
    if not HAS_ML:
        pytest.skip("ML stack not installed")
    from offline.fixture import write_fixture

    root = tmp_path_factory.mktemp("fixture")
    write_fixture(root)
    return root


@pytest.fixture(scope="session")
def faces_dir(fixture_root, tmp_path_factory):
    if not LANDMARKER.is_file():
        pytest.skip("needs the MediaPipe model (scripts/download_weights.sh)")
    from offline import extract_faces

    out = tmp_path_factory.mktemp("faces")
    assert extract_faces.main(["--data-root", str(fixture_root), "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="session")
def cache_dir(faces_dir, tmp_path_factory):
    from tests.conftest import GEND

    if not GEND.is_file():
        pytest.skip("needs the GenD weights (scripts/download_weights.sh)")
    from offline import cache_features
    from tests.conftest import WEIGHTS

    out = tmp_path_factory.mktemp("cache")
    args = ["--faces", str(faces_dir), "--out", str(out), "--weights-dir", str(WEIGHTS)]
    assert cache_features.main(args) == 0
    return out


@pytest.fixture(scope="session")
def heads_dir(cache_dir, tmp_path_factory):
    from offline import train_head

    out = tmp_path_factory.mktemp("heads")
    args = [
        "--cache",
        str(cache_dir),
        "--out-dir",
        str(out),
        "--final",
        "--final-out",
        str(out / "weights" / "head.pt"),
    ]
    assert train_head.main(args) == 0
    return out


@pytest.fixture(scope="session")
def card_path(heads_dir, cache_dir):
    from offline import calibrate

    out = heads_dir / "weights" / "model_card.json"
    args = [
        "--cache",
        str(cache_dir),
        "--head",
        str(heads_dir / "weights" / "head.pt"),
        "--head-report",
        str(heads_dir / "final.json"),
        "--out",
        str(out),
        "--bootstrap",
        "50",
    ]
    assert calibrate.main(args) == 0
    return out
