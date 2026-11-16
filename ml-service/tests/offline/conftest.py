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
