import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.media import sniff

FIXTURE = Path(__file__).parent / "fixtures" / "sample.png"
HEADERS = {"X-Internal-Secret": main.settings.internal_secret}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "upload_dir", tmp_path)
    monkeypatch.setattr(main.settings, "mock_mode", True)
    shutil.copy(FIXTURE, tmp_path / "sample.png")
    (tmp_path / "notes.png").write_text("this is not an image")
    return TestClient(main.app)


def test_health_is_open_and_others_need_the_secret(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 401
    assert client.get("/ready", headers={"X-Internal-Secret": "wrong"}).status_code == 401
    assert client.get("/ready", headers=HEADERS).json()["ready"] is True


def test_analyze_image_is_deterministic_and_well_formed(client):
    body = {"path": "sample.png"}
    first = client.post("/analyze/image", json=body, headers=HEADERS).json()
    second = client.post("/analyze/image", json=body, headers=HEADERS).json()
    for r in (first, second):
        r.pop("timingsMs")
    assert first == second
    assert first["verdict"] in {"FAKE", "REAL", "UNCERTAIN"}
    assert first["mock"] is True
    face = first["faces"][0]
    assert set(face["variantLogits"]) == {"clean", "F1", "F2", "F3", "mix"}
    assert (first["verdict"] == "UNCERTAIN") == (first["uncertainReason"] is not None)


def test_k_zero_scores_only_the_clean_crop(client):
    r = client.post("/analyze/image", json={"path": "sample.png", "k": 0}, headers=HEADERS).json()
    assert set(r["faces"][0]["variantLogits"]) == {"clean"}
    assert r["faces"][0]["spread"] == 0


def test_rejects_wrong_magic_bytes_and_path_escape(client):
    bad = client.post("/analyze/image", json={"path": "notes.png"}, headers=HEADERS)
    assert bad.status_code == 422
    escape = client.post("/analyze/image", json={"path": "../../etc/passwd"}, headers=HEADERS)
    assert escape.status_code == 422


def test_real_mode_reports_not_ready_until_models_load(client, monkeypatch):
    monkeypatch.setattr(main.settings, "mock_mode", False)
    monkeypatch.setattr(main.state, "pipeline", None)
    assert client.get("/ready", headers=HEADERS).status_code == 503
    r = client.post("/analyze/image", json={"path": "sample.png"}, headers=HEADERS)
    assert r.status_code == 503 and "loading" in r.json()["detail"]
    monkeypatch.setattr(main.state, "error", "FileNotFoundError: weights/gend/config.json")
    r = client.get("/ready", headers=HEADERS)
    assert r.status_code == 503 and "download_weights" in r.json()["detail"]


def test_filters_need_the_real_pipeline(client):
    body = {"path": "sample.png", "family": "F1", "strength": 0.5}
    assert client.post("/compare", json=body, headers=HEADERS).status_code == 501
    assert client.post("/beautify", json=body, headers=HEADERS).status_code == 501


def test_sniff():
    assert sniff(b"\xff\xd8\xff\xe0" + b"\0" * 12) == ("image", "jpeg")
    assert sniff(b"\0\0\0\x18ftypmp42" + b"\0" * 4) == ("video", "mp4")
    assert sniff(b"hello world, plain") is None
