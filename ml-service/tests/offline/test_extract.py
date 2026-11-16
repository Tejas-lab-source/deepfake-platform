import json
import shutil

import pytest

from offline import extract_faces
from offline.common import SUBSETS, read_png
from offline.fixture import MARKER
from tests.conftest import needs_ml

pytestmark = needs_ml


def test_sample_frame_indices():
    assert extract_faces.sample_frame_indices(300, 8) == [0, 43, 85, 128, 171, 214, 256, 299]
    assert extract_faces.sample_frame_indices(3, 8) == [0, 1, 2]
    assert extract_faces.sample_frame_indices(0, 8) == []


def test_fixture_videos_decode(fixture_root):
    frames = extract_faces.read_frames(fixture_root / "Celeb-synthesis/id0_id5_0000.mp4", 8)
    assert [i for i, _ in frames] == [0, 1, 2]
    assert frames[0][1].shape == (320, 320, 3)


def test_manifest_is_complete_and_leak_free(faces_dir):
    m = json.loads((faces_dir / "manifest.json").read_text())
    assert m["fixture"] is True and "PLUMBING FIXTURE" in m["fixture_notice"]
    assert "NOT comparable" in m["protocol"]
    assert len(m["faces"]) == 20 and len(m["videos"]) == 10
    # The blank frame is excluded and recorded, never scored.
    assert m["excluded_frames"] == [
        {
            "video_id": "Celeb-synthesis__id0_id5_0000",
            "frame_index": 2,
            "reason": "no_face_detected",
        }
    ]
    # Every subset has both classes, and whole components stay together.
    for subset in SUBSETS:
        assert m["counts"][subset]["real_faces"] > 0 and m["counts"][subset]["fake_faces"] > 0
    subset_of = {v["video_id"]: v["subset"] for v in m["videos"]}
    for k in range(5):
        assert (
            subset_of[f"Celeb-real__id{k}_0000"]
            == subset_of[f"Celeb-synthesis__id{k}_id{k + 5}_0000"]
        )
    face = m["faces"][0]
    assert read_png(faces_dir / face["gend_crop"]).shape == (224, 224, 3)
    assert read_png(faces_dir / face["sbi_crop"]).shape == (380, 380, 3)


def test_resume_skips_finished_videos(fixture_root, faces_dir, tmp_path, caplog):
    out = tmp_path / "faces"
    shutil.copytree(faces_dir / "videos", out / "videos")
    shutil.copytree(faces_dir / "crops", out / "crops")
    (out / "videos" / "Celeb-real__id3_0000.json").unlink()
    with caplog.at_level("INFO"):
        assert extract_faces.main(["--data-root", str(fixture_root), "--out", str(out)]) == 0
    assert "10 videos listed, 9 already extracted" in caplog.text
    first = json.loads((faces_dir / "manifest.json").read_text())
    again = json.loads((out / "manifest.json").read_text())
    assert [f["face_id"] for f in again["faces"]] == [f["face_id"] for f in first["faces"]]


def test_run_limit_stops_before_the_manifest(fixture_root, tmp_path):
    out = tmp_path / "faces"
    args = ["--data-root", str(fixture_root), "--out", str(out), "--limit", "3"]
    assert extract_faces.main(args) == 0
    assert len(list((out / "videos").glob("*.json"))) == 3
    assert not (out / "manifest.json").exists()


def test_real_data_split_is_refused(fixture_root, tmp_path):
    real = tmp_path / "celebdf"
    shutil.copytree(fixture_root, real)
    (real / MARKER).unlink()
    assert extract_faces.main(["--data-root", str(real), "--out", str(tmp_path / "o")]) == 3
    assert not (tmp_path / "o").exists()


def test_bad_list_is_refused(fixture_root, tmp_path):
    from offline.celebdf import ListFormatError

    bad = tmp_path / "bad"
    shutil.copytree(fixture_root, bad)
    (bad / "List_of_testing_videos.txt").write_text("0 Celeb-real/id0_0000.mp4\n")
    with pytest.raises(ListFormatError):
        extract_faces.main(["--data-root", str(bad), "--out", str(tmp_path / "o")])
