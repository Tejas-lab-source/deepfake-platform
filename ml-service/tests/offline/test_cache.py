import json

import numpy as np
import pytest

from offline import common, copies, generic_aug
from offline.cache_features import load_meta, load_split
from tests.conftest import needs_landmarker, needs_ml


@needs_ml
def test_plans_are_deterministic_and_follow_the_inventory():
    names = {n for c in copies.COPIES_BY_SPLIT.values() for n in c}
    for name in names:
        assert copies.plan_copy("vid#00001", name) == copies.plan_copy("vid#00001", name)
    assert copies.plan_copy("vid#00001", "F1_lo") != copies.plan_copy("vid#00002", "F1_lo")
    for k in range(200):
        fid = f"v#{k}"
        lo, hi = copies.plan_copy(fid, "F2_lo"), copies.plan_copy(fid, "F2_hi")
        assert 0.30 <= lo.filters[0][1] < 0.65 <= hi.filters[0][1] <= 1.0
        assert copies.plan_copy(fid, "F1_bilat").filters[0][2] == "bilateral"
        assert copies.plan_copy(fid, "F1_hi").filters[0][2] in copies.F1_METHODS
        mix = copies.plan_copy(fid, "mix")
        assert len(set(mix.families)) == 2 and list(mix.families) == sorted(mix.families)
        assert 0.5 <= copies.plan_copy(fid, "F4").filters[0][1] <= 1.0
    with pytest.raises(ValueError):
        copies.plan_copy("x", "F9")


def test_inventory_has_no_duplicate_work():
    by = copies.COPIES_BY_SPLIT
    assert "mix" not in by["train"] and "F4" not in by["train"] + by["val"]
    assert "F1_bilat" not in by["val"] + by["test"] and "generic" not in by["val"] + by["test"]
    for split in by.values():
        assert len(split) == len(set(split))
    # Inference-style slots reuse cached single-family copies: only "mix" is extra.
    for k in range(50):
        inf = copies.inference_copies(f"v#{k}")
        assert inf[0] == "clean" and inf[-1] == "mix"
        assert all(c in copies.SINGLE for c in inf[1:4])


def test_inference_pick_is_half_lo_half_hi():
    picks = [copies.inference_pick(f"face{k}")["F1"] for k in range(4000)]
    share = picks.count("F1_lo") / len(picks)
    assert 0.47 < share < 0.53


@needs_ml
def test_generic_augmentation_is_deterministic_and_changes_the_image():
    img = (np.arange(64 * 64 * 3) % 251).astype(np.uint8).reshape(64, 64, 3)
    params = generic_aug.draw_params(np.random.default_rng(3))
    a, b = generic_aug.apply(img, params), generic_aug.apply(img, params)
    assert np.array_equal(a, b) and a.shape == img.shape and not np.array_equal(a, img)


@needs_landmarker
def test_copies_render_identically_twice_and_f4_is_skipped(crop, geom, tmp_path):
    for name in ("F1_lo", "F2_hi", "F3_hi", "mix", "generic", "F1_bilat"):
        plan = copies.plan_copy("face#1", name)
        a, ta = copies.render_copy(plan, crop, geom, tmp_path)
        b, _ = copies.render_copy(plan, crop, geom, tmp_path)
        assert np.array_equal(a, b) and not np.array_equal(a, crop)
        assert (ta is not None) == ("F3" in plan.families)
    with pytest.raises(copies.CopyUnavailable):
        copies.render_copy(copies.plan_copy("face#1", "F4"), crop, geom, tmp_path)


def test_cache_contents_and_split_guard(cache_dir):
    meta = load_meta(cache_dir)
    assert meta["fixture"] is True and meta["f4_copies_present"] == 0
    assert meta["counts"]["train"]["faces"] == 8 and meta["counts"]["test"]["faces"] == 4
    train = load_split(cache_dir, "train", "train_head")
    val = load_split(cache_dir, "val", "calibrate")
    assert train["copy_names"] == list(copies.COPIES_BY_SPLIT["train"])
    assert train["feats"].dtype == np.float16 and train["feats"].shape[1:] == (9, 1024)
    assert train["present"].all()  # every train copy exists
    assert np.isnan(train["sbi"]).all()  # SBI is not computed for train: not needed
    assert np.isfinite(val["sbi"]).all() and np.isfinite(val["bl"]).all()
    assert set(val["subset"]) == {"val-A", "val-B"}
    assert all(set(p) == {"F1", "F2", "F3"} for p in val["inference_pick"])
    for stage in ("train_head", "calibrate"):
        with pytest.raises(common.SplitAccessError):
            load_split(cache_dir, "test", stage)
    test = load_split(cache_dir, "test", "eval_lofo")
    assert not test["present"][:, test["copy_names"].index("F4")].any()  # no GFPGAN here
    # Each split file is what meta.json recorded.
    for split, sha in meta["split_files_sha256"].items():
        assert common.sha256_file(cache_dir / f"features_{split}.npz") == sha


def test_cached_features_match_a_fresh_computation(cache_dir, faces_dir):
    """Regenerate one copy from its plan and re-run GenD: the cached feature must match."""
    import torch

    from app.detectors.gend import GenDDetector
    from app.faces import Landmarker
    from tests.conftest import LANDMARKER, WEIGHTS

    val = load_split(cache_dir, "val", "calibrate")
    manifest = json.loads((faces_dir / "manifest.json").read_text())
    face = next(f for f in manifest["faces"] if f["face_id"] == val["face_id"][0])
    crop = common.read_png(faces_dir / face["gend_crop"])
    lm = Landmarker(LANDMARKER)
    try:
        geom = lm.geometry(crop)
    finally:
        lm.close()
    j = val["copy_names"].index("F3_hi")
    img, _ = copies.render_copy(copies.plan_copy(face["face_id"], "F3_hi"), crop, geom, WEIGHTS)
    feats, _ = GenDDetector(WEIGHTS, torch.device("cpu")).score([img])
    assert np.allclose(val["feats"][0, j].astype(np.float32), feats[0], atol=2e-2, rtol=1e-2)
