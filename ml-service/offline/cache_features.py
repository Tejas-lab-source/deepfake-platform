"""Compute and cache, per face: GenD features for every copy in the inventory (fp16), plus
the SBI logit and beautifiedLikelihood of the clean crop on val and test.

    python -m offline.cache_features --faces artifacts/faces --out artifacts/cache
        [--weights-dir weights] [--batch-size 16] [--device cpu] [--limit N] [--max-minutes M]

Forward-only and resumable: one shard per video, written atomically and skipped when it
exists. When every video is done the shards are consolidated into one file per split
(features_train.npz, features_val.npz, features_test.npz) plus meta.json. Stages read the
split files only through `load_split`, which refuses the test split to training and
calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from offline import common
from offline.copies import (
    COPIES_BY_SPLIT,
    INVENTORY_VERSION,
    CopyUnavailable,
    inference_pick,
    plan_copy,
    render_copy,
)
from offline.heads import FEATURE_DIM


def load_split(cache_dir: Path, split: str, stage: str) -> dict[str, Any]:
    """Load one consolidated split. Raises SplitAccessError if `stage` may not read it."""
    common.check_split_access(stage, split)
    with np.load(cache_dir / f"features_{split}.npz", allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    out["plans"] = [json.loads(s) for s in out.pop("plans_json")]
    out["inference_pick"] = [json.loads(s) for s in out.pop("inference_pick_json")]
    out["copy_names"] = [str(c) for c in out["copy_names"]]
    return out


def load_meta(cache_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((cache_dir / "meta.json").read_text())
    return meta


class Models:
    def __init__(self, weights_dir: Path, device: str) -> None:
        import torch

        from app.detectors.gend import GenDDetector
        from app.detectors.sbi import SBIDetector
        from app.faces import Landmarker

        dev = torch.device(device)
        self.gend = GenDDetector(weights_dir, dev)
        sbi_path = weights_dir / "sbi" / "FFc23.tar"
        self.sbi = SBIDetector(sbi_path, dev) if sbi_path.is_file() else None
        self.landmarker = Landmarker(weights_dir / "mediapipe" / "face_landmarker.task")
        self.weights_dir = weights_dir

    def close(self) -> None:
        self.landmarker.close()


def cache_video(
    faces: list[dict[str, Any]], faces_dir: Path, models: Models, batch_size: int
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    from app.pipeline import beautified_likelihood

    split = common.SPLIT_OF_SUBSET[faces[0]["subset"]]
    names = COPIES_BY_SPLIT[split]
    images: list[NDArray[np.uint8]] = []
    slots: list[tuple[int, int]] = []
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for face in faces:
        crop = common.read_png(faces_dir / face["gend_crop"])
        geom = models.landmarker.geometry(crop)
        if geom is None:
            excluded.append({"face_id": face["face_id"], "reason": "no_landmarks"})
            continue
        i = len(rows)
        plans, present, strengths = [], np.zeros(len(names), bool), np.full(len(names), np.nan)
        for j, name in enumerate(names):
            plan = plan_copy(face["face_id"], name)
            plans.append(plan.to_json())
            try:
                img, _ = render_copy(plan, crop, geom, models.weights_dir)
            except CopyUnavailable:
                continue
            present[j] = True
            if plan.filters:
                strengths[j] = plan.filters[0][1] if len(plan.filters) == 1 else np.nan
            images.append(img)
            slots.append((i, j))
        sbi_logit = likelihood = float("nan")
        if split != "train":
            if models.sbi is not None:
                sbi_crop = common.read_png(faces_dir / face["sbi_crop"])
                sbi_logit = float(models.sbi.score([sbi_crop])[0])
            bl = beautified_likelihood(crop, geom)
            likelihood = float("nan") if bl is None else bl
        rows.append(
            {
                "face": face,
                "plans": plans,
                "present": present,
                "strengths": strengths,
                "sbi": sbi_logit,
                "bl": likelihood,
                "pick": inference_pick(face["face_id"]) if split != "train" else {},
            }
        )

    feats = np.zeros((len(rows), len(names), FEATURE_DIM), dtype=np.float16)
    for batch in common.chunks(list(range(len(images))), batch_size):
        features, _ = models.gend.score([images[k] for k in batch])
        for k, vec in zip(batch, features, strict=True):
            i, j = slots[k]
            feats[i, j] = vec.astype(np.float16)

    shard = {
        "split": np.array(split),
        "copy_names": np.array(names),
        "face_id": np.array([r["face"]["face_id"] for r in rows], dtype=str),
        "video_id": np.array([r["face"]["video_id"] for r in rows], dtype=str),
        "subset": np.array([r["face"]["subset"] for r in rows], dtype=str),
        "y": np.array([r["face"]["y"] for r in rows], dtype=np.int8),
        "feats": feats,
        "present": np.array([r["present"] for r in rows], dtype=bool).reshape(
            len(rows), len(names)
        ),
        "strengths": np.array([r["strengths"] for r in rows], dtype=np.float32).reshape(
            len(rows), len(names)
        ),
        "sbi": np.array([r["sbi"] for r in rows], dtype=np.float32),
        "bl": np.array([r["bl"] for r in rows], dtype=np.float32),
        "plans_json": np.array([json.dumps(r["plans"]) for r in rows], dtype=str),
        "inference_pick_json": np.array([json.dumps(r["pick"]) for r in rows], dtype=str),
    }
    return shard, excluded


def consolidate(out: Path, video_ids: dict[str, list[str]]) -> dict[str, str]:
    """Concatenate shards into one file per split. Returns {split: sha256}."""
    hashes = {}
    for split, vids in video_ids.items():
        shards = []
        for vid in sorted(vids):
            with np.load(out / "shards" / split / f"{vid}.npz", allow_pickle=False) as d:
                shards.append({k: d[k] for k in d.files})
        names = COPIES_BY_SPLIT[split]
        merged: dict[str, Any] = {"copy_names": np.array(names), "split": np.array(split)}
        for key in shards[0]:
            if key in ("copy_names", "split", "excluded_json"):
                continue
            merged[key] = np.concatenate([s[key] for s in shards if len(s["face_id"])], axis=0)
        path = out / f"features_{split}.npz"
        common.atomic_save_npz(path, **merged)
        hashes[split] = common.sha256_file(path)
    return hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--faces", type=Path, default=Path("artifacts/faces"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="stop after this many videos (resumable)")
    parser.add_argument("--max-minutes", type=float, help="stop after this long (resumable)")
    args = parser.parse_args(argv)
    log = common.setup_logging()

    manifest_path = args.faces / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = common.sha256_file(manifest_path)
    by_video: dict[str, list[dict[str, Any]]] = {}
    for face in manifest["faces"]:
        by_video.setdefault(face["video_id"], []).append(face)
    split_of_video = {
        v["video_id"]: common.SPLIT_OF_SUBSET[v["subset"]] for v in manifest["videos"]
    }

    def shard_path(vid: str) -> Path:
        return args.out / "shards" / split_of_video[vid] / f"{vid}.npz"

    stamp = args.out / "manifest.sha256"
    if stamp.is_file() and stamp.read_text().strip() != manifest_sha:
        log.error("%s belongs to a different manifest; use a fresh --out folder", args.out)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    stamp.write_text(manifest_sha + "\n")

    todo = [vid for vid in sorted(by_video) if not shard_path(vid).is_file()]
    log.info("%d videos with faces, %d already cached", len(by_video), len(by_video) - len(todo))
    if todo:
        import torch

        torch.set_num_threads(max(1, torch.get_num_threads()))
        models = Models(args.weights_dir, args.device)
        log.info("SBI loaded: %s", models.sbi is not None)
        limiter = common.RunLimiter(args.limit, args.max_minutes)
        try:
            for n, vid in enumerate(todo):
                if limiter.should_stop():
                    log.info("run limit reached; re-run the same command to resume")
                    return 0
                shard, excluded = cache_video(by_video[vid], args.faces, models, args.batch_size)
                shard["excluded_json"] = np.array(json.dumps(excluded))
                common.atomic_save_npz(shard_path(vid), **shard)
                limiter.tick()
                log.info(
                    "%s: %d faces | %s",
                    vid,
                    len(shard["face_id"]),
                    limiter.progress(len(todo) - n - 1),
                )
        finally:
            models.close()
        sbi_available, f4_available = models.sbi is not None, None
    else:
        sbi_available, f4_available = None, None

    video_ids: dict[str, list[str]] = {s: [] for s in common.SPLITS}
    excluded_all: list[dict[str, str]] = []
    f4_present = 0
    for vid in by_video:
        video_ids[split_of_video[vid]].append(vid)
        with np.load(shard_path(vid), allow_pickle=False) as d:
            excluded_all += json.loads(str(d["excluded_json"]))
            if split_of_video[vid] == "test":
                f4_present += int(d["present"][:, list(d["copy_names"]).index("F4")].sum())
    hashes = consolidate(args.out, video_ids)
    from app.beautify import f4_restore

    counts = {}
    for split in common.SPLITS:
        with np.load(args.out / f"features_{split}.npz", allow_pickle=False) as d:
            counts[split] = {"faces": int(len(d["face_id"])), "fake": int(d["y"].sum())}
            if split != "train":
                counts[split]["sbi_missing"] = int(np.isnan(d["sbi"]).sum())
    from offline.heads import gend_head_from_checkpoint

    w, b = gend_head_from_checkpoint(args.weights_dir / "gend")
    common.atomic_save_npz(args.out / "gend_head.npz", weight=w, bias=np.float32(b))
    meta = {
        "schema": 1,
        "inventory_version": INVENTORY_VERSION,
        "copies_by_split": {s: list(c) for s, c in COPIES_BY_SPLIT.items()},
        "manifest_sha256": manifest_sha,
        "fixture": manifest["fixture"],
        "protocol": manifest["protocol"],
        "split_files_sha256": hashes,
        "counts": counts,
        "excluded_faces": excluded_all,
        "sbi_available_last_run": sbi_available,
        "f4_available": (
            f4_available if f4_available is not None else f4_restore.available(args.weights_dir)
        ),
        "f4_copies_present": f4_present,
        "feature_dtype": "float16",
        "feature": "GenD CLIP ViT-L/14 pooler_output (1024-d), before L2 normalisation",
    }
    common.atomic_write_json(args.out / "meta.json", meta)
    log.info(
        "cache consolidated: %s; %d faces excluded; F4 copies: %d",
        counts,
        len(excluded_all),
        f4_present,
    )
    if manifest["fixture"]:
        log.warning(common.FIXTURE_LABEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
