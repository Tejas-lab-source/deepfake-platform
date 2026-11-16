"""Sample frames from each listed video, detect the largest face with MTCNN, and save the
two detector crops (lossless PNG). Writes manifest.json with identity-disjoint subsets.

    python -m offline.extract_faces --data-root DIR --out artifacts/faces
        [--frames-per-video 8] [--seed 0] [--limit N] [--max-minutes M]

Crops use the service's own code (app.faces), so they match inference exactly. Frames
with no detected face are excluded and listed, never scored. The run is resumable: each
finished video is written atomically and skipped on the next run; the manifest is written
only when every video is done.

Split assignment: only the plumbing fixture can be split for now. Real-data assignment
awaits review of the official list's identity graph (offline/celebdf_graph.py).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from offline import common
from offline.celebdf import VideoEntry, identity_components, parse_test_list
from offline.fixture import is_fixture
from offline.splits import (
    DEFAULT_SUBSET_RATIOS,
    RealDataSplitPending,
    assign_components,
    check_face_leakage,
    check_no_leakage,
)

LIST_NAME = "List_of_testing_videos.txt"


def sample_frame_indices(frame_count: int, k: int) -> list[int]:
    """Up to k uniformly spaced frame indices (unique, ascending)."""
    if frame_count <= 0 or k <= 0:
        return []
    return sorted({int(round(x)) for x in np.linspace(0, frame_count - 1, k)})


def read_frames(path: Path, k: int) -> list[tuple[int, NDArray[np.uint8]]]:
    """Decode the sampled frames as RGB. Reads sequentially, which is reliable for every
    container OpenCV supports (seeking is not)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open video {path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames: list[NDArray[np.uint8]] = []
        if count <= 0:  # unknown length: decode everything once
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(bgr)
            wanted = sample_frame_indices(len(frames), k)
            return [(i, frames[i][..., ::-1].copy()) for i in wanted]
        wanted_set = set(sample_frame_indices(count, k))
        out = []
        for index in range(count):
            ok, bgr = cap.read()
            if not ok:
                break
            if index in wanted_set:
                out.append((index, bgr[..., ::-1].copy()))
        return out
    finally:
        cap.release()


def plan_subsets(entries: list[VideoEntry], data_root: Path, seed: int) -> dict[str, str]:
    """{video_id: subset}. Refuses real data until the identity graph has been reviewed."""
    if not is_fixture(data_root):
        raise RealDataSplitPending(
            "Real-data split assignment is not implemented yet: it awaits review of the "
            "official list's identity graph. Run `python -m offline.celebdf_graph --list "
            f"{data_root / LIST_NAME}` and share the report (see offline/README.md)."
        )
    components = identity_components(entries)
    by_group = assign_components(components, DEFAULT_SUBSET_RATIOS, seed)
    return {v.video_id: by_group[c.group_id] for c in components for v in c.videos}


def extract_video(
    entry: VideoEntry,
    subset: str,
    data_root: Path,
    out: Path,
    frames_per_video: int,
    detector: Any,
    settings: Any,
) -> dict[str, Any]:
    from app.faces import gend_crop, sbi_crop

    faces, excluded = [], []
    for index, rgb in read_frames(data_root / entry.path, frames_per_video):
        detected = detector.detect(rgb)
        if not detected:
            excluded.append({"frame_index": index, "reason": "no_face_detected"})
            continue
        face = detected[0]  # largest
        try:
            g = gend_crop(rgb, face.points, settings.gend_crop_size, settings.gend_crop_scale)
            s = sbi_crop(rgb, face.box)
        except ValueError as exc:
            excluded.append({"frame_index": index, "reason": f"crop_failed: {exc}"})
            continue
        face_id = f"{entry.video_id}#{index:05d}"
        rel = Path("crops") / entry.video_id
        g_path, s_path = rel / f"{index:05d}_gend.png", rel / f"{index:05d}_sbi.png"
        common.atomic_write_png(out / g_path, g)
        common.atomic_write_png(out / s_path, s)
        faces.append(
            {
                "face_id": face_id,
                "video_id": entry.video_id,
                "subset": subset,
                "y": entry.y,
                "frame_index": index,
                "detection_prob": round(face.prob, 5),
                "box": [round(v, 2) for v in face.box],
                "gend_crop": str(g_path),
                "sbi_crop": str(s_path),
            }
        )
    return {
        "video": {
            "video_id": entry.video_id,
            "path": entry.path,
            "label": entry.label,
            "y": entry.y,
            "identities": list(entry.identities),
            "subset": subset,
        },
        "faces": faces,
        "excluded_frames": excluded,
    }


def build_manifest(
    shards: list[dict[str, Any]], data_root: Path, list_sha: str, args: argparse.Namespace
) -> dict[str, Any]:
    videos = [s["video"] for s in shards]
    faces = [f for s in shards for f in s["faces"]]
    check_no_leakage(videos)
    check_face_leakage(faces, {v["video_id"]: v["subset"] for v in videos})
    counts: dict[str, dict[str, int]] = {}
    for subset in common.SUBSETS:
        sub_faces = [f for f in faces if f["subset"] == subset]
        sub_videos = [v for v in videos if v["subset"] == subset]
        counts[subset] = {
            "videos": len(sub_videos),
            "faces": len(sub_faces),
            "real_faces": sum(1 for f in sub_faces if f["y"] == 0),
            "fake_faces": sum(1 for f in sub_faces if f["y"] == 1),
            "real_videos": sum(1 for v in sub_videos if v["y"] == 0),
            "fake_videos": sum(1 for v in sub_videos if v["y"] == 1),
        }
    total_videos = len(videos)
    fixture = is_fixture(data_root)
    return {
        "schema": 1,
        "protocol": common.PROTOCOL_LABEL,
        "fixture": fixture,
        "fixture_notice": common.FIXTURE_LABEL if fixture else None,
        "list_sha256": list_sha,
        "frames_per_video": args.frames_per_video,
        "seed": args.seed,
        "target_ratios": DEFAULT_SUBSET_RATIOS,
        "achieved_video_ratios": {
            s: round(counts[s]["videos"] / total_videos, 4) for s in common.SUBSETS
        },
        "counts": counts,
        "videos": videos,
        "faces": faces,
        "excluded_frames": [
            {"video_id": s["video"]["video_id"], **e} for s in shards for e in s["excluded_frames"]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/faces"))
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="stop after this many videos (resumable)")
    parser.add_argument("--max-minutes", type=float, help="stop after this long (resumable)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    log = common.setup_logging()

    list_path = args.data_root / LIST_NAME
    list_text = list_path.read_text()
    entries = parse_test_list(list_text)
    try:
        subsets = plan_subsets(entries, args.data_root, args.seed)
    except RealDataSplitPending as exc:
        log.error("%s", exc)
        return 3

    import torch

    from app.config import Settings
    from app.faces import FaceDetector

    settings = Settings()
    detector = FaceDetector(
        torch.device(args.device), settings.face_min_prob, settings.face_min_size
    )
    shard_dir = args.out / "videos"
    todo = [e for e in entries if not (shard_dir / f"{e.video_id}.json").is_file()]
    log.info("%d videos listed, %d already extracted", len(entries), len(entries) - len(todo))
    limiter = common.RunLimiter(args.limit, args.max_minutes)
    for n, entry in enumerate(todo):
        if limiter.should_stop():
            log.info("run limit reached; re-run the same command to resume")
            return 0
        shard = extract_video(
            entry,
            subsets[entry.video_id],
            args.data_root,
            args.out,
            args.frames_per_video,
            detector,
            settings,
        )
        common.atomic_write_json(shard_dir / f"{entry.video_id}.json", shard)
        limiter.tick()
        log.info(
            "%s: %d faces, %d frames excluded | %s",
            entry.video_id,
            len(shard["faces"]),
            len(shard["excluded_frames"]),
            limiter.progress(len(todo) - n - 1),
        )

    shards = [json.loads((shard_dir / f"{e.video_id}.json").read_text()) for e in entries]
    manifest = build_manifest(shards, args.data_root, common.sha256_text(list_text), args)
    common.atomic_write_json(args.out / "manifest.json", manifest)
    c = Counter(f["subset"] for f in manifest["faces"])
    log.info(
        "manifest written: %d faces %s, %d frames excluded",
        len(manifest["faces"]),
        dict(c),
        len(manifest["excluded_frames"]),
    )
    if manifest["fixture"]:
        log.warning(common.FIXTURE_LABEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
