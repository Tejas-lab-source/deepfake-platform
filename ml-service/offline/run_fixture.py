"""Run every Phase 3 stage end to end on the plumbing fixture, on CPU.

    python -m offline.run_fixture [--workdir artifacts/fixture_run] [--iou-faces -1]

Stages: generate fixture -> identity-graph report -> extract faces -> cache features ->
train heads (6 schemes x 3 folds x 3 seeds + shipped head) -> calibrate -> evaluate ->
load the result into the service's real pipeline and analyse one image.

Everything is written under --workdir, including head.pt and model_card.json. Nothing is
written to ml-service/weights/. FIXTURE OUTPUT IS A PLUMBING TEST, NOT AN ACCURACY RESULT.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, default=Path("artifacts/fixture_run"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--iou-faces", default="-1", help="-1: all fixture test faces")
    parser.add_argument("--bootstrap", default="200")
    args = parser.parse_args(argv)

    from offline import (
        cache_features,
        calibrate,
        celebdf_graph,
        common,
        eval_lofo,
        extract_faces,
        train_head,
    )
    from offline.fixture import write_fixture

    log = common.setup_logging()
    w = args.workdir.resolve()
    if (w / "weights").resolve() == args.weights_dir.resolve():
        log.error("refusing to write fixture outputs into the real weights folder")
        return 2
    paths = {
        "data": w / "fixture",
        "faces": w / "faces",
        "cache": w / "cache",
        "heads": w / "heads",
        "head": w / "weights" / "head.pt",
        "card": w / "weights" / "model_card.json",
        "eval": w / "eval" / "lofo.json",
        "graph": w / "eval" / "identity_graph.json",
    }
    timings: dict[str, float] = {}

    def stage(name: str, fn: object, stage_args: list[str]) -> None:
        start = time.perf_counter()
        code = fn(stage_args)  # type: ignore[operator]
        timings[name] = round(time.perf_counter() - start, 1)
        if code != 0:
            raise SystemExit(f"stage {name} failed with exit code {code}")
        log.info("stage %s done in %.1f s", name, timings[name])

    start = time.perf_counter()
    write_fixture(paths["data"])
    timings["fixture"] = round(time.perf_counter() - start, 1)
    list_file = str(paths["data"] / extract_faces.LIST_NAME)
    stage(
        "identity_graph", celebdf_graph.main, ["--list", list_file, "--json", str(paths["graph"])]
    )
    stage(
        "extract_faces",
        extract_faces.main,
        ["--data-root", str(paths["data"]), "--out", str(paths["faces"])],
    )
    stage(
        "cache_features",
        cache_features.main,
        [
            "--faces",
            str(paths["faces"]),
            "--out",
            str(paths["cache"]),
            "--weights-dir",
            str(args.weights_dir),
        ],
    )
    stage(
        "train_head",
        train_head.main,
        [
            "--cache",
            str(paths["cache"]),
            "--out-dir",
            str(paths["heads"]),
            "--final",
            "--final-out",
            str(paths["head"]),
        ],
    )
    stage(
        "calibrate",
        calibrate.main,
        [
            "--cache",
            str(paths["cache"]),
            "--head",
            str(paths["head"]),
            "--head-report",
            str(paths["heads"] / "final.json"),
            "--out",
            str(paths["card"]),
            "--bootstrap",
            args.bootstrap,
        ],
    )
    stage(
        "eval_lofo",
        eval_lofo.main,
        [
            "--cache",
            str(paths["cache"]),
            "--heads-dir",
            str(paths["heads"]),
            "--final-head",
            str(paths["head"]),
            "--model-card",
            str(paths["card"]),
            "--faces",
            str(paths["faces"]),
            "--weights-dir",
            str(args.weights_dir),
            "--out",
            str(paths["eval"]),
            "--iou-faces",
            args.iou_faces,
            "--bootstrap",
            args.bootstrap,
        ],
    )

    # Load the fixture head and card into the service's real pipeline, as the service would.
    start = time.perf_counter()
    service = check_service(paths, args.weights_dir.resolve(), w)
    timings["service_check"] = round(time.perf_counter() - start, 1)

    summary = {
        "notice": common.FIXTURE_LABEL,
        "outputs": {k: str(v) for k, v in paths.items()},
        "seconds_per_stage": timings,
        "service_check": service,
        "model_card_status": json.loads(paths["card"].read_text())["calibration_status"],
    }
    common.atomic_write_json(w / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def check_service(paths: dict[str, Path], weights_dir: Path, workdir: Path) -> dict[str, object]:
    from PIL import Image
    from skimage import data

    from app.config import Settings
    from app.head import ModelFilesError, load_calibration
    from app.pipeline import RealPipeline

    service_weights = workdir / "service_weights"
    service_weights.mkdir(parents=True, exist_ok=True)
    for sub in ("gend", "clip-vit-large-patch14", "mediapipe", "sbi"):
        link = service_weights / sub
        if (weights_dir / sub).exists() and not link.exists():
            link.symlink_to(weights_dir / sub)
    for name in ("head.pt", "model_card.json"):
        (service_weights / name).write_bytes((paths["head"].parent / name).read_bytes())
    uploads = workdir / "uploads"
    uploads.mkdir(exist_ok=True)
    Image.fromarray(data.astronaut()).save(uploads / "face.png")
    cal = load_calibration(service_weights / "model_card.json")

    refused = False
    try:
        RealPipeline(
            Settings(weights_dir=service_weights, upload_dir=uploads, mock_mode=False), cal
        )
    except ModelFilesError:
        refused = True
    settings = Settings(
        weights_dir=service_weights,
        upload_dir=uploads,
        mock_mode=False,
        allow_unvalidated_model_card=True,
    )
    pipe = RealPipeline(settings, cal)
    try:
        result = pipe.analyze_image(uploads / "face.png", 4, True)
        info = pipe.info()
    finally:
        pipe.close()
    face = result["faces"][0]
    return {
        "refused_without_permission": refused,
        "head_loaded": info["headLoaded"],
        "head_sha256_matches_card": info["headSha256"] == cal.head_sha256,
        "model_card_version": result["modelCardVersion"],
        "verdict": result["verdict"],
        "copies_scored": sorted(face["variantLogits"]),
        "heatmaps_written": sorted(k for k, v in (face["heatmaps"] or {}).items() if v),
    }


if __name__ == "__main__":
    raise SystemExit(main())
