"""Print pipeline latency on this machine.

    python scripts/bench.py [image] [--runs N]      (run from ml-service/, weights downloaded)

With no image it uses scikit-image's public-domain astronaut photo. It reports:

* each detector on its own (forward pass and Grad-CAM);
* the full pipeline with GenD only and with GenD + SBI, for K = 0 and K = 4, with heatmaps
  off and on.

K only applies to GenD: SBI always scores the clean crop once.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import explain  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.head import load_calibration  # noqa: E402
from app.pipeline import RealPipeline  # noqa: E402


def median_ms(fn: Callable[[], Any], runs: int) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main() -> None:
    import torch
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        settings.upload_dir = Path(tmp)
        image: Path = args.image
        if image is None:
            from skimage import data

            image = Path(tmp) / "astronaut.png"
            Image.fromarray(data.astronaut()).save(image)

        start = time.perf_counter()
        pipeline = RealPipeline(settings, load_calibration(settings.model_card_path))
        info = pipeline.info()
        print(f"device={info['device']} fp16={info['fp16']} threads={torch.get_num_threads()}")
        print(f"sbiLoaded={info['sbiLoaded']} headLoaded={info['headLoaded']} runs={args.runs}")
        print(f"model load: {time.perf_counter() - start:.1f} s")

        rgb, _ = pipeline._load(image)
        faces = pipeline.detector.detect(rgb)
        if not faces:
            sys.exit("no face detected in the image")
        crop, crop_sbi = pipeline._crops(rgb, faces[0])
        gend, sbi = pipeline.gend, pipeline.sbi
        gend.score([crop])  # warm-up
        if sbi is not None:
            sbi.score([crop_sbi])

        print("\nDetectors alone (median ms)")
        rows: list[tuple[str, Callable[[], Any]]] = [
            ("GenD forward, 1 crop (K=0)", lambda: gend.score([crop])),
            ("GenD forward, 5 crops (K=4)", lambda: gend.score([crop] * 5)),
            ("GenD Grad-CAM, 1 crop", lambda: explain.gradcam_gend(gend, [crop])),
            ("GenD Grad-CAM, 2 crops", lambda: explain.gradcam_gend(gend, [crop] * 2)),
        ]
        if sbi is not None:
            rows += [
                ("SBI forward, 1 crop", lambda: sbi.score([crop_sbi])),
                ("SBI Grad-CAM, 1 crop", lambda: explain.gradcam_sbi(sbi, [crop_sbi])),
            ]
        for label, fn in rows:
            print(f"  {label:<30}{median_ms(fn, args.runs):>8.0f}")

        modes: list[tuple[str, Any]] = [("GenD only", None)]
        if sbi is not None:
            modes.append(("GenD + SBI", sbi))
        print("\nFull pipeline, one face (median total; stages from the median run), ms")
        header = f"  {'detectors':<12}{'K':>3}{'heatmaps':>10}{'total':>8}{'detect':>8}"
        print(header + f"{'beautify':>10}{'score':>8}{'explain':>9}")
        for label, model in modes:
            pipeline.sbi = model
            for k, heatmaps in ((0, False), (0, True), (4, False), (4, True)):
                runs = []
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    result = pipeline.analyze_image(image, k, heatmaps)
                    runs.append(((time.perf_counter() - t0) * 1000, result["timingsMs"]))
                runs.sort(key=lambda r: r[0])
                total, t = runs[len(runs) // 2]
                print(
                    f"  {label:<12}{k:>3}{'on' if heatmaps else 'off':>10}{total:>8.0f}"
                    f"{t['detect']:>8}{t['beautify']:>10}{t['score']:>8}{t['explain']:>9}"
                )
        pipeline.sbi = sbi
        pipeline.close()


if __name__ == "__main__":
    main()
