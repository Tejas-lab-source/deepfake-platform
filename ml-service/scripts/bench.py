"""Print pipeline latency on this machine.

    python scripts/bench.py [image]        (run from ml-service/, weights downloaded)

With no image it uses scikit-image's public-domain astronaut photo.
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.head import load_calibration  # noqa: E402
from app.pipeline import RealPipeline  # noqa: E402


def main() -> None:
    import torch
    from PIL import Image

    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        settings.upload_dir = Path(tmp)
        if len(sys.argv) > 1:
            image = Path(sys.argv[1])
        else:
            from skimage import data

            image = Path(tmp) / "astronaut.png"
            Image.fromarray(data.astronaut()).save(image)

        start = time.perf_counter()
        pipeline = RealPipeline(settings, load_calibration(settings.model_card_path))
        info = pipeline.info()
        print(f"device={info['device']} fp16={info['fp16']} threads={torch.get_num_threads()}")
        print(f"sbiLoaded={info['sbiLoaded']} headLoaded={info['headLoaded']}")
        print(f"model load: {time.perf_counter() - start:.1f} s\n")

        pipeline.analyze_image(image, 0, False)  # warm-up
        print(
            f"{'config':<22}{'total':>8}{'detect':>8}{'beautify':>9}{'score':>8}{'explain':>8}  ms"
        )
        for k, explain in ((0, False), (4, False), (4, True)):
            runs = []
            for _ in range(3):
                t0 = time.perf_counter()
                result = pipeline.analyze_image(image, k, explain)
                runs.append(((time.perf_counter() - t0) * 1000, result["timingsMs"]))
            total = statistics.median(r[0] for r in runs)
            t = sorted(runs, key=lambda r: r[0])[1][1]
            label = f"K={k} explain={'on' if explain else 'off'}"
            print(
                f"{label:<22}{total:>8.0f}{t['detect']:>8}{t['beautify']:>9}"
                f"{t['score']:>8}{t['explain']:>8}"
            )
        pipeline.close()


if __name__ == "__main__":
    main()
