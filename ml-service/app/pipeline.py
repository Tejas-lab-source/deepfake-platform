"""Aggregation and decision rules (pipeline steps 7-9).

Steps 1-6 and 10 (faces, beautified copies, detectors, Grad-CAM) arrive in Phase 2. The
functions here are pure and are shared by the mock path, so mock verdicts follow exactly
the same rules as real ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.head import Calibration, fuse, probability, sigmoid

Verdict = Literal["FAKE", "REAL", "UNCERTAIN"]


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    probability: float
    confident: bool
    uncertain_reason: str | None


def aggregate(variant_logits: dict[str, float]) -> tuple[float, float]:
    """Return (mean logit over copies, spread = max - min)."""
    values = list(variant_logits.values())
    if not values:
        raise ValueError("variant_logits is empty")
    return sum(values) / len(values), max(values) - min(values)


def decide(
    cal: Calibration,
    variant_logits: dict[str, float],
    sbi_logit: float | None,
    beautified_likelihood: float,
) -> Decision:
    mean_logit, spread = aggregate(variant_logits)
    p = probability(cal, fuse(cal, mean_logit, sbi_logit, spread, beautified_likelihood))
    distance = abs(p - cal.threshold)

    # Each copy votes on the same scale as the verdict: fused with the face's other inputs,
    # then temperature-scaled. With identity fusion this is sigmoid(copy_logit / T).
    votes = {
        probability(cal, fuse(cal, v, sbi_logit, spread, beautified_likelihood)) >= cal.threshold
        for v in variant_logits.values()
    }
    detectors_disagree = False
    if sbi_logit is not None:
        p_head, p_sbi = sigmoid(mean_logit), sigmoid(sbi_logit)
        detectors_disagree = (p_head >= 0.5) != (p_sbi >= 0.5) and abs(
            p_head - p_sbi
        ) > cal.detector_disagree_limit

    reason: str | None = None
    if distance < cal.uncertain_margin:
        reason = "near_threshold"
    elif spread > cal.spread_limit:
        reason = "unstable_under_beautification"
    elif len(votes) > 1:
        reason = "variants_disagree"
    elif detectors_disagree:
        reason = "detectors_disagree"

    if reason is not None:
        return Decision("UNCERTAIN", p, False, reason)
    verdict: Verdict = "FAKE" if p >= cal.threshold else "REAL"
    return Decision(verdict, p, distance >= cal.confident_margin, None)


def top_k_mean(logits: list[float], k: int) -> float:
    """Mean of the k largest frame logits (video aggregation)."""
    if not logits:
        raise ValueError("no frame logits")
    top = sorted(logits, reverse=True)[: max(1, k)]
    return sum(top) / len(top)


# --------------------------------------------------------------------------------------
# Real pipeline (steps 2-10). Heavy imports stay inside so mock mode needs none of them.
# --------------------------------------------------------------------------------------

import hashlib  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from app.beautify import Variant
    from app.config import Settings
    from app.faces import DetectedFace, FaceGeometry
    from app.progress import ProgressFn

log = logging.getLogger("ml-service.pipeline")

_VERDICT_RANK = {"FAKE": 2, "UNCERTAIN": 1, "REAL": 0}


class PipelineError(RuntimeError):
    """A request the pipeline cannot serve (bad family, F4 not installed, no face)."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def beautified_likelihood(crop: NDArray[np.uint8], geom: FaceGeometry) -> float | None:
    """Training-free cue for skin smoothing, in [0, 1].

    Compares high-frequency energy (mean absolute Laplacian) inside the skin mask with the
    energy in hair and background: likelihood = 1 / (1 + skin / background). Smoothed skin
    next to sharp hair gives a value near 1. It is a heuristic, and it is unreliable when
    the background is flat or out of focus.
    """
    import cv2
    import numpy as np

    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    energy = np.abs(cv2.Laplacian(grey, cv2.CV_32F))
    skin, background = geom.skin, geom.background
    if skin.sum() < 1.0 or background.sum() < 1.0:
        return None
    e_skin = float((energy * skin).sum() / skin.sum())
    e_bg = float((energy * background).sum() / background.sum())
    return 1.0 / (1.0 + e_skin / max(e_bg, 1e-6))


class RealPipeline:
    """Loads the frozen models once and runs the per-face pipeline."""

    def __init__(self, settings: Settings, calibration: Calibration) -> None:
        import torch

        from app.detectors.gend import GenDDetector
        from app.detectors.sbi import SBIDetector
        from app.faces import FaceDetector, Landmarker
        from app.head import LinearHead, check_model_files

        self.settings = settings
        self.cal = calibration
        if settings.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(settings.device)

        self.detector = FaceDetector(self.device, settings.face_min_prob, settings.face_min_size)
        self.landmarker = Landmarker(settings.landmarker_path)
        self.gend = GenDDetector(settings.weights_dir, self.device)
        self.sbi: SBIDetector | None = None
        if settings.sbi_path.is_file():
            self.sbi = SBIDetector(settings.sbi_path, self.device)
        else:
            log.warning("SBI weights not found; running without the second detector")
        self.head = LinearHead.load(settings.head_path)
        check_model_files(calibration, self.head, settings.allow_unvalidated_model_card)
        if calibration.fixture or calibration.calibration_status not in ("defaults", "ok"):
            log.warning(
                "using an unvalidated model card (%s, fixture=%s): results are not meaningful",
                calibration.calibration_status,
                calibration.fixture,
            )
        settings.derived_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "fp16": self.gend.fp16,
            "gend": "yermandy/GenD_CLIP_L_14",
            "sbiLoaded": self.sbi is not None,
            "headLoaded": self.head is not None,
            "headSha256": None if self.head is None else self.head.sha256,
        }

    # ---- helpers ---------------------------------------------------------------------

    def _load(self, path: Path) -> tuple[NDArray[np.uint8], float]:
        """Read an image as RGB. Returns it (downscaled if huge) and the scale back to the file."""
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
        scale = 1.0
        longest = max(rgb.size)
        if longest > self.settings.max_image_side:
            scale = longest / self.settings.max_image_side
            rgb = rgb.resize((round(rgb.width / scale), round(rgb.height / scale)), Image.LANCZOS)
        return np.asarray(rgb, dtype=np.uint8), scale

    def _save(self, image: NDArray[np.uint8]) -> str:
        """Write an RGB image under derived/ and return its key relative to the upload dir."""
        from PIL import Image

        name = f"{uuid.uuid4().hex}.png"
        Image.fromarray(image).save(self.settings.derived_dir / name)
        return f"derived/{name}"

    @staticmethod
    def _digest(path: Path) -> bytes:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.digest()

    @staticmethod
    def _rng_from(digest: bytes, face_index: int) -> np.random.Generator:
        import numpy as np

        return np.random.default_rng([int.from_bytes(digest[:8], "big"), face_index])

    def _rng(self, path: Path, face_index: int) -> np.random.Generator:
        """Seeded from the file contents, so the same upload gives the same copies."""
        return self._rng_from(self._digest(path), face_index)

    def _crops(self, image: NDArray[np.uint8], face: DetectedFace) -> tuple[Any, Any]:
        from app.faces import gend_crop, sbi_crop

        s = self.settings
        return (
            gend_crop(image, face.points, s.gend_crop_size, s.gend_crop_scale),
            sbi_crop(image, face.box),
        )

    def _head_logits(self, crops: list[NDArray[np.uint8]]) -> list[float]:
        features, gend_logits = self.gend.score(crops)
        logits = self.head.logits(features) if self.head is not None else gend_logits
        return [float(v) for v in logits]

    def _sbi_logit(self, crop: NDArray[np.uint8]) -> float | None:
        return float(self.sbi.score([crop])[0]) if self.sbi is not None else None

    def _aligned_heatmap(self, heatmap: Any, variant: Variant | None) -> Any:
        """Bring a beautified copy's heatmap back onto the clean face (needed after F3)."""
        import numpy as np

        if variant is None or variant.transform is None:
            return heatmap
        return variant.transform.backward(heatmap).astype(np.float32)

    # ---- /analyze/image ----------------------------------------------------------------

    def analyze_image(
        self, path: Path, k: int, explain: bool, progress: ProgressFn | None = None
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        if progress is not None:
            progress("detecting_faces", 0, 1, "images")
        image, scale = self._load(path)
        return self._analyze_rgb(
            image, scale, self._digest(path), k, explain, progress, time.perf_counter() - t0
        )

    def _analyze_rgb(
        self,
        image: NDArray[np.uint8],
        scale: float,
        digest: bytes,
        k: int,
        explain: bool,
        progress: ProgressFn | None = None,
        load_s: float = 0.0,
    ) -> dict[str, Any]:
        """Steps 2-10 on one RGB image (an image upload or one video frame).

        `digest` seeds the beautified copies, so the same input always gives the same copies.
        """
        from app import beautify
        from app import explain as ex

        s, cal = self.settings, self.cal
        timings = {"detect": 0.0, "beautify": 0.0, "score": 0.0, "explain": 0.0}

        t0 = time.perf_counter()
        detected = self.detector.detect(image)[: s.max_faces]
        timings["detect"] = load_s + time.perf_counter() - t0

        base: dict[str, Any] = {"threshold": cal.threshold, "modelCardVersion": cal.version}
        if not detected:
            # Faces that are not detected are excluded, never scored as real.
            return {
                **base,
                "verdict": "UNCERTAIN",
                "probability": None,
                "confident": False,
                "uncertainReason": "no_face_detected",
                "faces": [],
                "timingsMs": {name: round(v * 1000) for name, v in timings.items()},
            }

        records: list[dict[str, Any]] = []
        for index, face in enumerate(detected):
            if progress is not None:
                progress("scoring", index, len(detected), "faces")
            t0 = time.perf_counter()
            crop, crop_sbi = self._crops(image, face)
            geom = self.landmarker.geometry(crop)
            variants: list[Variant] = []
            if geom is not None and k > 0:
                rng = self._rng_from(digest, index)
                variants = beautify.make_variants(
                    crop, geom, k, rng, (s.strength_min, s.strength_max)
                )
            timings["beautify"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            logits = self._head_logits([crop, *(v.image for v in variants)])
            sbi_logit = self._sbi_logit(crop_sbi)
            timings["score"] += time.perf_counter() - t0

            variant_logits = {"clean": logits[0]}
            variant_logits.update({v.name: lg for v, lg in zip(variants, logits[1:], strict=True)})
            mean_logit, spread = aggregate(variant_logits)
            likelihood = beautified_likelihood(crop, geom) if geom is not None else None
            decision = decide(cal, variant_logits, sbi_logit, likelihood or 0.0)
            records.append(
                {
                    "decision": decision,
                    "crop": crop,
                    "crop_sbi": crop_sbi,
                    "variants": variants,
                    "variant_logits": variant_logits,
                    "out": {
                        "box": face.xywh(scale),
                        "detectionProb": round(face.prob, 4),
                        "verdict": decision.verdict,
                        "probability": round(decision.probability, 4),
                        "uncertainReason": decision.uncertain_reason,
                        "scores": {
                            "gendHead": round(mean_logit, 4),
                            "sbi": None if sbi_logit is None else round(sbi_logit, 4),
                        },
                        "variantLogits": {n: round(v, 4) for n, v in variant_logits.items()},
                        "copies": {
                            v.name: [{"family": f, "strength": round(st, 3)} for f, st in v.filters]
                            for v in variants
                        },
                        "spread": round(spread, 4),
                        "beautifiedLikelihood": (
                            None if likelihood is None else round(likelihood, 4)
                        ),
                        "landmarksFound": geom is not None,
                        "crops": None,
                        "heatmaps": None,
                        "heatmapIoU": None,
                    },
                }
            )

        # The deciding face goes first: any FAKE beats UNCERTAIN beats REAL, then by probability.
        records.sort(
            key=lambda r: (_VERDICT_RANK[r["decision"].verdict], r["decision"].probability),
            reverse=True,
        )

        if explain:
            t0 = time.perf_counter()
            explained = records[: s.max_explain_faces]
            if progress is not None:
                progress("aggregating", 0, len(explained), "heatmaps")
            for rec in explained:
                # Compare the clean heatmap with the copy whose logit moved most.
                clean_logit = rec["variant_logits"]["clean"]
                moved: Variant | None = None
                if rec["variants"]:
                    shifts = [
                        abs(rec["variant_logits"][v.name] - clean_logit) for v in rec["variants"]
                    ]
                    moved = rec["variants"][shifts.index(max(shifts))]
                batch = [rec["crop"]] + ([moved.image] if moved is not None else [])
                maps = ex.gradcam_gend(self.gend, batch, self.head)
                keys = {"gend": self._save(ex.colourise(maps[0])), "sbi": None}
                if self.sbi is not None:
                    sbi_map = ex.gradcam_sbi(self.sbi, [rec["crop_sbi"]])[0]
                    keys["sbi"] = self._save(ex.colourise(sbi_map))
                rec["out"]["crops"] = {
                    "gend": self._save(rec["crop"]),
                    "sbi": self._save(rec["crop_sbi"]),
                }
                rec["out"]["heatmaps"] = keys
                if moved is not None:
                    other = self._aligned_heatmap(maps[1], moved)
                    rec["out"]["heatmapIoU"] = round(
                        ex.top_region_iou(maps[0], other, s.heatmap_top_fraction), 4
                    )
                    rec["out"]["heatmapIoUCopy"] = moved.name
            timings["explain"] = time.perf_counter() - t0

        top: Decision = records[0]["decision"]
        return {
            **base,
            "verdict": top.verdict,
            "probability": round(top.probability, 4),
            "confident": top.confident,
            "uncertainReason": top.uncertain_reason,
            "faces": [r["out"] for r in records],
            "timingsMs": {name: round(v * 1000) for name, v in timings.items()},
        }

    # ---- /analyze/video ------------------------------------------------------------------

    def analyze_video(
        self,
        path: Path,
        k: int,
        max_frames: int,
        top_k: int,
        explain: bool = True,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        """Score evenly sampled frames with the image pipeline and aggregate them.

        Frames without a face are kept in the timeline but excluded from the verdict. Only
        the most and the least suspicious frames get heatmaps (Grad-CAM is the slow step).
        """
        from app import video as vid
        from app.media import MediaError

        s, cal = self.settings, self.cal
        report = progress or (lambda _st, _d, _t, _u: None)
        timings = {"decode": 0.0, "detect": 0.0, "beautify": 0.0, "score": 0.0, "explain": 0.0}

        t0 = time.perf_counter()
        report("detecting_faces", 0, 1, "videos")
        info = vid.probe(path, s.max_video_decode_frames)
        indices = vid.sample_indices(info.frame_count, max_frames)
        timings["decode"] += time.perf_counter() - t0
        report("extracting_frames", 0, len(indices), "frames")

        digest = self._digest(path)
        frames: dict[int, dict[str, Any]] = {}
        images: dict[int, NDArray[np.uint8]] = {}
        best: tuple[float, int] | None = None
        worst: tuple[float, int] | None = None
        t_decode = time.perf_counter()
        for n, (index, rgb) in enumerate(vid.read_frames(path, indices, s.max_image_side)):
            timings["decode"] += time.perf_counter() - t_decode
            report("scoring", n, len(indices), "frames")
            frame_digest = hashlib.sha256(digest + index.to_bytes(8, "big")).digest()
            r = self._analyze_rgb(rgb, 1.0, frame_digest, k, False)
            for name in ("detect", "beautify", "score"):
                timings[name] += r["timingsMs"][name] / 1000.0
            entry: dict[str, Any] = {
                "index": index,
                "timestamp": vid.timestamp(index, info.fps),
                "decoded": True,
                "hasFace": bool(r["faces"]),
                "faceCount": len(r["faces"]),
                "logit": None,
                "sbi": None,
                "beautifiedLikelihood": None,
                "probability": r["probability"],
                "verdict": r["verdict"],
                "uncertainReason": r["uncertainReason"],
                "box": None,
                "explained": False,
            }
            if r["faces"]:
                top_face = r["faces"][0]
                logit = float(top_face["scores"]["gendHead"])
                entry.update(
                    logit=logit,
                    sbi=top_face["scores"]["sbi"],
                    beautifiedLikelihood=top_face["beautifiedLikelihood"],
                    box=top_face["box"],
                )
                if best is None or logit > best[0]:
                    best = (logit, index)
                if worst is None or logit < worst[0]:
                    worst = (logit, index)
                # Keep only the (at most two) frames that may be explained in memory.
                images[index] = rgb
                keep = {best[1], worst[1]}
                for stale in [i for i in images if i not in keep]:
                    del images[stale]
            frames[index] = entry
            t_decode = time.perf_counter()
        report("scoring", len(frames), len(indices), "frames")

        if not frames:
            raise MediaError("no frame of the video could be decoded")
        # Sampled frames that did not decode are shown as gaps, never scored.
        timeline = [
            frames.get(i)
            or {
                "index": i,
                "timestamp": vid.timestamp(i, info.fps),
                "decoded": False,
                "hasFace": False,
                "faceCount": 0,
                "logit": None,
                "probability": None,
                "verdict": None,
                "explained": False,
            }
            for i in indices
        ]
        result = vid.aggregate_frames(timeline, cal, top_k)

        faces: list[dict[str, Any]] = []
        if explain and best is not None and worst is not None:
            t0 = time.perf_counter()
            targets = [best[1]] if best[1] == worst[1] else [best[1], worst[1]]
            report("aggregating", 0, len(targets), "frames")
            for n, index in enumerate(targets):
                frame_digest = hashlib.sha256(digest + index.to_bytes(8, "big")).digest()
                r = self._analyze_rgb(images[index], 1.0, frame_digest, k, True)
                frames[index]["explained"] = True
                frames[index]["faces"] = r["faces"]
                if index == best[1]:
                    faces = r["faces"]
                report("aggregating", n + 1, len(targets), "frames")
            timings["explain"] = time.perf_counter() - t0

        return {
            **result,
            "kind": "video",
            "video": {**info.to_dict(), "sampledFrames": len(indices)},
            "frames": timeline,
            "faces": faces,
            "modelCardVersion": cal.version,
            "timingsMs": {name: round(v * 1000) for name, v in timings.items()},
        }

    # ---- /beautify and /compare ----------------------------------------------------------

    def _filter(
        self, family: str, strength: float, crop: Any, geom: FaceGeometry, rng: Any
    ) -> Variant:
        from app import beautify
        from app.beautify import f4_restore

        if family == "F4":
            try:
                image = f4_restore.apply(crop, strength, self.settings.weights_dir)
            except f4_restore.F4Unavailable as exc:
                raise PipelineError(str(exc), status=501) from exc
            return beautify.Variant("F4", image, (("F4", strength),))
        if family not in beautify.TRAIN_FAMILIES:
            raise PipelineError(f"unknown filter family: {family}")
        return beautify.apply_chain(family, ((family, strength),), crop, geom, rng)

    def _largest_face(self, path: Path) -> tuple[Any, DetectedFace, float]:
        image, scale = self._load(path)
        detected = self.detector.detect(image)
        if not detected:
            raise PipelineError("no face detected")
        return image, detected[0], scale

    def beautify(self, path: Path, family: str, strength: float) -> bytes:
        """Filter the largest face's aligned crop and return it as PNG bytes."""
        import io

        from PIL import Image

        image, face, _ = self._largest_face(path)
        crop, _ = self._crops(image, face)
        geom = self.landmarker.geometry(crop)
        if geom is None:
            raise PipelineError("no face landmarks found in the face crop")
        variant = self._filter(family, strength, crop, geom, self._rng(path, 0))
        buffer = io.BytesIO()
        Image.fromarray(variant.image).save(buffer, format="PNG")
        return buffer.getvalue()

    def compare(self, path: Path, family: str, strength: float) -> dict[str, Any]:
        """Score the largest face before and after one filter."""
        from app import explain as ex

        cal = self.cal
        image, face, scale = self._largest_face(path)
        crop, crop_sbi = self._crops(image, face)
        geom = self.landmarker.geometry(crop)
        if geom is None:
            raise PipelineError("no face landmarks found in the face crop")
        variant = self._filter(family, strength, crop, geom, self._rng(path, 0))

        # SBI sees a different crop, so the same filter is applied to it with its own landmarks.
        sbi_clean = sbi_filtered = None
        if self.sbi is not None:
            sbi_clean = self._sbi_logit(crop_sbi)
            geom_sbi = self.landmarker.geometry(crop_sbi)
            if geom_sbi is not None:
                filtered_sbi = self._filter(
                    family, strength, crop_sbi, geom_sbi, self._rng(path, 0)
                )
                sbi_filtered = self._sbi_logit(filtered_sbi.image)

        logits = self._head_logits([crop, variant.image])
        likelihoods = [beautified_likelihood(c, geom) for c in (crop, variant.image)]
        maps = ex.gradcam_gend(self.gend, [crop, variant.image], self.head)
        iou = ex.top_region_iou(
            maps[0], self._aligned_heatmap(maps[1], variant), self.settings.heatmap_top_fraction
        )

        def side(i: int, sbi_logit: float | None, img: Any) -> dict[str, Any]:
            likelihood = likelihoods[i]
            d = decide(cal, {"clean": logits[i]}, sbi_logit, likelihood or 0.0)
            return {
                "verdict": d.verdict,
                "probability": round(d.probability, 4),
                "confident": d.confident,
                "uncertainReason": d.uncertain_reason,
                "scores": {
                    "gendHead": round(logits[i], 4),
                    "sbi": None if sbi_logit is None else round(sbi_logit, 4),
                },
                "beautifiedLikelihood": None if likelihood is None else round(likelihood, 4),
                "image": self._save(img),
                "heatmap": self._save(ex.colourise(maps[i])),
            }

        clean, filtered = side(0, sbi_clean, crop), side(1, sbi_filtered, variant.image)
        return {
            "family": family,
            "strength": strength,
            "box": face.xywh(scale),
            "threshold": cal.threshold,
            "clean": clean,
            "filtered": filtered,
            "flipped": clean["verdict"] != filtered["verdict"],
            "heatmapIoU": round(iou, 4),
            "modelCardVersion": cal.version,
        }

    def close(self) -> None:
        self.landmarker.close()
