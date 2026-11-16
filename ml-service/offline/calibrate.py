"""Fit the service's calibration on the validation split only and write model_card.json.

    python -m offline.calibrate --cache artifacts/cache --head weights/head.pt
        --head-report artifacts/heads/final.json --out weights/model_card.json

Procedure (approved; fixed before seeing any data):

| Value             | Data  | Rule                                                              |
|-------------------|-------|-------------------------------------------------------------------|
| fusion weights    | val-A | logistic regression on [M1, SBI, spread, likelihood], C = 1,      |
|                   |       | class-balanced                                                    |
| temperature T     | val-B | minimise log-loss of sigmoid(fused / T), T in [0.05, 20]          |
| threshold         | val-B | maximise balanced accuracy, within [0.05, 0.95]                   |
| spread_limit      | val-B | 95th percentile of spread                                         |
| uncertain_margin  | val-B | largest m in 0.00..0.30 (step 0.01) keeping UNCERTAIN <= 20%      |
| confident_margin, |  —    | defaults, recorded as "default, not fitted"                      |
| detector_disagree |       |                                                                   |

Inputs mirror the service: M1 and spread come from clean + the 4 inference-style copies,
and the UNCERTAIN rate comes from the service's own `app.pipeline.decide`. The test split
cannot be opened. If validation data is too small, the card says so in
`calibration_status` and `warnings`, and the service refuses it unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from offline import common
from offline.cache_features import load_meta, load_split
from offline.copies import INFERENCE_SLOTS
from offline.heads import normalize

STAGE = "calibrate"
T_GRID = np.exp(np.linspace(np.log(0.05), np.log(20.0), 2001))
THRESHOLD_RANGE = (0.05, 0.95)
SPREAD_PERCENTILE = 95.0
MARGIN_GRID = np.round(np.arange(0.0, 0.3001, 0.01), 2)
MAX_UNCERTAIN_RATE = 0.20
FUSION_C = 1.0
# Minimum validation size per half for the card to be marked "ok".
MIN_VIDEOS_PER_CLASS = 20
MIN_FACES = 100

Array = NDArray[np.float64]


@dataclass
class FaceInputs:
    """What the service computes for each face, rebuilt from the cache."""

    video_id: NDArray[np.str_]
    subset: NDArray[np.str_]
    y: NDArray[np.int8]
    variants: list[dict[str, float]]  # {"clean", "F1", "F2", "F3", "mix"} -> head logit
    m1: Array
    spread: Array
    sbi: list[float | None]
    bl: Array

    def take(self, mask: NDArray[np.bool_]) -> FaceInputs:
        idx = np.flatnonzero(mask)
        return FaceInputs(
            self.video_id[idx],
            self.subset[idx],
            self.y[idx],
            [self.variants[i] for i in idx],
            self.m1[idx],
            self.spread[idx],
            [self.sbi[i] for i in idx],
            self.bl[idx],
        )

    def features(self) -> Array:
        sbi = np.array([0.0 if s is None else s for s in self.sbi])
        return np.column_stack([self.m1, sbi, self.spread, self.bl]).astype(np.float64)


def face_inputs(split: dict[str, Any], w: NDArray[np.float32], b: float) -> FaceInputs:
    names = split["copy_names"]
    logits = normalize(split["feats"].astype(np.float32)) @ w + b  # (N, C)
    variants = []
    for i, pick in enumerate(split["inference_pick"]):
        slots = {"clean": "clean", **{f: pick[f] for f in ("F1", "F2", "F3")}, "mix": "mix"}
        variants.append({slot: float(logits[i, names.index(c)]) for slot, c in slots.items()})
    m1 = np.array([np.mean(list(v.values())) for v in variants])
    spread = np.array([max(v.values()) - min(v.values()) for v in variants])
    sbi = [None if not np.isfinite(s) else float(s) for s in split["sbi"]]
    bl = np.nan_to_num(split["bl"].astype(np.float64), nan=0.0)  # service: `likelihood or 0.0`
    assert tuple(variants[0]) == ("clean", *INFERENCE_SLOTS)
    return FaceInputs(split["video_id"], split["subset"], split["y"], variants, m1, spread, sbi, bl)


# ---- the individual fits (pure; also used by eval_lofo for the LOFO heads) -------------


def sigmoid(x: Array) -> Array:
    out: Array = 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))
    return out


def log_loss(z: Array, y: NDArray[np.integer[Any]], temperature: float) -> float:
    p = np.clip(sigmoid(z / temperature), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(z: Array, y: NDArray[np.integer[Any]]) -> float:
    losses = [log_loss(z, y, t) for t in T_GRID]
    return float(T_GRID[int(np.argmin(losses))])


def balanced_accuracy(p: Array, y: NDArray[np.integer[Any]], threshold: float) -> float:
    pred = p >= threshold
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    tnr = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return 0.5 * (tpr + tnr)


def fit_threshold(p: Array, y: NDArray[np.integer[Any]]) -> float:
    """Maximise balanced accuracy; ties go to the candidate closest to 0.5."""
    lo, hi = THRESHOLD_RANGE
    u = np.unique(np.clip(p, lo, hi))
    candidates = np.unique(np.concatenate([[lo, hi, 0.5], (u[:-1] + u[1:]) / 2.0]))
    candidates = candidates[(candidates >= lo) & (candidates <= hi)]
    scores = np.array([balanced_accuracy(p, y, t) for t in candidates])
    best = scores.max()
    tied = candidates[np.isclose(scores, best)]
    return float(tied[np.argmin(np.abs(tied - 0.5))])


def fit_spread_limit(spread: Array) -> float:
    return float(np.percentile(spread, SPREAD_PERCENTILE))


def fit_fusion(x: Array, y: NDArray[np.integer[Any]]) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=FUSION_C, class_weight="balanced", max_iter=5000)
    model.fit(x, y)
    head, sbi, spread, bl = (float(c) for c in model.coef_[0])
    return {
        "head": head,
        "sbi": sbi,
        "spread": spread,
        "beautified_likelihood": bl,
        "bias": float(model.intercept_[0]),
    }


def uncertain_rate(cal: Any, faces: FaceInputs) -> float:
    from app.pipeline import decide

    verdicts = [
        decide(cal, faces.variants[i], faces.sbi[i], float(faces.bl[i])).verdict
        for i in range(len(faces.y))
    ]
    return float(np.mean([v == "UNCERTAIN" for v in verdicts])) if verdicts else float("nan")


def fit_margin(cal: Any, faces: FaceInputs) -> tuple[float, float, bool]:
    """Largest margin with UNCERTAIN rate <= the cap. Returns (margin, rate, cap_met)."""
    chosen, chosen_rate = 0.0, uncertain_rate(replace(cal, uncertain_margin=0.0), faces)
    if chosen_rate > MAX_UNCERTAIN_RATE:
        return 0.0, chosen_rate, False
    for m in MARGIN_GRID[1:]:
        rate = uncertain_rate(replace(cal, uncertain_margin=float(m)), faces)
        if rate > MAX_UNCERTAIN_RATE:
            break
        chosen, chosen_rate = float(m), rate
    return chosen, chosen_rate, True


def fused_logits(fusion: dict[str, float], faces: FaceInputs) -> Array:
    """Vectorised app.head.fuse."""
    x = faces.features()
    w = np.array([fusion["head"], fusion["sbi"], fusion["spread"], fusion["beautified_likelihood"]])
    out: Array = x @ w + fusion["bias"]
    return out


def sufficiency(faces: FaceInputs, name: str) -> list[str]:
    problems = []
    for cls, label in ((0, "real"), (1, "fake")):
        videos = len(set(faces.video_id[faces.y == cls]))
        if videos < MIN_VIDEOS_PER_CLASS:
            problems.append(f"{name}: {videos} {label} videos (< {MIN_VIDEOS_PER_CLASS})")
    if len(faces.y) < MIN_FACES:
        problems.append(f"{name}: {len(faces.y)} faces (< {MIN_FACES})")
    return problems


def bootstrap_ci(faces: FaceInputs, fn: Any, n: int, seed: int) -> list[float] | None:
    from offline.metrics import cluster_bootstrap

    def stat(idx: NDArray[np.int64]) -> float:
        mask = np.zeros(len(faces.y), bool)
        mask[idx] = True
        sub = faces.take(mask)
        if len(np.unique(sub.y)) < 2:
            return float("nan")
        return float(fn(sub))

    if n <= 0:
        return None
    lo, hi = cluster_bootstrap(faces.video_id, stat, n_resamples=n, seed=seed)
    return None if not np.isfinite(lo) else [round(lo, 6), round(hi, 6)]


# ---- the full procedure -----------------------------------------------------------------


def calibrate(
    faces: FaceInputs, n_boot: int = 200, seed: int = 0
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Return (values, sources, warnings)."""
    from app.head import Calibration, FusionWeights

    defaults = Calibration()
    val_a = faces.take(faces.subset == "val-A")
    val_b = faces.take(faces.subset == "val-B")
    warnings = sufficiency(val_a, "val-A") + sufficiency(val_b, "val-B")
    sources: dict[str, Any] = {}

    def two_classes(f: FaceInputs) -> bool:
        return len(np.unique(f.y)) == 2

    # 1. Fusion weights on val-A.
    fusion = {k: float(v) for k, v in vars(defaults.fusion).items()}
    if two_classes(val_a):
        fusion = fit_fusion(val_a.features(), val_a.y)
        sources["fusion"] = {
            "method": f"logistic regression on [M1, SBI, spread, beautifiedLikelihood], "
            f"C={FUSION_C}, class-balanced",
            "data": "val-A",
            "faces": len(val_a.y),
            "videos": len(set(val_a.video_id)),
        }
        if all(s is None for s in val_a.sbi):
            warnings.append(
                "SBI logits are missing for val-A: the SBI fusion weight is fitted "
                "on zeros and has no effect"
            )
    else:
        warnings.append("val-A lacks both classes: fusion weights NOT fitted (identity fusion)")
        sources["fusion"] = {"method": "default identity fusion, NOT fitted", "data": None}

    # 2. Temperature on val-B.
    z_b = fused_logits(fusion, val_b)
    if two_classes(val_b):
        temperature = fit_temperature(z_b, val_b.y)
        sources["temperature"] = {
            "method": "minimise log-loss over 2001 log-spaced T in [0.05, 20]",
            "data": "val-B",
            "faces": len(val_b.y),
            "ci95": bootstrap_ci(
                val_b, lambda f: fit_temperature(fused_logits(fusion, f), f.y), n_boot, seed
            ),
        }
    else:
        temperature = defaults.temperature
        warnings.append("val-B lacks both classes: temperature, threshold NOT fitted")
        sources["temperature"] = {"method": "default, NOT fitted", "data": None}

    # 3. Threshold on val-B.
    p_b = sigmoid(z_b / temperature)
    if two_classes(val_b):
        threshold = fit_threshold(p_b, val_b.y)
        sources["threshold"] = {
            "method": "maximise balanced accuracy within [0.05, 0.95]; ties -> closest to 0.5",
            "data": "val-B",
            "faces": len(val_b.y),
            "balanced_accuracy": round(balanced_accuracy(p_b, val_b.y, threshold), 6),
            "ci95": bootstrap_ci(
                val_b,
                lambda f: fit_threshold(sigmoid(fused_logits(fusion, f) / temperature), f.y),
                n_boot,
                seed,
            ),
            "note": "fitted on clean inputs only (copies of beautified inputs are not cached)",
        }
    else:
        threshold = defaults.threshold
        sources["threshold"] = {"method": "default, NOT fitted", "data": None}

    # 4. Spread limit on val-B.
    if len(val_b.y) >= 2:
        spread_limit = fit_spread_limit(val_b.spread)
        sources["spread_limit"] = {
            "method": f"{SPREAD_PERCENTILE:g}th percentile of val-B spread",
            "data": "val-B",
            "faces": len(val_b.y),
            "ci95": bootstrap_ci(val_b, lambda f: fit_spread_limit(f.spread), n_boot, seed),
        }
    else:
        spread_limit = defaults.spread_limit
        warnings.append("val-B has fewer than 2 faces: spread limit NOT fitted")
        sources["spread_limit"] = {"method": "default, NOT fitted", "data": None}

    # 5. Uncertain margin on val-B, through the service's own decision rule.
    cal = replace(
        defaults,
        temperature=temperature,
        threshold=threshold,
        spread_limit=spread_limit,
        fusion=FusionWeights(**fusion),
    )
    if len(val_b.y):
        margin, rate, cap_met = fit_margin(cal, val_b)
        sources["uncertain_margin"] = {
            "method": f"largest m in 0.00..0.30 (step 0.01) with UNCERTAIN rate <= "
            f"{MAX_UNCERTAIN_RATE:.0%} on val-B (all four UNCERTAIN rules, app.pipeline.decide)",
            "data": "val-B",
            "faces": len(val_b.y),
            "uncertain_rate": round(rate, 6),
            "cap_met": cap_met,
        }
        if not cap_met:
            warnings.append(
                f"UNCERTAIN rate is {rate:.1%} on val-B even with margin 0 "
                f"(cap {MAX_UNCERTAIN_RATE:.0%}): the other rules exceed the cap"
            )
    else:
        margin = defaults.uncertain_margin
        sources["uncertain_margin"] = {"method": "default, NOT fitted", "data": None}

    for key in ("confident_margin", "detector_disagree_limit"):
        sources[key] = {"method": "default, not fitted (not in the spec's calibration list)"}
    values = {
        "temperature": temperature,
        "threshold": threshold,
        "uncertain_margin": margin,
        "confident_margin": defaults.confident_margin,
        "spread_limit": spread_limit,
        "detector_disagree_limit": defaults.detector_disagree_limit,
        "fusion": fusion,
    }
    return values, sources, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--head", type=Path, default=Path("weights/head.pt"))
    parser.add_argument("--head-report", type=Path, default=Path("artifacts/heads/final.json"))
    parser.add_argument("--out", type=Path, default=Path("weights/model_card.json"))
    parser.add_argument("--bootstrap", type=int, default=200, help="resamples per CI (0: none)")
    args = parser.parse_args(argv)
    log = common.setup_logging()

    import torch

    head = torch.load(args.head, map_location="cpu", weights_only=True)
    w = head["weight"].numpy().astype(np.float32).reshape(-1)
    b = float(head["bias"])
    head_sha = common.sha256_file(args.head)
    report = json.loads(args.head_report.read_text()) if args.head_report.is_file() else {}
    if report and report.get("sha256") != head_sha:
        log.error("%s does not describe %s (SHA-256 differs)", args.head_report, args.head)
        return 2

    meta = load_meta(args.cache)
    faces = face_inputs(load_split(args.cache, "val", STAGE), w, b)
    values, sources, warnings = calibrate(faces, n_boot=args.bootstrap)
    fixture = bool(meta["fixture"])
    status = "ok" if not warnings else "insufficient_validation_data"
    if fixture:
        warnings.insert(0, common.FIXTURE_LABEL)
    created = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    card = {
        **values,
        "version": f"{created[:10]}-{head_sha[:8]}-{'fixture' if fixture else status}",
        "calibration_status": status,
        "fixture": fixture,
        "warnings": warnings,
        "head_sha256": head_sha,
        "sources": sources,
        "protocol": meta["protocol"],
        "head": {
            "file": args.head.name,
            "sha256": head_sha,
            **{
                k: report[k]
                for k in ("scheme", "families", "seed", "lambda_p", "lambda_rule")
                if k in report
            },
            **{k: head.get("meta", {}).get(k) for k in ("init",)},
        },
        "data": {
            "manifest_sha256": meta["manifest_sha256"],
            "cache_meta_sha256": common.sha256_file(args.cache / "meta.json"),
            "val_faces": {s: int((faces.subset == s).sum()) for s in ("val-A", "val-B")},
            "test_split_used": False,
        },
        "limits": {
            "min_videos_per_class_per_half": MIN_VIDEOS_PER_CLASS,
            "min_faces_per_half": MIN_FACES,
        },
        "code_commit": common.git_commit(Path(__file__).resolve().parent),
        "created": created,
    }
    common.atomic_write_json(args.out, card)
    log.info(
        "model card written: status=%s, T=%.3f, threshold=%.3f, margin=%.2f, spread=%.3f",
        status,
        values["temperature"],
        values["threshold"],
        values["uncertain_margin"],
        values["spread_limit"],
    )
    for warning in warnings:
        log.warning("%s", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
