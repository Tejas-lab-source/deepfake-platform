"""Leave-one-family-out evaluation on the test split, the inference-time ablation, and the
pre-registered success criteria.

    python -m offline.eval_lofo --cache artifacts/cache --heads-dir artifacts/heads
        --final-head weights/head.pt --model-card weights/model_card.json
        --faces artifacts/faces --out artifacts/eval/lofo.json
        [--iou-faces 100 | --iou-faces -1 (all) | --iou-faces 0 (skip)] [--bootstrap 1000]

For every scheme x fold x seed, each head gets its own temperature and threshold, fitted on
val-B with clean + the fold's two training families (never the held-out family), plus the
spread limit and uncertain margin by the calibration rules. The test split is used only
here, only for reporting. Results are JSON; the run also prints the tables.

Metrics, per head (forced decisions: p >= threshold means FAKE):
  clean_auc            AUC of the clean logit
  fpr_beautified_real  real faces whose held-out-family copies (lo and hi) read FAKE
  fnr_beautified_fake  fake faces whose held-out-family copies read REAL
  flip_rate            held-out copies whose decision differs from the clean decision
  uncertain_rate       the full UNCERTAIN rule on clean + the 4 inference-style copies
  clean_error_forced   error on clean inputs with UNCERTAIN forced to a decision
  f4_*                 the same on F4 copies, when GFPGAN was available
  heatmap_iou          seed 0 only, a sample of test faces (see --iou-faces)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from offline import calibrate as cb
from offline import common, metrics
from offline.cache_features import load_meta, load_split
from offline.heads import normalize
from offline.schemes import FAMILY_COPIES, FOLDS, SCHEMES, SEEDS, is_held_out, training_families

STAGE = "eval_lofo"
METRICS = (
    "clean_auc",
    "fpr_beautified_real",
    "fnr_beautified_fake",
    "flip_rate",
    "uncertain_rate",
    "clean_error_forced",
    "f4_fpr_real",
    "f4_fnr_fake",
    "f4_flip_rate",
)
SUCCESS_FLIP_REDUCTION = 0.30
SUCCESS_AUC_POINTS = 0.01


def load_head(path: Path) -> tuple[NDArray[np.float32], float, str]:
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    return (
        data["weight"].numpy().astype(np.float32).reshape(-1),
        float(data["bias"]),
        common.sha256_file(path),
    )


class Split:
    """One cache split with normalised features, ready for any head."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.names: list[str] = data["copy_names"]
        self.z = normalize(data["feats"].astype(np.float32))
        self.present: NDArray[np.bool_] = data["present"]
        self.y: NDArray[np.int8] = data["y"]
        self.video_id: NDArray[np.str_] = data["video_id"]
        self.subset: NDArray[np.str_] = data["subset"]

    def logits(self, w: NDArray[np.float32], b: float) -> NDArray[np.float32]:
        out: NDArray[np.float32] = self.z @ w + b
        return out

    def col(self, name: str) -> int:
        return self.names.index(name)


def head_calibration(
    val: Split, w: NDArray[np.float32], b: float, held_out: str
) -> tuple[Any, dict[str, Any]]:
    """T and threshold on val-B (clean + training-family copies as single inputs), then
    spread limit and margin by the calibration rules. Identity fusion: LOFO compares heads."""
    from app.head import Calibration

    rows = val.subset == "val-B"
    logits = val.logits(w, b)
    cols = [val.col("clean")] + [
        val.col(c) for f in training_families(held_out) for c in FAMILY_COPIES[f]
    ]
    z = np.concatenate([logits[rows, c][val.present[rows, c]] for c in cols]).astype(np.float64)
    y = np.concatenate([val.y[rows][val.present[rows, c]] for c in cols])
    info: dict[str, Any] = {"val_b_inputs": int(len(y))}
    if len(np.unique(y)) < 2:
        info["warning"] = "val-B lacks both classes: default T = 1, threshold = 0.5"
        temperature, threshold = 1.0, 0.5
    else:
        temperature = cb.fit_temperature(z, y)
        threshold = cb.fit_threshold(cb.sigmoid(z / temperature), y)
    faces_b = cb.face_inputs(val.data, w, b).take(rows)
    spread_limit = cb.fit_spread_limit(faces_b.spread) if len(faces_b.y) >= 2 else 1.5
    cal = Calibration(temperature=temperature, threshold=threshold, spread_limit=spread_limit)
    margin, rate, cap_met = cb.fit_margin(cal, faces_b)
    cal = replace(cal, uncertain_margin=margin)
    info.update(
        temperature=temperature,
        threshold=threshold,
        spread_limit=spread_limit,
        uncertain_margin=margin,
        val_b_uncertain_rate=rate,
        uncertain_cap_met=cap_met,
    )
    return cal, info


def face_level(
    test: Split, w: NDArray[np.float32], b: float, cal: Any, held_out: str
) -> dict[str, Any]:
    """Per-face arrays for one head; metrics and bootstraps are computed from these."""
    from app.pipeline import decide

    logits = test.logits(w, b)
    p = cb.sigmoid(logits.astype(np.float64) / cal.temperature)
    clean = test.col("clean")
    held = [test.col(c) for c in FAMILY_COPIES[held_out]]
    f4 = test.col("F4") if "F4" in test.names else None
    inputs = cb.face_inputs(test.data, w, b)
    uncertain = np.array(
        [
            decide(cal, inputs.variants[i], inputs.sbi[i], float(inputs.bl[i])).verdict
            == "UNCERTAIN"
            for i in range(len(test.y))
        ]
    )
    return {
        "logit_clean": logits[:, clean],
        "p_clean": p[:, clean],
        "p_held": p[:, held],
        "held_present": test.present[:, held],
        "p_f4": None if f4 is None else p[:, f4],
        "f4_present": None if f4 is None else test.present[:, f4],
        "uncertain": uncertain,
    }


def metrics_for(
    fl: dict[str, Any], y: NDArray[np.int8], thr: float, idx: Any = None
) -> dict[str, float]:
    sel = np.arange(len(y)) if idx is None else np.asarray(idx)
    y = y[sel]
    pc, ph, hp = fl["p_clean"][sel], fl["p_held"][sel], fl["held_present"][sel]
    rows, cols = np.nonzero(hp)
    y_pairs, ph_pairs, pc_pairs = y[rows], ph[rows, cols], pc[rows]
    out = {
        "clean_auc": metrics.auc(y, fl["logit_clean"][sel]),
        "fpr_beautified_real": metrics.false_positive_rate(y_pairs, ph_pairs, thr),
        "fnr_beautified_fake": metrics.false_negative_rate(y_pairs, ph_pairs, thr),
        "flip_rate": metrics.flip_rate(pc_pairs, ph_pairs, thr),
        "uncertain_rate": metrics.rate(fl["uncertain"][sel]),
        "clean_error_forced": metrics.rate((pc >= thr) != (y == 1)),
        "f4_fpr_real": float("nan"),
        "f4_fnr_fake": float("nan"),
        "f4_flip_rate": float("nan"),
    }
    if fl["p_f4"] is not None:
        ok = fl["f4_present"][sel]
        if ok.any():
            out["f4_fpr_real"] = metrics.false_positive_rate(y[ok], fl["p_f4"][sel][ok], thr)
            out["f4_fnr_fake"] = metrics.false_negative_rate(y[ok], fl["p_f4"][sel][ok], thr)
            out["f4_flip_rate"] = metrics.flip_rate(pc[ok], fl["p_f4"][sel][ok], thr)
    return out


def _clean(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 6)


def heatmap_ious(
    scheme_folds: list[tuple[str, str, NDArray[np.float32], float]],
    test: Split,
    faces_dir: Path,
    weights_dir: Path,
    n_faces: int,
    top: float,
    log: Any,
) -> dict[str, dict[str, Any]]:
    """IoU of the top regions of the clean and the held-out (hi) copy's Grad-CAM, through
    each head (seed 0), with F3 maps warped back. Returns {"scheme__fold": {...}}."""
    import torch

    from app.detectors.gend import GenDDetector
    from app.explain import gradcam_gend, top_region_iou
    from app.faces import Landmarker
    from app.head import LinearHead
    from offline.copies import plan_copy, render_copy

    manifest = json.loads((faces_dir / "manifest.json").read_text())
    crop_of = {f["face_id"]: f["gend_crop"] for f in manifest["faces"]}
    face_ids = sorted(test.data["face_id"], key=lambda f: common.seed_for("iou", f))
    face_ids = face_ids if n_faces < 0 else face_ids[:n_faces]
    gend = GenDDetector(weights_dir, torch.device("cpu"))
    lm = Landmarker(weights_dir / "mediapipe" / "face_landmarker.task")
    prepared: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
    try:
        for fid in face_ids:
            crop = common.read_png(faces_dir / crop_of[fid])
            geom = lm.geometry(crop)
            if geom is None:
                continue
            for fold in {f for _, f, _, _ in scheme_folds}:
                plan = plan_copy(fid, f"{fold}_hi")
                img, transform = render_copy(plan, crop, geom, weights_dir)
                prepared[(fid, fold)] = (crop, img, transform)
    finally:
        lm.close()
    out: dict[str, dict[str, Any]] = {}
    for scheme, fold, w, b in scheme_folds:
        head = LinearHead(w, b, True, "eval")
        values = []
        for fid in face_ids:
            if (fid, fold) not in prepared:
                continue
            crop, img, transform = prepared[(fid, fold)]
            maps = gradcam_gend(gend, [crop, img], head)
            other = maps[1] if transform is None else transform.backward(maps[1]).astype(np.float32)
            values.append(top_region_iou(maps[0], other, top))
        out[f"{scheme}__{fold}"] = {
            "mean": _clean(float(np.mean(values))) if values else None,
            "faces": len(values),
            "copy": f"{fold}_hi",
        }
        log.info(
            "heatmap IoU %s %s: %s over %d faces",
            scheme,
            fold,
            out[f"{scheme}__{fold}"]["mean"],
            len(values),
        )
    return out


def ablation(test: Split, w: NDArray[np.float32], b: float, card_cal: Any) -> list[dict[str, Any]]:
    """Inference-time ablation of the shipped head on clean test inputs."""
    from app.head import Calibration, FusionWeights
    from app.pipeline import aggregate, decide

    inputs = cb.face_inputs(test.data, w, b)
    y = test.y
    steps: list[tuple[str, Any, str]] = [
        ("base: clean logit, T=1, threshold 0.5", None, "clean"),
        ("+M1: mean over clean + 4 copies", None, "m1"),
        ("+M2: spread rule (card spread limit)", None, "m2"),
        (
            "+calibration: card T, threshold, margin; all UNCERTAIN rules; identity fusion",
            replace(card_cal, fusion=FusionWeights()),
            "decide",
        ),
        ("+fusion: full model card", card_cal, "decide"),
    ]
    rows = []
    for label, cal, mode in steps:
        p = np.zeros(len(y))
        unc = np.zeros(len(y), bool)
        thr = 0.5
        for i in range(len(y)):
            v = inputs.variants[i]
            if mode == "clean":
                p[i] = float(cb.sigmoid(np.array([v["clean"]]))[0])
            elif mode in ("m1", "m2"):
                m1, spread = aggregate(v)
                p[i] = float(cb.sigmoid(np.array([m1]))[0])
                unc[i] = mode == "m2" and spread > card_cal.spread_limit
            else:
                d = decide(cal, v, inputs.sbi[i], float(inputs.bl[i]))
                p[i], unc[i], thr = d.probability, d.verdict == "UNCERTAIN", cal.threshold
        wrong = (p >= thr) != (y == 1)
        rows.append(
            {
                "step": label,
                "error_forced": _clean(metrics.rate(wrong)),
                "uncertain_rate": _clean(metrics.rate(unc)),
                "error_on_decided": _clean(metrics.rate(wrong[~unc])) if (~unc).any() else None,
                "auc": _clean(metrics.auc(y, p)),
            }
        )
    _ = Calibration  # imported for type clarity in readers
    return rows


def criteria(aggregate: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    def get(scheme: str, metric: str) -> float:
        value = aggregate.get(scheme, {}).get(metric, {}).get("mean")
        return float("nan") if value is None else float(value)

    def verdict(ok: bool | None) -> str:
        return "NOT EVALUABLE" if ok is None else ("MET" if ok else "NOT MET")

    flip_f, flip_c = get("H-f", "flip_rate"), get("H-c", "flip_rate")
    fpr_f, fpr_a, fpr_c = (get(s, "fpr_beautified_real") for s in ("H-f", "H-a", "H-c"))
    auc_f, auc_a = get("H-f", "clean_auc"), get("H-a", "clean_auc")
    c1 = (
        None
        if not np.isfinite([flip_f, flip_c]).all()
        else bool(flip_f <= (1 - SUCCESS_FLIP_REDUCTION) * flip_c)
    )
    c2 = (
        None
        if not np.isfinite([fpr_f, fpr_a, fpr_c]).all()
        else bool(fpr_f < fpr_a and fpr_f < fpr_c)
    )
    c3 = (
        None
        if not np.isfinite([auc_f, auc_a]).all()
        else bool(abs(auc_f - auc_a) <= SUCCESS_AUC_POINTS)
    )
    return [
        {
            "criterion": "flip rate on the held-out family at least 30% lower than H-c",
            "values": {"H-f": _clean(flip_f), "H-c": _clean(flip_c)},
            "result": verdict(c1),
        },
        {
            "criterion": "FPR on beautified real faces below both H-a and H-c",
            "values": {"H-f": _clean(fpr_f), "H-a": _clean(fpr_a), "H-c": _clean(fpr_c)},
            "result": verdict(c2),
        },
        {
            "criterion": "clean AUC within 1 point of H-a",
            "values": {"H-f": _clean(auc_f), "H-a": _clean(auc_a)},
            "result": verdict(c3),
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--heads-dir", type=Path, default=Path("artifacts/heads"))
    parser.add_argument("--final-head", type=Path, default=Path("weights/head.pt"))
    parser.add_argument("--model-card", type=Path, default=Path("weights/model_card.json"))
    parser.add_argument("--faces", type=Path, default=Path("artifacts/faces"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval/lofo.json"))
    parser.add_argument("--iou-faces", type=int, default=100, help="-1: all, 0: skip")
    parser.add_argument("--iou-seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    log = common.setup_logging()
    from app.head import load_calibration

    meta = load_meta(args.cache)
    cache_meta_sha = common.sha256_file(args.cache / "meta.json")
    index = json.loads((args.heads_dir / "index.json").read_text())
    for key, entry in index.items():  # every head must be the one that was trained
        if (
            "file" in entry
            and common.sha256_file(args.heads_dir / entry["file"]) != entry["sha256"]
        ):
            log.error("%s changed since training (SHA-256 mismatch)", key)
            return 2
        if entry.get("cache_meta_sha256") != cache_meta_sha:
            log.error("%s was trained on a different cache", key)
            return 2
    card = json.loads(args.model_card.read_text())
    w_f, b_f, sha_f = load_head(args.final_head)
    if card.get("head_sha256") != sha_f:
        log.error("the model card was not fitted for %s", args.final_head)
        return 2

    val = Split(load_split(args.cache, "val", STAGE))
    test = Split(load_split(args.cache, "test", STAGE))
    with np.load(args.cache / "gend_head.npz") as g:
        w0, b0 = g["weight"].astype(np.float32), float(g["bias"])

    runs: list[dict[str, Any]] = []
    per_fold: dict[str, dict[str, Any]] = {}
    iou_inputs: list[tuple[str, str, NDArray[np.float32], float]] = []
    for scheme in SCHEMES.values():
        for fold in FOLDS:
            fold_runs = []
            for seed in SEEDS:
                entry = index.get(f"{scheme.name}__{fold}__seed{seed}")
                if entry is None:
                    log.error(
                        "missing head %s %s seed %s: run train_head first", scheme.name, fold, seed
                    )
                    return 2
                if scheme.trains:
                    w, b, _ = load_head(args.heads_dir / entry["file"])
                else:
                    w, b = w0, b0
                cal, cal_info = head_calibration(val, w, b, fold)
                fl = face_level(test, w, b, cal, fold)
                m = metrics_for(fl, test.y, cal.threshold)
                fold_runs.append((fl, cal.threshold))
                runs.append(
                    {
                        "scheme": scheme.name,
                        "fold": fold,
                        "seed": seed,
                        "held_out": is_held_out(scheme, fold),
                        "calibration": {
                            k: (_clean(v) if isinstance(v, float) else v)
                            for k, v in cal_info.items()
                        },
                        "metrics": {k: _clean(v) for k, v in m.items()},
                    }
                )
                if seed == args.iou_seed:
                    iou_inputs.append((scheme.name, fold, w, b))
            # Seed-mean, seed std, and a video-level bootstrap CI of the seed-mean.
            summary: dict[str, Any] = {"held_out": is_held_out(scheme, fold)}
            for name in METRICS:
                vals = [r["metrics"][name] for r in runs[-len(SEEDS) :]]
                mean, std = metrics.mean_std([np.nan if v is None else v for v in vals])

                def stat(
                    idx: NDArray[np.int64], name: str = name, fold_runs: list[Any] = fold_runs
                ) -> float:
                    seed_vals = [metrics_for(fl_, test.y, t_, idx)[name] for fl_, t_ in fold_runs]
                    return (
                        float(np.nanmean(seed_vals))
                        if np.isfinite(seed_vals).any()
                        else float("nan")
                    )

                ci = (
                    metrics.cluster_bootstrap(test.video_id, stat, args.bootstrap, seed=0)
                    if args.bootstrap > 0 and np.isfinite(mean)
                    else (float("nan"), float("nan"))
                )
                summary[name] = {
                    "mean": _clean(mean),
                    "std_over_seeds": _clean(std),
                    "ci95": None if not np.isfinite(ci[0]) else [_clean(ci[0]), _clean(ci[1])],
                }
            per_fold[f"{scheme.name}__{fold}"] = summary
            log.info(
                "%s fold %s: flip %s, FPR-beautified-real %s, clean AUC %s%s",
                scheme.name,
                fold,
                summary["flip_rate"]["mean"],
                summary["fpr_beautified_real"]["mean"],
                summary["clean_auc"]["mean"],
                "" if summary["held_out"] else " (NOT HELD OUT: excluded from aggregates)",
            )

    aggregate: dict[str, dict[str, Any]] = {}
    for name_s in SCHEMES:
        folds = [
            per_fold[f"{name_s}__{f}"] for f in FOLDS if per_fold[f"{name_s}__{f}"]["held_out"]
        ]
        aggregate[name_s] = {
            "folds_included": [f for f in FOLDS if per_fold[f"{name_s}__{f}"]["held_out"]]
        }
        for name in METRICS:
            vals = [f[name]["mean"] for f in folds if f[name]["mean"] is not None]
            aggregate[name_s][name] = {"mean": _clean(float(np.mean(vals))) if vals else None}

    iou = {}
    if args.iou_faces != 0:
        iou = heatmap_ious(iou_inputs, test, args.faces, args.weights_dir, args.iou_faces, 0.2, log)
    card_cal = load_calibration(args.model_card)
    report = {
        "schema": 1,
        "protocol": meta["protocol"],
        "fixture": bool(meta["fixture"]),
        "notice": common.FIXTURE_LABEL if meta["fixture"] else None,
        "created": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "inputs": {
            "cache_meta_sha256": cache_meta_sha,
            "manifest_sha256": meta["manifest_sha256"],
            "model_card_sha256": common.sha256_file(args.model_card),
            "model_card_status": card.get("calibration_status"),
            "final_head_sha256": sha_f,
            "test_faces": int(len(test.y)),
            "test_videos": int(len(set(test.video_id))),
            "f4_copies_present": int(meta.get("f4_copies_present", 0)),
        },
        "definitions": __doc__,
        "runs": runs,
        "per_fold": per_fold,
        "aggregate_over_held_out_folds": aggregate,
        "heatmap_iou_seed0": iou,
        "ablation_shipped_head_clean_test_inputs": ablation(test, w_f, b_f, card_cal),
        "success_criteria": [
            (
                {**c, "fixture_only": "plumbing fixture: not an accuracy result"}
                if meta["fixture"]
                else c
            )
            for c in criteria(aggregate)
        ],
        "code_commit": common.git_commit(Path(__file__).resolve().parent),
    }
    common.atomic_write_json(args.out, report)

    print("\n" + report["protocol"])
    if report["fixture"]:
        print(common.FIXTURE_LABEL)
    print(
        f"\n{'scheme':<6}{'clean AUC':>11}{'FPR beaut. real':>17}{'FNR beaut. fake':>17}"
        f"{'flip':>8}{'UNCERTAIN':>11}"
    )
    for name_s, agg in aggregate.items():

        def fmt(name: str, agg: dict[str, Any] = agg) -> str:
            v = agg[name]["mean"]
            return "n/a" if v is None else f"{v:.3f}"

        print(
            f"{name_s:<6}{fmt('clean_auc'):>11}{fmt('fpr_beautified_real'):>17}"
            f"{fmt('fnr_beautified_fake'):>17}{fmt('flip_rate'):>8}{fmt('uncertain_rate'):>11}"
        )
    print("\nPre-registered success criteria:")
    for c in report["success_criteria"]:
        tag = "  (FIXTURE: meaningless)" if report["fixture"] else ""
        print(f"  [{c['result']}] {c['criterion']}  {c['values']}{tag}")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
