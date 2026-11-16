"""Train the linear heads on cached features: the six schemes x 3 LOFO folds x 3 seeds,
and the shipped head (H-f on all of F1-F3, λp chosen on val-A).

    python -m offline.train_head --cache artifacts/cache --out-dir artifacts/heads
        [--schemes H-a,...] [--folds F1,F2,F3] [--seeds 0,1,2] [--lambda-p 1.0]
        [--final --final-out weights/head.pt]

Uses the train split for fitting and only the val-A half of validation, for early stopping
and the λp choice. val-B is left for calibration; the test split cannot be opened.
Every head starts from GenD's own head (w = W[1] - W[0], b = b[1] - b[0]).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from offline import common, metrics
from offline.cache_features import load_meta, load_split
from offline.heads import class_weights, invariance_loss, normalize, save_head
from offline.schemes import (
    FAMILY_COPIES,
    FOLDS,
    SCHEMES,
    SEEDS,
    Scheme,
    candidate_copies,
    is_held_out,
    lambda_for,
    select_copies,
    training_families,
)

LAMBDA_GRID = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
STAGE = "train_head"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    patience: int = 15


@dataclass
class Data:
    names: list[str]
    z: NDArray[np.float32]  # (N, C, D) L2-normalised
    present: NDArray[np.bool_]
    y: NDArray[np.int8]


def _prepare(split: dict[str, Any], mask: NDArray[np.bool_] | None = None) -> Data:
    keep = np.ones(len(split["y"]), bool) if mask is None else mask
    return Data(
        names=list(split["copy_names"]),
        z=normalize(split["feats"][keep].astype(np.float32)),
        present=split["present"][keep],
        y=split["y"][keep],
    )


def load_training_data(
    cache: Path,
) -> tuple[Data, Data, NDArray[np.float32], float, dict[str, Any]]:
    """(train, val-A, GenD w, GenD b, meta). val-B is dropped here, never used in training."""
    meta = load_meta(cache)
    train = _prepare(load_split(cache, "train", STAGE))
    val = load_split(cache, "val", STAGE)
    val_a = _prepare(val, val["subset"] == "val-A")
    with np.load(cache / "gend_head.npz") as g:
        w0, b0 = g["weight"].astype(np.float32), float(g["bias"])
    return train, val_a, w0, b0, meta


def stopping_copies(held_out: str | None) -> tuple[str, ...]:
    """Copies scored for early stopping: clean + the fold's training families, the same for
    every scheme so the comparison is fair."""
    return ("clean", *(c for f in training_families(held_out) for c in FAMILY_COPIES[f]))


def _logits(z: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return z @ w + b


def stopping_loss(data: Data, held_out: str | None, w: torch.Tensor, b: torch.Tensor) -> float:
    cols = [data.names.index(c) for c in stopping_copies(held_out)]
    z = torch.from_numpy(data.z[:, cols])
    avail = torch.from_numpy(data.present[:, cols])
    y = torch.from_numpy(data.y.astype(np.float32))[:, None].expand(-1, len(cols))
    weight = torch.from_numpy(class_weights(data.y))[:, None].expand(-1, len(cols)) * avail
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        _logits(z, w, b), y, reduction="none"
    )
    return float((bce * weight).sum() / weight.sum())


def train_one(
    scheme: Scheme,
    held_out: str | None,
    seed: int,
    lambda_p: float,
    train: Data,
    val_a: Data,
    w0: NDArray[np.float32],
    b0: float,
    cfg: TrainConfig,
) -> tuple[NDArray[np.float32], float, dict[str, Any]]:
    if not scheme.trains:
        return w0.copy(), b0, {"epochs": 0, "note": "no training: GenD as released"}
    cands = candidate_copies(scheme, held_out)
    cols = [train.names.index(c) for c in cands]
    clean_col = train.names.index("clean")
    lam = lambda_for(scheme, lambda_p)
    torch.manual_seed(common.seed_for("torch", scheme.name, held_out, seed))
    rng = common.rng_for("train", scheme.name, held_out, seed)

    w = torch.tensor(w0, requires_grad=True)
    b = torch.tensor(b0, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=cfg.lr)
    z_clean = torch.from_numpy(train.z[:, clean_col])
    z_cand = torch.from_numpy(train.z[:, cols])  # (N, K, D)
    avail = train.present[:, cols]
    y = torch.from_numpy(train.y.astype(np.float32))
    sw = torch.from_numpy(class_weights(train.y))
    usable = np.flatnonzero(avail.any(axis=1))

    can_stop = len(val_a.y) > 0 and len(np.unique(val_a.y)) == 2
    best = (float("inf"), w0.copy(), b0, 0)
    stale = 0
    epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(usable)
        for batch in common.chunks(order.tolist(), cfg.batch_size):
            idx = torch.tensor(batch)
            with torch.no_grad():
                cand_logits = _logits(z_cand[idx], w, b).numpy()
                clean_logits = _logits(z_clean[idx], w, b).numpy()
            pick = select_copies(scheme.selection, clean_logits, cand_logits, avail[batch], rng)
            z_copy = z_cand[idx, torch.from_numpy(pick)]
            loss = invariance_loss(
                _logits(z_clean[idx], w, b), _logits(z_copy, w, b), y[idx], lam, sw[idx]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        if can_stop:
            with torch.no_grad():
                val_loss = stopping_loss(val_a, held_out, w, b)
            if val_loss < best[0] - 1e-6:
                best, stale = (val_loss, w.detach().numpy().copy(), float(b), epoch), 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
    info: dict[str, Any]
    if can_stop:
        _, w_best, b_best, best_epoch = best
        info = {"epochs_run": epoch, "best_epoch": best_epoch, "val_a_loss": round(best[0], 6)}
    else:
        w_best, b_best = w.detach().numpy().copy(), float(b)
        info = {
            "epochs_run": epoch,
            "best_epoch": epoch,
            "warning": "val-A lacks both classes: no early stopping, fixed epoch count used",
        }
    info.update({"lambda_p": lam, "candidates": list(cands), "train_faces": int(len(usable))})
    return w_best.astype(np.float32), b_best, info


def val_a_rule_stats(data: Data, w: NDArray[np.float32], b: float) -> tuple[float, float]:
    """(clean AUC, flip rate) on val-A, with flips counted at logit 0 over F1-F3 copies."""
    logits = data.z @ w + b
    clean = logits[:, data.names.index("clean")]
    flips = []
    for fam in ("F1", "F2", "F3"):
        for c in FAMILY_COPIES[fam]:
            j = data.names.index(c)
            ok = data.present[:, j]
            flips.append((clean[ok] >= 0) != (logits[ok, j] >= 0))
    flip = float(np.concatenate(flips).mean()) if flips else float("nan")
    return metrics.auc(data.y, clean), flip


def choose_lambda(
    train: Data, val_a: Data, w0: NDArray[np.float32], b0: float, cfg: TrainConfig
) -> tuple[float, list[dict[str, Any]], str]:
    """Pre-registered rule: the lowest val-A flip rate among λp values whose val-A clean AUC
    is within 1 point of H-a's. Falls back to λp = 1.0 (and says so) if the rule cannot be
    applied."""
    auc_a, _ = val_a_rule_stats(val_a, w0, b0)
    rows = []
    for lam in LAMBDA_GRID:
        w, b, _ = train_one(SCHEMES["H-f"], None, 0, lam, train, val_a, w0, b0, cfg)
        auc_l, flip_l = val_a_rule_stats(val_a, w, b)
        rows.append({"lambda_p": lam, "val_a_clean_auc": auc_l, "val_a_flip_rate": flip_l})
    if not np.isfinite(auc_a):
        return 1.0, rows, "fallback λp = 1.0: val-A AUC undefined (needs both classes)"
    eligible = [
        r
        for r in rows
        if np.isfinite(r["val_a_clean_auc"])
        and r["val_a_clean_auc"] >= auc_a - 0.01
        and np.isfinite(r["val_a_flip_rate"])
    ]
    if not eligible:
        return 1.0, rows, "fallback λp = 1.0: no λp kept val-A clean AUC within 1 point of H-a"
    best = min(eligible, key=lambda r: (r["val_a_flip_rate"], r["lambda_p"]))
    return (
        float(best["lambda_p"]),
        rows,
        (f"lowest val-A flip rate with val-A clean AUC >= H-a ({auc_a:.4f}) - 0.01"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/heads"))
    parser.add_argument("--schemes", default=",".join(SCHEMES))
    parser.add_argument("--folds", default=",".join(FOLDS))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--lambda-p", type=float, default=1.0)
    parser.add_argument("--final", action="store_true", help="also train the shipped head")
    parser.add_argument("--final-out", type=Path, default=Path("weights/head.pt"))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    args = parser.parse_args(argv)
    log = common.setup_logging()
    cfg = TrainConfig(args.epochs, args.batch_size, args.lr, args.patience)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    train, val_a, w0, b0, meta = load_training_data(args.cache)
    cache_meta_sha = common.sha256_file(args.cache / "meta.json")
    base_meta = {
        "init": "gend-probe",
        "cache_meta_sha256": cache_meta_sha,
        "manifest_sha256": meta["manifest_sha256"],
        "fixture": bool(meta["fixture"]),
    }
    log.info("train faces: %d, val-A faces: %d", len(train.y), len(val_a.y))

    index_path = args.out_dir / "index.json"
    index: dict[str, Any] = json.loads(index_path.read_text()) if index_path.is_file() else {}
    for scheme_name in args.schemes.split(","):
        scheme = SCHEMES[scheme_name]
        for fold in args.folds.split(","):
            for seed in (int(s) for s in args.seeds.split(",")):
                key = f"{scheme.name}__{fold}__seed{seed}"
                entry = index.get(key)
                if (
                    entry
                    and entry.get("cache_meta_sha256") == cache_meta_sha
                    and (not scheme.trains or (args.out_dir / f"{key}.pt").is_file())
                ):
                    continue
                w, b, info = train_one(scheme, fold, seed, args.lambda_p, train, val_a, w0, b0, cfg)
                entry = {
                    "scheme": scheme.name,
                    "fold": fold,
                    "seed": seed,
                    "held_out": is_held_out(scheme, fold),
                    "cache_meta_sha256": cache_meta_sha,
                    **info,
                }
                if scheme.trains:
                    entry["sha256"] = save_head(
                        args.out_dir / f"{key}.pt",
                        w,
                        b,
                        {
                            **base_meta,
                            "scheme": scheme.name,
                            "fold": fold,
                            "seed": seed,
                            "lambda_p": float(info["lambda_p"]),
                        },
                    )
                    entry["file"] = f"{key}.pt"
                index[key] = entry
                common.atomic_write_json(index_path, index)
                log.info(
                    "%s: %s",
                    key,
                    {
                        k: entry[k]
                        for k in entry
                        if k in ("epochs_run", "best_epoch", "val_a_loss", "warning")
                    },
                )

    if args.final:
        lam, table, rule = choose_lambda(train, val_a, w0, b0, cfg)
        w, b, info = train_one(SCHEMES["H-f"], None, 0, lam, train, val_a, w0, b0, cfg)
        sha = save_head(
            args.final_out,
            w,
            b,
            {
                **base_meta,
                "scheme": "H-f",
                "fold": "none (all of F1-F3)",
                "seed": 0,
                "lambda_p": lam,
            },
        )
        report: dict[str, Any] = {
            "file": str(args.final_out),
            "sha256": sha,
            "scheme": "H-f",
            "families": ["F1", "F2", "F3"],
            "seed": 0,
            "lambda_p": lam,
            "lambda_rule": rule,
            "lambda_table": table,
            **base_meta,
            **info,
        }
        common.atomic_write_json(args.out_dir / "final.json", report)
        log.info("shipped head: λp = %s (%s), sha256 %s", lam, rule, sha[:12])
    if meta["fixture"]:
        log.warning(common.FIXTURE_LABEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
