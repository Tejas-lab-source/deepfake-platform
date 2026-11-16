"""The six head schemes, exactly as defined in CLAUDE.md §5 item 3, and their LOFO folds.

| Scheme | Head trained on                                   | Invariance loss |
|--------|---------------------------------------------------|-----------------|
| H-a    | No head (GenD as released)                        | —               |
| H-b    | Clean + bilateral smoothing only                  | No              |
| H-c    | Clean + random F1–F3 (the critical control)       | No              |
| H-d    | Clean + generic augmentations                     | Yes             |
| H-e    | Clean + random F1–F3                              | Yes             |
| H-f    | Clean + random F1–F3, hardest-of-K (the headline) | Yes             |

Implementation details (approved): "random F1–F3" draws one cached copy per face per step
from the fold's two training families (2 families x 2 strengths); H-f takes, from the same
candidates, the copy whose logit moves most under the current head; H-d uses the single
cached generic copy; H-b uses the dedicated bilateral F1 copy, so its F1-held-out fold is
NOT HELD OUT and is excluded from aggregates; schemes without invariance loss use λp = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FAMILIES = ("F1", "F2", "F3")
FOLDS = ("F1", "F2", "F3")  # the held-out family of each fold
SEEDS = (0, 1, 2)
FAMILY_COPIES = {f: (f"{f}_lo", f"{f}_hi") for f in FAMILIES}


@dataclass(frozen=True)
class Scheme:
    name: str
    trains: bool
    candidates: str  # "none", "bilateral", "families", "generic"
    selection: str  # "none", "random", "hardest"
    invariance: bool
    description: str


SCHEMES: dict[str, Scheme] = {
    s.name: s
    for s in (
        Scheme("H-a", False, "none", "none", False, "No head (GenD as released)"),
        Scheme("H-b", True, "bilateral", "random", False, "Clean + bilateral smoothing only"),
        Scheme("H-c", True, "families", "random", False, "Clean + random F1–F3 (control)"),
        Scheme("H-d", True, "generic", "random", True, "Clean + generic augmentations"),
        Scheme("H-e", True, "families", "random", True, "Clean + random F1–F3"),
        Scheme("H-f", True, "families", "hardest", True, "Clean + random F1–F3, hardest-of-K"),
    )
}


def training_families(held_out: str | None) -> tuple[str, ...]:
    """Families a head may train on. held_out=None (the shipped head) uses all three."""
    if held_out is not None and held_out not in FAMILIES:
        raise ValueError(f"unknown family {held_out!r}")
    return tuple(f for f in FAMILIES if f != held_out)


def candidate_copies(scheme: Scheme, held_out: str | None) -> tuple[str, ...]:
    if scheme.candidates == "none":
        return ()
    if scheme.candidates == "bilateral":
        return ("F1_bilat",)
    if scheme.candidates == "generic":
        return ("generic",)
    return tuple(c for f in training_families(held_out) for c in FAMILY_COPIES[f])


def is_held_out(scheme: Scheme, held_out: str) -> bool:
    """False when the scheme trains on the fold's held-out family (H-b in the F1 fold)."""
    return not (scheme.candidates == "bilateral" and held_out == "F1")


def lambda_for(scheme: Scheme, lambda_p: float) -> float:
    return lambda_p if scheme.invariance else 0.0


def select_copies(
    selection: str,
    clean_logits: NDArray[np.floating[Any]],
    candidate_logits: NDArray[np.floating[Any]],
    available: NDArray[np.bool_],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Pick one candidate per face. Returns column indices, or -1 where none is available.

    clean_logits: (N,); candidate_logits and available: (N, K).
    "random": uniform among available; "hardest": max |candidate - clean| among available.
    """
    n, k = candidate_logits.shape
    picks = np.full(n, -1, dtype=np.int64)
    has_any = available.any(axis=1)
    if selection == "hardest":
        moved = np.abs(candidate_logits - clean_logits[:, None])
        moved = np.where(available, moved, -np.inf)
        picks[has_any] = np.argmax(moved[has_any], axis=1)
    elif selection == "random":
        noise = rng.random((n, k))
        noise = np.where(available, noise, -1.0)
        picks[has_any] = np.argmax(noise[has_any], axis=1)
    else:
        raise ValueError(f"unknown selection {selection!r}")
    return picks
