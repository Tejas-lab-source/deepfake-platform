"""The feature-cache copy inventory (approved in the Phase 3 plan) and how each copy is
made. Every random choice is drawn from a generator seeded by (face ID, copy name), so any
copy can be regenerated exactly, for example to rebuild an F3 transform for heatmap IoU.

| Copy        | Family              | Strength       | Splits            |
|-------------|---------------------|----------------|-------------------|
| clean       | —                   | 0              | train, val, test  |
| F{1,2,3}_lo | F1 / F2 / F3        | U(0.30, 0.65)  | train, val, test  |
| F{1,2,3}_hi | F1 / F2 / F3        | U(0.65, 1.00)  | train, val, test  |
| F1_bilat    | F1, bilateral only  | U(0.30, 1.00)  | train             |
| generic     | blur + JPEG + jitter| random params  | train             |
| mix         | two of F1–F3        | U(0.30, 1.00)  | val, test         |
| F4          | GFPGAN              | U(0.50, 1.00)  | test (if installed)|

F1 draws its smoothing method (bilateral, guided, Gaussian) like the service does.

Inference-style copies (the service's K = 4: F1, F2, F3, mix) are NOT computed again:
each single-family slot reuses F*_lo or F*_hi, chosen with probability 1/2 by a seeded
draw. Since lo and hi are uniform on adjacent halves of [0.30, 1.00], the chosen strength
is uniform on [0.30, 1.00], the service's distribution. Only `mix` is new.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from offline.common import rng_for

INVENTORY_VERSION = 1
SINGLE = ("F1_lo", "F1_hi", "F2_lo", "F2_hi", "F3_lo", "F3_hi")
COPIES_BY_SPLIT: dict[str, tuple[str, ...]] = {
    "train": ("clean", *SINGLE, "F1_bilat", "generic"),
    "val": ("clean", *SINGLE, "mix"),
    "test": ("clean", *SINGLE, "mix", "F4"),
}
STRENGTH = {"lo": (0.30, 0.65), "hi": (0.65, 1.00), "any": (0.30, 1.00), "F4": (0.50, 1.00)}
F1_METHODS = ("bilateral", "guided", "gaussian")
INFERENCE_SLOTS = ("F1", "F2", "F3", "mix")


@dataclass(frozen=True)
class CopyPlan:
    face_id: str
    name: str
    # (family, strength, F1 method or None), in application order
    filters: tuple[tuple[str, float, str | None], ...]
    generic: dict[str, float] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filters": [list(f) for f in self.filters],
            "generic": self.generic,
        }

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(f[0] for f in self.filters)


def _f1_method(rng: np.random.Generator) -> str:
    return str(F1_METHODS[int(rng.integers(len(F1_METHODS)))])


def plan_copy(face_id: str, name: str) -> CopyPlan:
    rng = rng_for("copy", face_id, name)
    if name == "clean":
        return CopyPlan(face_id, name, ())
    if name in SINGLE:
        family, band = name.split("_")
        strength = float(rng.uniform(*STRENGTH[band]))
        method = _f1_method(rng) if family == "F1" else None
        return CopyPlan(face_id, name, ((family, strength, method),))
    if name == "F1_bilat":
        return CopyPlan(face_id, name, (("F1", float(rng.uniform(*STRENGTH["any"])), "bilateral"),))
    if name == "generic":
        from offline import generic_aug

        return CopyPlan(face_id, name, (), generic=generic_aug.draw_params(rng))
    if name == "mix":
        pair = sorted(rng.choice(["F1", "F2", "F3"], size=2, replace=False).tolist())
        filters = []
        for family in pair:
            strength = float(rng.uniform(*STRENGTH["any"]))
            filters.append((family, strength, _f1_method(rng) if family == "F1" else None))
        return CopyPlan(face_id, name, tuple(filters))
    if name == "F4":
        return CopyPlan(face_id, name, (("F4", float(rng.uniform(*STRENGTH["F4"])), None),))
    raise ValueError(f"unknown copy {name!r}")


class CopyUnavailable(RuntimeError):
    """The copy cannot be made here (F4 without GFPGAN)."""


def render_copy(
    plan: CopyPlan, crop: NDArray[np.uint8], geom: Any, weights_dir: Path
) -> tuple[NDArray[np.uint8], Any]:
    """Make the copy's image. Returns (image, F3 transform or None)."""
    from app.beautify import f1_smooth, f2_makeup, f3_reshape, f4_restore

    if plan.name == "clean":
        return crop.copy(), None
    if plan.generic is not None:
        from offline import generic_aug

        return generic_aug.apply(crop, plan.generic), None
    out, transform = crop, None
    apply_rng = rng_for("apply", plan.face_id, plan.name)
    for family, strength, method in plan.filters:
        if family == "F1":
            out = f1_smooth.apply(out, geom, strength, apply_rng, method=method)
        elif family == "F2":
            out = f2_makeup.apply(out, geom, strength, apply_rng)
        elif family == "F3":
            out, transform = f3_reshape.apply(out, geom, strength, apply_rng)
        elif family == "F4":
            try:
                out = f4_restore.apply(out, strength, weights_dir)
            except f4_restore.F4Unavailable as exc:
                raise CopyUnavailable(str(exc)) from exc
        else:
            raise ValueError(f"unknown family {family!r}")
    return out, transform


def inference_pick(face_id: str) -> dict[str, str]:
    """Which cached copy fills each single-family inference slot."""
    rng = rng_for("inference", face_id)
    return {f: f"{f}_{'lo' if rng.random() < 0.5 else 'hi'}" for f in ("F1", "F2", "F3")}


def inference_copies(face_id: str) -> tuple[str, ...]:
    """Cached copy names standing in for the service's clean + K = 4 copies."""
    pick = inference_pick(face_id)
    return ("clean", pick["F1"], pick["F2"], pick["F3"], "mix")
