"""Beautification library: identity-preserving filters, each with a strength in [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.beautify import f1_smooth, f2_makeup, f3_reshape
from app.beautify.f3_reshape import ReshapeTransform
from app.faces import FaceGeometry, Image

# Families allowed in training and in inference copies. F4 is test only.
TRAIN_FAMILIES = ("F1", "F2", "F3")
ALL_FAMILIES = (*TRAIN_FAMILIES, "F4")


@dataclass(frozen=True)
class Variant:
    name: str
    image: Image
    # (family, strength) pairs in the order they were applied.
    filters: tuple[tuple[str, float], ...]
    # Set when F3 was applied: lets a heatmap be warped back to the clean face.
    transform: ReshapeTransform | None = None


def apply_family(
    family: str, img: Image, geom: FaceGeometry, strength: float, rng: np.random.Generator
) -> tuple[Image, ReshapeTransform | None]:
    """Apply one of F1, F2 or F3. F4 has its own entry point in f4_restore."""
    if family == "F1":
        return f1_smooth.apply(img, geom, strength, rng), None
    if family == "F2":
        return f2_makeup.apply(img, geom, strength, rng), None
    if family == "F3":
        return f3_reshape.apply(img, geom, strength, rng)
    raise ValueError(f"unknown filter family: {family}")


def apply_chain(
    name: str,
    filters: tuple[tuple[str, float], ...],
    img: Image,
    geom: FaceGeometry,
    rng: np.random.Generator,
) -> Variant:
    """Apply filters in the fixed order F1, F2, F3.

    F3 goes last because it moves the face, which would invalidate the landmarks that F1
    and F2 use.
    """
    out, transform = img, None
    ordered = tuple(sorted(filters, key=lambda f: f[0]))
    for family, strength in ordered:
        out, t = apply_family(family, out, geom, strength, rng)
        transform = t or transform
    return Variant(name, out, ordered, transform)


def make_variants(
    img: Image,
    geom: FaceGeometry,
    k: int,
    rng: np.random.Generator,
    strength_range: tuple[float, float],
) -> list[Variant]:
    """K beautified copies: F1, F2, F3, then a random two-filter mix, repeating for K > 4."""
    lo, hi = strength_range
    variants: list[Variant] = []
    for i in range(max(0, k)):
        slot, lap = i % 4, i // 4
        suffix = "" if lap == 0 else f"_{lap + 1}"
        if slot < 3:
            family = TRAIN_FAMILIES[slot]
            filters: tuple[tuple[str, float], ...] = ((family, float(rng.uniform(lo, hi))),)
            name = family + suffix
        else:
            pair = rng.choice(len(TRAIN_FAMILIES), size=2, replace=False)
            filters = tuple((TRAIN_FAMILIES[int(j)], float(rng.uniform(lo, hi))) for j in pair)
            name = "mix" + suffix
        variants.append(apply_chain(name, filters, img, geom, rng))
    return variants
