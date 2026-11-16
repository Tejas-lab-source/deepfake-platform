"""Split assignment by identity component.

Whole components are assigned, so no identity and no video can appear in two splits and
no video is ever dropped. This is used for the plumbing fixture only. Real-data split
assignment is deliberately NOT available until the official list's identity graph has
been reviewed (see offline/celebdf_graph.py and the README).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from offline.celebdf import Component, stable_rank
from offline.common import SPLIT_OF_SUBSET, SUBSETS

DEFAULT_SUBSET_RATIOS = {"train": 0.6, "val-A": 0.1, "val-B": 0.1, "test": 0.2}


class LeakageError(AssertionError):
    """An identity or video appears in more than one split."""


class RealDataSplitPending(RuntimeError):
    """Real-data split assignment awaits review of the identity graph."""


def assign_components(
    components: list[Component], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """Return {group_id: subset}.

    Every subset first receives one component (largest first, in order of target ratio),
    then each remaining component goes to the subset furthest below its target video
    count. Deterministic for a given seed.
    """
    if set(ratios) != set(SUBSETS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must cover {SUBSETS} and sum to 1")
    if len(components) < len(SUBSETS):
        raise ValueError(f"need at least {len(SUBSETS)} components, got {len(components)}")
    ordered = sorted(components, key=lambda c: (-len(c.videos), stable_rank(c.group_id, seed)))
    total = sum(len(c.videos) for c in components)
    counts: Counter[str] = Counter()
    assignment: dict[str, str] = {}
    by_ratio = sorted(SUBSETS, key=lambda s: -ratios[s])
    for subset, comp in zip(by_ratio, ordered, strict=False):
        assignment[comp.group_id] = subset
        counts[subset] += len(comp.videos)
    for comp in ordered[len(SUBSETS) :]:
        subset = max(SUBSETS, key=lambda s: (ratios[s] * total - counts[s], -SUBSETS.index(s)))
        assignment[comp.group_id] = subset
        counts[subset] += len(comp.videos)
    return assignment


def check_no_leakage(videos: list[dict[str, object]]) -> None:
    """Refuse if any video ID or identity is in more than one split.

    `videos` items need "video_id", "identities" (list) and "subset".
    """
    video_split: dict[str, str] = {}
    identity_splits: defaultdict[str, set[str]] = defaultdict(set)
    for v in videos:
        vid, subset = str(v["video_id"]), str(v["subset"])
        if subset not in SPLIT_OF_SUBSET:
            raise LeakageError(f"{vid}: unknown subset {subset!r}")
        split = SPLIT_OF_SUBSET[subset]
        if vid in video_split and video_split[vid] != split:
            raise LeakageError(f"video {vid} is in both {video_split[vid]} and {split}")
        video_split[vid] = split
        identities = v["identities"]
        assert isinstance(identities, list)
        for ident in identities:
            identity_splits[str(ident)].add(split)
    crossing = {i: sorted(s) for i, s in identity_splits.items() if len(s) > 1}
    if crossing:
        sample = dict(list(crossing.items())[:5])
        raise LeakageError(f"{len(crossing)} identities cross splits, e.g. {sample}")


def check_face_leakage(faces: list[dict[str, object]], video_subset: dict[str, str]) -> None:
    """Every face's subset must be its video's subset, and face IDs must be unique."""
    seen: set[str] = set()
    for f in faces:
        fid, vid = str(f["face_id"]), str(f["video_id"])
        if fid in seen:
            raise LeakageError(f"duplicate face {fid}")
        seen.add(fid)
        if f.get("subset") != video_subset.get(vid):
            raise LeakageError(f"face {fid} subset differs from its video {vid}")
