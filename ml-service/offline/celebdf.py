"""Celeb-DF v2 test-list parsing and the identity graph.

The official list (`List_of_testing_videos.txt`, shipped with the dataset) has one video
per line: `<label> <folder>/<file>`. In it, 1 means real and 0 means fake; this matches
how the official SBI code reads the list (`label = 1 - int(line[0])` with its own 1 = fake).
Every label is checked against its folder, and the parser refuses the file on any mismatch.

Identities come from file names:

* `Celeb-real/id{N}_{XXXX}.mp4`          -> one identity, id{N}
* `Celeb-synthesis/id{A}_id{B}_{XXXX}.mp4` -> two identities, id{A} and id{B}
* `YouTube-real/{XXXXX}.mp4`             -> no celebrity identity

A fake involves two people, so identities are joined with a union-find. A connected
component is the smallest unit that can be placed in one split without any identity
appearing in two splits. This module only analyses; it never drops videos.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

FOLDER_LABEL = {"Celeb-real": "real", "YouTube-real": "real", "Celeb-synthesis": "fake"}
# In the official list: 1 = real, 0 = fake.
LIST_LABEL = {"1": "real", "0": "fake"}

_REAL = re.compile(r"^(id\d+)_\d+$")
_FAKE = re.compile(r"^(id\d+)_(id\d+)_\d+$")
_YOUTUBE = re.compile(r"^\d+$")


class ListFormatError(ValueError):
    """The test list does not match the expected format."""


@dataclass(frozen=True)
class VideoEntry:
    path: str  # as written in the list, e.g. "Celeb-real/id0_0001.mp4"
    folder: str
    label: str  # "real" or "fake"
    identities: tuple[str, ...]

    @property
    def video_id(self) -> str:
        """Stable ID: the path without its extension, with "/" replaced by "__"."""
        return str(Path(self.path).with_suffix("")).replace("/", "__")

    @property
    def y(self) -> int:
        """Training target: 1 = fake, 0 = real."""
        return 1 if self.label == "fake" else 0


def parse_identities(folder: str, stem: str) -> tuple[str, ...]:
    if folder == "Celeb-real":
        m = _REAL.match(stem)
        if m:
            return (m.group(1),)
    elif folder == "Celeb-synthesis":
        m = _FAKE.match(stem)
        if m:
            return (m.group(1), m.group(2))
    elif folder == "YouTube-real":
        if _YOUTUBE.match(stem):
            return ()
    raise ListFormatError(f"unexpected file name {folder}/{stem}")


def parse_test_list(text: str) -> list[VideoEntry]:
    """Parse the test list. Raises ListFormatError on any malformed or inconsistent line."""
    entries: list[VideoEntry] = []
    seen: set[str] = set()
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] not in LIST_LABEL:
            raise ListFormatError(f"line {number}: expected '<0|1> <folder>/<file>': {raw!r}")
        label, path = LIST_LABEL[parts[0]], parts[1]
        folder, _, name = path.partition("/")
        if folder not in FOLDER_LABEL or not name or "/" in name:
            raise ListFormatError(f"line {number}: unexpected path {path!r}")
        if FOLDER_LABEL[folder] != label:
            raise ListFormatError(
                f"line {number}: label {parts[0]} ({label}) contradicts folder {folder} "
                f"({FOLDER_LABEL[folder]})"
            )
        if path in seen:
            raise ListFormatError(f"line {number}: duplicate video {path}")
        seen.add(path)
        entries.append(VideoEntry(path, folder, label, parse_identities(folder, Path(name).stem)))
    if not entries:
        raise ListFormatError("the list is empty")
    return entries


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: the lexicographically smaller root wins.
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo


@dataclass(frozen=True)
class Component:
    group_id: str
    identities: tuple[str, ...]
    videos: tuple[VideoEntry, ...]

    @property
    def labels(self) -> Counter[str]:
        return Counter(v.label for v in self.videos)


def identity_components(entries: list[VideoEntry]) -> list[Component]:
    """Group videos into identity components. Videos without an identity are singletons.

    Sorted by number of videos (largest first), then by group ID.
    """
    uf = UnionFind()
    for e in entries:
        for ident in e.identities:
            uf.add(ident)
        for a, b in zip(e.identities, e.identities[1:], strict=False):
            uf.union(a, b)
    groups: dict[str, list[VideoEntry]] = {}
    for e in entries:
        key = f"cc:{uf.find(e.identities[0])}" if e.identities else f"solo:{e.video_id}"
        groups.setdefault(key, []).append(e)
    components = []
    for key, videos in groups.items():
        idents = sorted({i for v in videos for i in v.identities})
        components.append(Component(key, tuple(idents), tuple(videos)))
    return sorted(components, key=lambda c: (-len(c.videos), c.group_id))


def stable_rank(key: str, seed: int) -> str:
    """A seeded, platform-independent ordering key."""
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def simulate_identity_split(
    entries: list[VideoEntry], ratios: dict[str, float], seed: int
) -> dict[str, object]:
    """READ-ONLY simulation: assign identities to splits at random by the ratios, then count
    the fakes whose two identities land in different splits (they would have to be dropped).

    YouTube videos (no identity) are assigned individually. Nothing is dropped.
    """
    idents = sorted({i for e in entries for i in e.identities}, key=lambda i: stable_rank(i, seed))
    names = list(ratios)
    bounds, acc = [], 0.0
    for name in names:
        acc += ratios[name]
        bounds.append(acc)

    def pick(position: float) -> str:
        for name, bound in zip(names, bounds, strict=True):
            if position < bound:
                return name
        return names[-1]

    ident_split = {i: pick((k + 0.5) / len(idents)) for k, i in enumerate(idents)}
    kept: Counter[str] = Counter()
    spanning: list[str] = []
    for e in entries:
        if e.identities:
            splits = {ident_split[i] for i in e.identities}
            if len(splits) > 1:
                spanning.append(e.path)
                continue
            kept[splits.pop()] += 1
        else:
            kept[pick(int(stable_rank(e.video_id, seed)[:8], 16) / 16**8)] += 1
    return {
        "seed": seed,
        "spanning_fake_videos": len(spanning),
        "videos_discarded": len(spanning),
        "videos_kept_per_split": {n: kept.get(n, 0) for n in names},
    }


def graph_report(
    entries: list[VideoEntry], ratios: dict[str, float], seeds: range
) -> dict[str, object]:
    """Statistics for review before any split is fixed. Analysis only."""
    components = identity_components(entries)
    identity_cc = [c for c in components if c.group_id.startswith("cc:")]
    largest = components[0]
    sims = [simulate_identity_split(entries, ratios, s) for s in seeds]
    discards = sorted(int(str(s["videos_discarded"])) for s in sims)
    return {
        "videos": len(entries),
        "videos_by_folder": dict(Counter(e.folder for e in entries)),
        "videos_by_label": dict(Counter(e.label for e in entries)),
        "identities": len({i for e in entries for i in e.identities}),
        "videos_without_identity": sum(1 for e in entries if not e.identities),
        "components_total": len(components),
        "identity_components": len(identity_cc),
        "singleton_components_without_identity": len(components) - len(identity_cc),
        "largest_component": {
            "group_id": largest.group_id,
            "identities": len(largest.identities),
            "videos": len(largest.videos),
            "share_of_videos": round(len(largest.videos) / len(entries), 4),
            "labels": dict(largest.labels),
        },
        "component_sizes_videos": [len(c.videos) for c in components],
        "identity_level_split_simulation": {
            "ratios": ratios,
            "note": "Identities assigned at random by the ratios; fakes spanning two splits "
            "would have to be discarded. Simulation only: nothing is dropped.",
            "seed_0": sims[0],
            "discarded_over_seeds": {
                "seeds": len(sims),
                "min": discards[0],
                "median": discards[len(discards) // 2],
                "max": discards[-1],
            },
        },
    }
