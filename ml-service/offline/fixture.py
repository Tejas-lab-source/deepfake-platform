"""The plumbing fixture: a miniature dataset laid out like Celeb-DF v2.

FIXTURE DATA IS FOR PLUMBING TESTS ONLY. Every face is a variant of one public-domain
photo (scikit-image's astronaut, NASA), and the real/fake labels are arbitrary. Nothing
computed on it says anything about accuracy.

Layout: 5 identity components, each with one real and one fake video, 2 face frames per
video = 20 faces. One fake video has an extra frame with no face, which must be excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MARKER = "FIXTURE_NOT_REAL_DATA.txt"
MARKER_TEXT = (
    "This folder is a generated plumbing fixture. Faces are variants of one public-domain "
    "photo and the labels are arbitrary. Results computed on it are not accuracy results.\n"
)


@dataclass(frozen=True)
class FixtureVideo:
    path: str  # relative to the fixture root
    list_label: str  # as in the official list: "1" real, "0" fake
    variant: int  # which astronaut variant the frames use
    face_frames: int = 2
    blank_frames: int = 0


def fixture_videos() -> list[FixtureVideo]:
    videos = []
    for k in range(5):
        videos.append(FixtureVideo(f"Celeb-real/id{k}_0000.mp4", "1", variant=2 * k))
        videos.append(
            FixtureVideo(
                f"Celeb-synthesis/id{k}_id{k + 5}_0000.mp4",
                "0",
                variant=2 * k + 1,
                blank_frames=1 if k == 0 else 0,
            )
        )
    return videos


def fixture_list_text() -> str:
    return "".join(f"{v.list_label} {v.path}\n" for v in fixture_videos())


def is_fixture(root: Path) -> bool:
    return (root / MARKER).is_file()
