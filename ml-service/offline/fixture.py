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


FRAME_SIZE = 320


def _variant_frame(base: object, variant: int, frame: int) -> object:
    """One RGB frame of a fixture video: the astronaut photo, shifted, scaled, flipped and
    colour-shifted by `variant`, with a small per-frame jitter."""
    import cv2
    import numpy as np

    img = np.asarray(base)
    rng = np.random.default_rng(1000 * variant + frame)
    zoom = 0.75 + 0.05 * (variant % 4)
    side = int(min(img.shape[:2]) * zoom)
    # The astronaut's face is in the upper-middle of the photo; keep it inside the crop.
    x0 = int(150 - (side - 220) / 2 + rng.integers(-6, 7))
    y0 = int(40 + rng.integers(-6, 7))
    x0 = int(np.clip(x0, 0, img.shape[1] - side))
    y0 = int(np.clip(y0, 0, img.shape[0] - side))
    crop = img[y0 : y0 + side, x0 : x0 + side]
    if variant % 2:
        crop = crop[:, ::-1]
    out = cv2.resize(np.ascontiguousarray(crop), (FRAME_SIZE, FRAME_SIZE))
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + 7 * variant) % 180
    hsv[..., 2] = np.clip(hsv[..., 2] + 4 * (variant % 3) - 4, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def write_fixture(root: Path) -> list[Path]:
    """Generate the fixture under `root`: the marker, the test list and the videos."""
    import cv2
    import numpy as np
    from skimage import data

    base = data.astronaut()
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text(MARKER_TEXT)
    (root / "List_of_testing_videos.txt").write_text(fixture_list_text())
    written = []
    for video in fixture_videos():
        path = root / video.path
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"mp4v"), 5.0, (FRAME_SIZE, FRAME_SIZE)
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot write MP4 files here ({path})")
        frames = [_variant_frame(base, video.variant, i) for i in range(video.face_frames)]
        frames += [np.full((FRAME_SIZE, FRAME_SIZE, 3), 90, np.uint8)] * video.blank_frames
        for rgb in frames:
            writer.write(np.ascontiguousarray(np.asarray(rgb)[..., ::-1]))
        writer.release()
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the plumbing fixture dataset.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = write_fixture(args.out)
    print(f"wrote {len(paths)} fixture videos to {args.out} ({MARKER})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
