"""Regenerate the tiny validation split with deterministic, non-overlapping fixtures.

The demo is intentionally synthetic, but train and validation files must still be
different bytes so the runtime report cannot accidentally imply independent
evaluation when it is only measuring memorized files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
VAL_ROOT = ROOT / "demo" / "classification_dataset" / "val"
SIZE = (64, 64)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(class_name: str, index: int) -> Image.Image:
    """Create a distinct checker/stripe fixture for the requested class."""
    bright = class_name == "bright"
    base = (248, 214, 82) if bright else (61, 43, 105)
    accent = (255, 240, 155) if bright else (111, 84, 165)
    image = Image.new("RGB", SIZE, base)
    draw = ImageDraw.Draw(image)
    step = 8 + index * 2
    for y in range(0, SIZE[1], step):
        draw.line((0, y, SIZE[0], max(0, y - 18)), fill=accent, width=2)
    for x in range(index * 5, SIZE[0], 16):
        draw.ellipse((x, 8 + index * 7, min(x + 12, 63), 20 + index * 7), outline=accent, width=2)
    draw.rectangle((4 + index * 3, 44 - index * 3, 24 + index * 3, 58 - index * 2), outline=accent, width=2)
    return image


def main() -> None:
    for class_name in ("bright", "dark"):
        class_dir = VAL_ROOT / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in (1, 2):
            render(class_name, index).save(class_dir / f"{class_name}_{index}.png")

    train_hashes = {
        digest(path)
        for path in (ROOT / "demo" / "classification_dataset" / "train").rglob("*.png")
    }
    overlap = [path.relative_to(ROOT).as_posix() for path in VAL_ROOT.rglob("*.png") if digest(path) in train_hashes]
    if overlap:
        raise RuntimeError(f"Validation fixtures still overlap train hashes: {overlap}")
    print(f"Regenerated {len(list(VAL_ROOT.rglob('*.png')))} validation fixtures; train/val hash overlap: 0")


if __name__ == "__main__":
    main()
