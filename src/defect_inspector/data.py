"""MVTec data preparation and small PyTorch datasets."""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Image path does not exist: {root}")
    candidates = root.rglob("*") if root.is_dir() else [root]
    files = [p for p in candidates if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise ValueError(f"No supported images found in: {root}")
    return sorted(files)


def deterministic_split(
    files: list[Path], validation_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if len(files) < 2:
        raise ValueError("At least two normal training images are required")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_ratio))
    return sorted(shuffled[validation_count:]), sorted(shuffled[:validation_count])


def _category_root(source: Path, category: str) -> Path:
    candidates = [source / category, source]
    for candidate in candidates:
        if (candidate / "train" / "good").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise ValueError(
        f"{source} is not an extracted MVTec AD root or {category} category directory"
    )


def _materialize(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare_mvtec_category(
    source: str | Path,
    destination: str | Path,
    category: str = "bottle",
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Create a canonical category tree with a held-out normal calibration split."""
    source_root = _category_root(Path(source).resolve(), category)
    destination_root = Path(destination).resolve()
    if destination_root.exists() and any(destination_root.iterdir()):
        raise FileExistsError(
            f"Destination is not empty: {destination_root}. Move it aside before preparing again."
        )

    normal_files = image_files(source_root / "train" / "good")
    train_files, calibration_files = deterministic_split(normal_files, validation_ratio, seed)

    records: dict[str, list[str] | str | int | float] = {
        "category": category,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "train": [],
        "calibration": [],
        "test": [],
    }

    for split, files in (("train", train_files), ("calibration", calibration_files)):
        for file in files:
            target = destination_root / split / "good" / file.name
            _materialize(file, target)
            records[split].append(target.relative_to(destination_root).as_posix())  # type: ignore[union-attr]

    for file in image_files(source_root / "test"):
        relative = file.relative_to(source_root)
        target = destination_root / relative
        _materialize(file, target)
        records["test"].append(target.relative_to(destination_root).as_posix())  # type: ignore[union-attr]

    ground_truth = source_root / "ground_truth"
    if ground_truth.exists():
        for mask in image_files(ground_truth):
            _materialize(mask, destination_root / mask.relative_to(source_root))

    validate_prepared_category(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return records


def validate_prepared_category(root: str | Path) -> None:
    root = Path(root)
    image_files(root / "train" / "good")
    image_files(root / "calibration" / "good")
    for image_path in image_files(root / "test"):
        defect_type = image_path.parent.name
        if defect_type == "good":
            continue
        mask = root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
        if not mask.is_file():
            raise ValueError(f"Missing mask for {image_path}: expected {mask}")


def load_rgb_tensor(path: str | Path, image_size: int) -> torch.Tensor:
    try:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot read image: {path}") from error
    return torch.from_numpy(array).permute(2, 0, 1)


def load_mask_tensor(path: str | Path | None, image_size: int) -> torch.Tensor:
    if path is None:
        return torch.zeros((image_size, image_size), dtype=torch.float32)
    try:
        with Image.open(path) as image:
            image = image.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
            return torch.from_numpy((np.asarray(image) > 0).astype(np.float32))
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Cannot read mask: {path}") from error


def test_samples(root: str | Path) -> list[tuple[Path, int, Path | None]]:
    root = Path(root)
    samples = []
    for image_path in image_files(root / "test"):
        defect_type = image_path.parent.name
        label = int(defect_type != "good")
        mask = (
            None
            if not label
            else root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
        )
        if mask is not None and not mask.is_file():
            raise ValueError(f"Missing mask for {image_path}")
        samples.append((image_path, label, mask))
    return samples


class ImageDataset(Dataset):
    """Return image, mask, label, and path without torchvision coupling."""

    def __init__(self, samples: list[tuple[Path, int, Path | None]], image_size: int) -> None:
        self.samples = samples
        self.image_size = image_size

    @classmethod
    def normal_directory(cls, path: str | Path, image_size: int) -> ImageDataset:
        return cls([(file, 0, None) for file in image_files(path)], image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, mask_path = self.samples[index]
        return (
            load_rgb_tensor(path, self.image_size),
            load_mask_tensor(mask_path, self.image_size),
            torch.tensor(label, dtype=torch.int64),
            str(path),
        )
