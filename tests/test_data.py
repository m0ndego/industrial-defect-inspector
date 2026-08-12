from pathlib import Path

import numpy as np
from PIL import Image

from defect_inspector.data import deterministic_split, prepare_mvtec_category
from defect_inspector.data import test_samples as collect_test_samples


def write_image(path: Path, value: int, mask: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((32, 32), value, dtype=np.uint8)
    if mask:
        array[:, 14:18] = 255
    Image.fromarray(array).save(path)


def make_source(root: Path, normal_count: int = 10) -> Path:
    category = root / "bottle"
    for index in range(normal_count):
        write_image(category / "train" / "good" / f"{index:03d}.png", 100 + index)
    write_image(category / "test" / "good" / "000.png", 103)
    write_image(category / "test" / "crack" / "000.png", 180, mask=True)
    write_image(category / "ground_truth" / "crack" / "000_mask.png", 0, mask=True)
    return root


def test_deterministic_split_has_no_overlap(tmp_path: Path) -> None:
    files = [Path(f"{index}.png") for index in range(10)]
    first = deterministic_split(files, 0.2, 42)
    second = deterministic_split(files, 0.2, 42)
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert len(first[0]) == 8
    assert len(first[1]) == 2


def test_prepare_preserves_masks_and_heldout_normals(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source")
    destination = tmp_path / "processed" / "bottle"
    manifest = prepare_mvtec_category(source, destination, validation_ratio=0.2, seed=42)
    assert len(manifest["train"]) == 8
    assert len(manifest["calibration"]) == 2
    samples = collect_test_samples(destination)
    assert [label for _, label, _ in samples] == [1, 0]
    assert next(mask for _, label, mask in samples if label).is_file()
