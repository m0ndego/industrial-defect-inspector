"""Dependency-light anomaly heatmaps and overlays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    low, high = np.percentile(anomaly_map, [1, 99])
    if high <= low:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return np.clip((anomaly_map - low) / (high - low), 0.0, 1.0).astype(np.float32)


def heatmap_rgb(anomaly_map: np.ndarray) -> np.ndarray:
    values = normalize_map(anomaly_map)
    red = (255 * values).astype(np.uint8)
    blue = (255 * (1.0 - values)).astype(np.uint8)
    green = (100 * (1.0 - np.abs(2.0 * values - 1.0))).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def save_visualization(
    image_path: str | Path, anomaly_map: np.ndarray, output_dir: str | Path
) -> tuple[Path, Path]:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(image_path.resolve()).encode()).hexdigest()[:8]
    prefix = f"{image_path.stem}-{digest}"

    with Image.open(image_path) as source:
        source = source.convert("RGB")
        heatmap = Image.fromarray(heatmap_rgb(anomaly_map)).resize(
            source.size, Image.Resampling.BILINEAR
        )
        overlay = Image.blend(source, heatmap, alpha=0.45)

    heatmap_path = output_dir / f"{prefix}-heatmap.png"
    overlay_path = output_dir / f"{prefix}-overlay.png"
    heatmap.save(heatmap_path)
    overlay.save(overlay_path)
    return heatmap_path, overlay_path
