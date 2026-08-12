import numpy as np
import torch

from defect_inspector.autoencoder import ConvAutoencoder
from defect_inspector.pipelines import calibrate_threshold, compute_metrics, image_score


def test_autoencoder_preserves_image_shape() -> None:
    images = torch.rand(2, 3, 32, 32)
    output = ConvAutoencoder()(images)
    assert output.shape == images.shape
    assert torch.isfinite(output).all()


def test_threshold_and_metrics_are_finite() -> None:
    threshold = calibrate_threshold([0.1, 0.2, 0.3], 0.99)
    normal_map = np.zeros((4, 4), dtype=np.float32)
    anomaly_map = np.zeros((4, 4), dtype=np.float32)
    anomaly_map[1:3, 1:3] = 1.0
    metrics = compute_metrics(
        labels=[0, 1],
        scores=[0.1, 0.9],
        masks=[normal_map, anomaly_map],
        anomaly_maps=[normal_map, anomaly_map],
        threshold=threshold,
    )
    assert image_score(anomaly_map, 0.99) == 1.0
    assert metrics == {"image_auroc": 1.0, "pixel_auroc": 1.0, "image_f1": 1.0}
