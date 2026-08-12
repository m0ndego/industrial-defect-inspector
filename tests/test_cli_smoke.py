from pathlib import Path

import yaml
from test_data import make_source

from defect_inspector.cli import main


def test_autoencoder_cli_smoke(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", normal_count=6)
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    config = {
        "seed": 42,
        "data": {
            "category": "bottle",
            "root": "data/processed/bottle",
            "image_size": 32,
            "validation_ratio": 0.33,
            "num_workers": 0,
        },
        "autoencoder": {"batch_size": 2, "epochs": 1, "learning_rate": 0.001},
        "patchcore": {
            "backbone": "resnet18",
            "layers": ["layer2", "layer3"],
            "pre_trained": False,
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
        },
        "scoring": {"normal_threshold_quantile": 0.99, "image_score_quantile": 0.99},
        "artifacts": {"root": "artifacts"},
    }
    config_path = config_dir / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    prefix = ["--config", str(config_path)]

    assert main([*prefix, "prepare-data", "--source", str(source)]) == 0
    assert main([*prefix, "train", "--model", "autoencoder"]) == 0
    assert main([*prefix, "evaluate", "--model", "autoencoder"]) == 0
    sample = project / "data" / "processed" / "bottle" / "test" / "crack" / "000.png"
    assert main([*prefix, "predict", "--model", "autoencoder", "--input", str(sample)]) == 0
    assert (project / "artifacts" / "autoencoder" / "metrics.json").is_file()
    assert (project / "artifacts" / "autoencoder" / "predictions" / "predictions.json").is_file()
