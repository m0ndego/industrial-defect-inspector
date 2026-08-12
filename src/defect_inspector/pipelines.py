"""Training, evaluation, and prediction pipelines for both project models."""

from __future__ import annotations

import json
import platform
import random
import shutil
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from defect_inspector.autoencoder import ConvAutoencoder
from defect_inspector.data import ImageDataset, image_files, test_samples
from defect_inspector.visualization import save_visualization


def project_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(config["_project_root"]) / path


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("warning: CUDA is unavailable; using CPU")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_score(anomaly_map: np.ndarray, quantile: float) -> float:
    if not 0.0 < quantile <= 1.0:
        raise ValueError("image score quantile must be in (0, 1]")
    return float(np.quantile(anomaly_map, quantile))


def calibrate_threshold(scores: list[float], quantile: float) -> float:
    if not scores:
        raise ValueError("Cannot calibrate a threshold without normal scores")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("threshold quantile must be in (0, 1]")
    return float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))


def compute_metrics(
    labels: list[int],
    scores: list[float],
    masks: list[np.ndarray],
    anomaly_maps: list[np.ndarray],
    threshold: float,
) -> dict[str, float]:
    if len(set(labels)) < 2:
        raise ValueError("Image AUROC requires both normal and anomalous test images")
    predictions = [int(score >= threshold) for score in scores]
    pixel_labels = np.concatenate([mask.reshape(-1) for mask in masks])
    pixel_scores = np.concatenate([anomaly_map.reshape(-1) for anomaly_map in anomaly_maps])
    if len(np.unique(pixel_labels)) < 2:
        raise ValueError("Pixel AUROC requires both normal and anomalous pixels")
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "pixel_auroc": float(roc_auc_score(pixel_labels, pixel_scores)),
        "image_f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def environment_metadata() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "anomalib": _package_version("anomalib"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _artifact_dir(config: dict, model_name: str) -> Path:
    return project_path(config, config["artifacts"]["root"]) / model_name


def _autoencoder_map(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    reconstructed = model(images)
    return torch.mean(torch.abs(images - reconstructed), dim=1)


def train_autoencoder(config: dict) -> Path:
    seed = int(config["seed"])
    seed_everything(seed)
    data_root = project_path(config, config["data"]["root"])
    image_size = int(config["data"]["image_size"])
    settings = config["autoencoder"]
    device = select_device()

    train_data = ImageDataset.normal_directory(data_root / "train" / "good", image_size)
    loader = DataLoader(
        train_data,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
    )
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    loss_function = nn.L1Loss()
    history = []

    for epoch in range(int(settings["epochs"])):
        model.train()
        total_loss = 0.0
        for images, _, _, _ in loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), images)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
        epoch_loss = total_loss / len(train_data)
        history.append(epoch_loss)
        print(f"epoch {epoch + 1:03d}: loss={epoch_loss:.6f}")

    artifact_dir = _artifact_dir(config, "autoencoder")
    checkpoint = artifact_dir / "model.pt"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "image_size": image_size}, checkpoint)

    calibration = predict_autoencoder_maps(
        model, image_files(data_root / "calibration" / "good"), image_size, device, config
    )
    threshold = calibrate_threshold(
        [item[1] for item in calibration], float(config["scoring"]["normal_threshold_quantile"])
    )
    _save_json(
        artifact_dir / "metadata.json",
        {
            "model": "autoencoder",
            "threshold": threshold,
            "history": history,
            "environment": environment_metadata(),
        },
    )
    return checkpoint


@torch.inference_mode()
def predict_autoencoder_maps(
    model: nn.Module,
    paths: list[Path],
    image_size: int,
    device: torch.device,
    config: dict,
) -> list[tuple[np.ndarray, float, float]]:
    from defect_inspector.data import load_rgb_tensor

    model.eval()
    results = []
    score_quantile = float(config["scoring"]["image_score_quantile"])
    for path in paths:
        image = load_rgb_tensor(path, image_size).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        anomaly_map = _autoencoder_map(model, image)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        map_array = anomaly_map.detach().cpu().numpy()
        results.append((map_array, image_score(map_array, score_quantile), latency_ms))
    return results


def _load_autoencoder(config: dict) -> tuple[ConvAutoencoder, torch.device, float]:
    artifact_dir = _artifact_dir(config, "autoencoder")
    checkpoint = artifact_dir / "model.pt"
    metadata_path = artifact_dir / "metadata.json"
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Autoencoder checkpoint is missing; run train first")
    device = select_device()
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(payload["state_dict"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, device, float(metadata["threshold"])


def train_patchcore(config: dict) -> Path:
    try:
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except ImportError as error:
        raise RuntimeError("Anomalib 2.6.0 is required for PatchCore") from error

    seed_everything(int(config["seed"]))
    data_root = project_path(config, config["data"]["root"])
    image_size = int(config["data"]["image_size"])
    settings = config["patchcore"]
    artifact_dir = _artifact_dir(config, "patchcore")
    pre_processor = Patchcore.configure_pre_processor(image_size=(image_size, image_size))
    model = Patchcore(
        backbone=str(settings["backbone"]),
        layers=list(settings["layers"]),
        pre_trained=bool(settings.get("pre_trained", True)),
        coreset_sampling_ratio=float(settings["coreset_sampling_ratio"]),
        num_neighbors=int(settings["num_neighbors"]),
        pre_processor=pre_processor,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    datamodule = MVTecAD(
        root=data_root.parent,
        category=str(config["data"]["category"]),
        train_batch_size=int(config["autoencoder"]["batch_size"]),
        eval_batch_size=int(config["autoencoder"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        seed=int(config["seed"]),
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator == "cpu":
        print("warning: CUDA is unavailable; PatchCore will use CPU")
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=True,
        default_root_dir=artifact_dir / "engine",
    )
    engine.fit(model=model, datamodule=datamodule)

    checkpoint_candidates = list((artifact_dir / "engine").rglob("model.ckpt"))
    if engine.best_model_path:
        source_checkpoint = Path(engine.best_model_path)
    elif checkpoint_candidates:
        source_checkpoint = checkpoint_candidates[-1]
    else:
        raise RuntimeError("Anomalib did not produce a PatchCore checkpoint")
    checkpoint = artifact_dir / "model.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, checkpoint)

    calibration = _predict_patchcore(config, data_root / "calibration" / "good", checkpoint)
    threshold = calibrate_threshold(
        [item[1] for item in calibration], float(config["scoring"]["normal_threshold_quantile"])
    )
    _save_json(
        artifact_dir / "metadata.json",
        {"model": "patchcore", "threshold": threshold, "environment": environment_metadata()},
    )
    return checkpoint


def _patchcore_model(config: dict):
    from anomalib.models import Patchcore

    image_size = int(config["data"]["image_size"])
    settings = config["patchcore"]
    return Patchcore(
        backbone=str(settings["backbone"]),
        layers=list(settings["layers"]),
        pre_trained=bool(settings.get("pre_trained", True)),
        coreset_sampling_ratio=float(settings["coreset_sampling_ratio"]),
        num_neighbors=int(settings["num_neighbors"]),
        pre_processor=Patchcore.configure_pre_processor(image_size=(image_size, image_size)),
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )


def _flatten_predictions(predictions) -> list[tuple[Path, np.ndarray, float, float]]:
    flattened = []
    for batch in predictions or []:
        paths = list(batch.image_path)
        maps = batch.anomaly_map.detach().cpu().numpy()
        scores = batch.pred_score.detach().cpu().numpy().reshape(-1)
        for index, path in enumerate(paths):
            anomaly_map = np.asarray(maps[index]).squeeze()
            flattened.append((Path(path), anomaly_map, float(scores[index]), 0.0))
    return flattened


def _predict_patchcore(
    config: dict, input_path: str | Path, checkpoint: Path
) -> list[tuple[Path, float, np.ndarray, float]]:
    from anomalib.engine import Engine

    if not checkpoint.is_file():
        raise FileNotFoundError("PatchCore checkpoint is missing; run train first")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=False,
        default_root_dir=_artifact_dir(config, "patchcore") / "predict-engine",
    )
    if accelerator == "gpu":
        torch.cuda.synchronize()
    started = time.perf_counter()
    batches = engine.predict(
        model=_patchcore_model(config),
        ckpt_path=checkpoint,
        data_path=Path(input_path),
        return_predictions=True,
    )
    if accelerator == "gpu":
        torch.cuda.synchronize()
    flattened = _flatten_predictions(batches)
    latency_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(flattened))
    return [(path, score, anomaly_map, latency_ms) for path, anomaly_map, score, _ in flattened]


def _model_threshold(config: dict, model_name: str) -> float:
    metadata_path = _artifact_dir(config, model_name) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{model_name} metadata is missing; run train first")
    return float(json.loads(metadata_path.read_text(encoding="utf-8"))["threshold"])


def evaluate_model(config: dict, model_name: str) -> Path:
    data_root = project_path(config, config["data"]["root"])
    samples = test_samples(data_root)
    paths = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    image_size = int(config["data"]["image_size"])
    threshold = _model_threshold(config, model_name)

    if model_name == "autoencoder":
        model, device, _ = _load_autoencoder(config)
        raw = predict_autoencoder_maps(model, paths, image_size, device, config)
        predictions = [
            (path, score, anomaly_map, latency)
            for path, (anomaly_map, score, latency) in zip(paths, raw, strict=True)
        ]
    elif model_name == "patchcore":
        predictions = _predict_patchcore(
            config, data_root / "test", _artifact_dir(config, "patchcore") / "model.ckpt"
        )
        by_path = {
            path.resolve(): (path, score, anomaly_map, latency)
            for path, score, anomaly_map, latency in predictions
        }
        predictions = [by_path[path.resolve()] for path in paths]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    masks = []
    anomaly_maps = []
    scores = []
    latencies = []
    from defect_inspector.data import load_mask_tensor

    preview_dir = _artifact_dir(config, model_name) / "evaluation-images"
    paired_results = zip(samples, predictions, strict=True)
    for index, ((_, _, mask_path), (path, score, anomaly_map, latency)) in enumerate(
        paired_results
    ):
        masks.append(load_mask_tensor(mask_path, image_size).numpy())
        anomaly_maps.append(anomaly_map)
        scores.append(score)
        latencies.append(latency)
        if index < 12:
            save_visualization(path, anomaly_map, preview_dir)

    metrics = compute_metrics(labels, scores, masks, anomaly_maps, threshold)
    report = {
        "model": model_name,
        "category": config["data"]["category"],
        "seed": int(config["seed"]),
        **metrics,
        "threshold": threshold,
        "mean_latency_ms": float(np.mean(latencies)),
        "test_images": len(samples),
        "environment": environment_metadata(),
    }
    path = _save_json(_artifact_dir(config, model_name) / "metrics.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return path


def predict_model(
    config: dict, model_name: str, input_path: str | Path, output_dir: str | Path | None = None
) -> Path:
    paths = image_files(input_path)
    output = (
        Path(output_dir)
        if output_dir is not None
        else _artifact_dir(config, model_name) / "predictions"
    )
    threshold = _model_threshold(config, model_name)
    image_size = int(config["data"]["image_size"])

    if model_name == "autoencoder":
        model, device, _ = _load_autoencoder(config)
        raw = predict_autoencoder_maps(model, paths, image_size, device, config)
        predictions = [
            (path, score, anomaly_map, latency)
            for path, (anomaly_map, score, latency) in zip(paths, raw, strict=True)
        ]
    elif model_name == "patchcore":
        predictions = _predict_patchcore(
            config, input_path, _artifact_dir(config, "patchcore") / "model.ckpt"
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    records = []
    for path, score, anomaly_map, latency in predictions:
        heatmap, overlay = save_visualization(path, anomaly_map, output)
        records.append(
            {
                "input": str(path.resolve()),
                "model": model_name,
                "anomaly_score": score,
                "threshold": threshold,
                "is_anomaly": bool(score >= threshold),
                "latency_ms": latency,
                "heatmap_path": str(heatmap.resolve()),
                "overlay_path": str(overlay.resolve()),
            }
        )
    report_path = _save_json(output / "predictions.json", records)
    print(json.dumps(records, indent=2, ensure_ascii=False))
    return report_path
