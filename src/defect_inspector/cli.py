"""Command-line interface for the project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

from defect_inspector.data import prepare_mvtec_category
from defect_inspector.pipelines import (
    evaluate_model,
    predict_model,
    project_path,
    train_autoencoder,
    train_patchcore,
)


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    config["_project_root"] = str(config_path.parent.parent)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Industrial defect inspection")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare-data", help="prepare an extracted MVTec category")
    prepare.add_argument("--source", required=True, help="MVTec AD root or category directory")

    train = commands.add_parser("train", help="train a model")
    train.add_argument("--model", required=True, choices=["autoencoder", "patchcore"])

    evaluate = commands.add_parser("evaluate", help="evaluate a trained model")
    evaluate.add_argument("--model", required=True, choices=["autoencoder", "patchcore"])

    predict = commands.add_parser("predict", help="predict one image or a directory")
    predict.add_argument("--model", required=True, choices=["autoencoder", "patchcore"])
    predict.add_argument("--input", required=True, help="image or directory")
    predict.add_argument("--output", help="output directory")
    return parser


def _make_console_robust() -> None:
    """Keep third-party Unicode logs from crashing legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    _make_console_robust()
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "prepare-data":
            prepare_mvtec_category(
                source=args.source,
                destination=project_path(config, config["data"]["root"]),
                category=str(config["data"]["category"]),
                validation_ratio=float(config["data"]["validation_ratio"]),
                seed=int(config["seed"]),
            )
        elif args.command == "train":
            (train_autoencoder if args.model == "autoencoder" else train_patchcore)(config)
        elif args.command == "evaluate":
            evaluate_model(config, args.model)
        elif args.command == "predict":
            predict_model(config, args.model, args.input, args.output)
        return 0
    except torch.cuda.OutOfMemoryError:
        print(
            "error: CUDA ran out of memory. Reduce batch_size or PatchCore coreset_sampling_ratio; "
            "the program did not change the experiment automatically.",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
