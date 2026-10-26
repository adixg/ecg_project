"""Train and evaluate the compact CNN-BiLSTM baseline on PTB-XL."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import CLASS_NAMES, PTBXLDataset, make_data_loader
from src.metrics import (
    compute_metrics,
    compute_per_class_metrics,
    find_f1_thresholds,
)
from src.model import CNNBiLSTM


@dataclass(frozen=True)
class TrainConfig:
    data_root: str
    split_dir: str
    output_dir: str
    train_fraction: int = 100
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    lstm_hidden_size: int = 64
    patience: int = 8
    num_workers: int = 4
    normalize: bool = False
    use_pos_weight: bool = False
    device: str = "auto"


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("PTBXL_ROOT", "data/ptbxl"),
        help="Directory containing records100/.",
    )
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output-dir", default="outputs/cnn_bilstm")
    parser.add_argument(
        "--train-fraction", type=int, choices=(5, 10, 25, 50, 100), default=100
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lstm-hidden-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--use-pos-weight",
        action="store_true",
        help="Use positive-class weights derived from train_100 for all fractions.",
    )
    parser.add_argument("--device", default="auto")
    return TrainConfig(**vars(parser.parse_args()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(config: TrainConfig, device: torch.device) -> dict[str, object]:
    split_dir = Path(config.split_dir)
    data_root = Path(config.data_root)
    datasets = {
        "train": PTBXLDataset(
            split_dir / f"train_{config.train_fraction}.csv",
            data_root,
            normalize=config.normalize,
        ),
        "val": PTBXLDataset(
            split_dir / "val.csv", data_root, normalize=config.normalize
        ),
        "test": PTBXLDataset(
            split_dir / "test.csv", data_root, normalize=config.normalize
        ),
    }
    pin_memory = device.type == "cuda"
    return {
        name: make_data_loader(
            dataset,
            batch_size=config.batch_size,
            shuffle=name == "train",
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            seed=config.seed,
        )
        for name, dataset in datasets.items()
    }


def make_pos_weight(config: TrainConfig, device: torch.device) -> torch.Tensor | None:
    if not config.use_pos_weight:
        return None
    full_train = pd.read_csv(Path(config.split_dir) / "train_100.csv")
    positives = full_train[list(CLASS_NAMES)].sum(axis=0).to_numpy(np.float32)
    negatives = len(full_train) - positives
    return torch.as_tensor(negatives / np.maximum(positives, 1), device=device)


def train_one_epoch(
    model: nn.Module,
    loader: object,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        signals = batch["signal"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = signals.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / total_examples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: object,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets = []
    probabilities = []
    ecg_ids = []

    for batch in loader:
        signals = batch["signal"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(signals)
        loss = criterion(logits, labels)

        batch_size = signals.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        targets.append(labels.cpu().numpy())
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        ecg_ids.append(batch["ecg_id"].numpy())

    return (
        total_loss / total_examples,
        np.concatenate(targets),
        np.concatenate(probabilities),
        np.concatenate(ecg_ids),
    )


def save_predictions(
    path: Path,
    ecg_ids: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    result = {"ecg_id": ecg_ids}
    for index, class_name in enumerate(CLASS_NAMES):
        result[f"target_{class_name}"] = targets[:, index].astype(np.int8)
        result[f"probability_{class_name}"] = probabilities[:, index]
    pd.DataFrame(result).to_csv(path, index=False)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir) / (
        f"fraction_{config.train_fraction}_seed_{config.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n"
    )

    loaders = build_loaders(config, device)
    model = CNNBiLSTM(
        lstm_hidden_size=config.lstm_hidden_size,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=make_pos_weight(config, device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_path = output_dir / "best_model.pt"
    history = []
    best_val_auroc = -np.inf
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_targets, val_probabilities, _ = evaluate(
            model, loaders["val"], criterion, device
        )
        val_metrics = compute_metrics(val_targets, val_probabilities)
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        if val_metrics["macro_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["macro_auroc"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "class_names": list(CLASS_NAMES),
                    "epoch": epoch,
                    "val_macro_auroc": best_val_auroc,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_targets, val_probabilities, val_ids = evaluate(
        model, loaders["val"], criterion, device
    )
    thresholds = find_f1_thresholds(val_targets, val_probabilities)
    val_metrics = compute_metrics(
        val_targets, val_probabilities, thresholds=thresholds
    )

    test_loss, test_targets, test_probabilities, test_ids = evaluate(
        model, loaders["test"], criterion, device
    )
    test_metrics = compute_metrics(
        test_targets, test_probabilities, thresholds=thresholds
    )

    final_results = {
        "best_epoch": checkpoint["epoch"],
        "thresholds": dict(zip(CLASS_NAMES, thresholds.tolist())),
        "validation": {
            "loss": val_loss,
            **val_metrics,
            "per_class": compute_per_class_metrics(
                val_targets,
                val_probabilities,
                thresholds=thresholds,
                class_names=CLASS_NAMES,
            ),
        },
        "test": {
            "loss": test_loss,
            **test_metrics,
            "per_class": compute_per_class_metrics(
                test_targets,
                test_probabilities,
                thresholds=thresholds,
                class_names=CLASS_NAMES,
            ),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(final_results, indent=2) + "\n"
    )
    save_predictions(
        output_dir / "validation_predictions.csv",
        val_ids,
        val_targets,
        val_probabilities,
    )
    save_predictions(
        output_dir / "test_predictions.csv",
        test_ids,
        test_targets,
        test_probabilities,
    )
    print(json.dumps(final_results, indent=2))


if __name__ == "__main__":
    main()
