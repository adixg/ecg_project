"""Reusable training, checkpoint loading, inference, and experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import (
    CLASSES, N_LEADS, SAMPLE_RATE, SIGNAL_LENGTH, PreparedData,
    prepare_data, project_path, resolve_data_root, save_json, training_statistics,
)
from .metrics import (
    choose_validation_thresholds, classification_metrics, save_predictions, summarize_metrics,
)
from .transformer_v1 import ECGTransformerV1
from .transformer_v2 import ECGTransformerV2


@dataclass(frozen=True)
class TrainConfig:
    """V2 defaults; use default_config('v1') for the original notebook setup."""

    model_version: str = "v2"
    train_percents: tuple[int, ...] = (10, 25, 50)
    data_root: str = field(default_factory=lambda: os.environ.get("PTBXL_ROOT", "data/ptbxl"))
    split_dir: str = "splits"
    output_dir: str = "artifacts/transformer_v2"
    seed: int = 42
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 8
    min_delta: float = 1e-4
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 1e-2
    schedule: str = "cosine"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    patch_size: int = 20
    initialize_bias: bool = True
    tune_thresholds: bool = True
    run_test: bool = True
    plots: bool = True
    device: str = "auto"
    threads: int = 4

    def __post_init__(self):
        if self.model_version not in ("v1", "v2"):
            raise ValueError("model_version must be v1 or v2")
        if not self.train_percents or tuple(sorted(set(self.train_percents))) != self.train_percents:
            raise ValueError("train_percents must be a nonempty, increasing tuple")
        if not set(self.train_percents) <= {5, 10, 25, 50, 100}:
            raise ValueError("Available training percentages: 5, 10, 25, 50, 100")
        if min(self.batch_size, self.max_epochs, self.threads, self.d_model, self.n_heads, self.n_layers) < 1:
            raise ValueError("Batch size, epochs, threads, and model dimensions must be positive")
        if self.d_model % self.n_heads or (self.model_version == "v2" and self.d_model % 2):
            raise ValueError("Embedding size must divide into heads; v2 also requires an even size")
        if self.patch_size < 1 or SIGNAL_LENGTH % self.patch_size:
            raise ValueError("patch_size must divide the 1000-sample signal")
        if not 0 <= self.dropout < 1 or self.patience < 0 or self.min_delta < 0:
            raise ValueError("Invalid dropout, patience, or minimum improvement")
        if self.learning_rate <= 0 or not 0 <= self.min_learning_rate <= self.learning_rate or self.weight_decay < 0:
            raise ValueError("Invalid learning rate or weight decay")
        if self.schedule not in ("constant", "cosine"):
            raise ValueError("schedule must be constant or cosine")

    @property
    def model_config(self) -> dict:
        architecture = {
            "d_model": self.d_model, "n_heads": self.n_heads,
            "n_layers": self.n_layers, "dropout": self.dropout,
        }
        if self.model_version == "v1":
            architecture.update(signal_length=SIGNAL_LENGTH, n_leads=N_LEADS,
                                n_classes=len(CLASSES), patch_size=self.patch_size)
        return architecture


def default_config(model_version: str = "v2") -> TrainConfig:
    """Return the version-specific defaults from the original notebooks."""
    if model_version == "v1":
        return replace(
            TrainConfig(), model_version="v1", train_percents=(5,),
            output_dir="artifacts/transformer_v1", batch_size=32, max_epochs=5,
            patience=0, schedule="constant", initialize_bias=False,
            tune_thresholds=False, run_test=False,
        )
    if model_version != "v2":
        raise ValueError("model_version must be v1 or v2")
    return TrainConfig()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def create_model(model_version: str, lead_mean, lead_std, **architecture) -> nn.Module:
    """Construct a model without loading data or starting an experiment."""
    constructors = {"v1": ECGTransformerV1, "v2": ECGTransformerV2}
    if model_version not in constructors:
        raise ValueError("model_version must be v1 or v2")
    return constructors[model_version](lead_mean, lead_std, **architecture)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load either notebook or runner checkpoints; return an eval model and metadata.

    Normalization comes from the checkpoint buffers. Legacy v1 checkpoints did
    not store thresholds, so their metadata defaults to the original fixed 0.5.
    """
    checkpoint = torch.load(project_path(path), map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    architecture = config["model"]
    version = config.get("model_version", "v1" if "patch_size" in architecture else "v2")
    state = checkpoint["model_state_dict"]
    model = create_model(version, state["lead_mean"].flatten(), state["lead_std"].flatten(), **architecture)
    model.load_state_dict(state)
    model.to(resolve_device(device)).eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    metadata["model_version"] = version
    metadata.setdefault("thresholds", [config.get("f1_threshold", 0.5)] * len(CLASSES))
    return model, metadata


@torch.inference_mode()
def predict_probabilities(model: nn.Module, signals, batch_size: int = 64) -> np.ndarray:
    """Predict from raw mV ECGs shaped (N, 1000, 12), in input order."""
    if batch_size < 1 or len(signals) == 0:
        raise ValueError("Provide a positive batch size and at least one ECG")
    device = next(model.parameters()).device
    model.eval()
    batches = []
    for start in range(0, len(signals), batch_size):
        if isinstance(signals, torch.Tensor):
            batch = signals[start:start + batch_size].detach().to(device=device, dtype=torch.float32)
        else:
            batch = torch.as_tensor(
                np.array(signals[start:start + batch_size], copy=True),
                dtype=torch.float32, device=device,
            )
        batches.append(model(batch).sigmoid().cpu().numpy())
    return np.concatenate(batches)


def train_one_epoch(model, loader, optimizer, device) -> float:
    """One unweighted BCE epoch with finite-loss checks and gradient clipping."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for signals, targets in loader:
        signals, targets = signals.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(signals), targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate without fitting; return BCE, targets, and probabilities in row order."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, targets_out, probabilities = 0.0, [], []
    for signals, targets in loader:
        signals, targets = signals.to(device), targets.to(device)
        logits = model(signals)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss")
        total_loss += loss.item() * len(targets)
        targets_out.append(targets.cpu().numpy())
        probabilities.append(logits.sigmoid().cpu().numpy())
    return total_loss / len(loader.dataset), np.concatenate(targets_out), np.concatenate(probabilities)


def fit_model(data: PreparedData, config: TrainConfig, percent: int,
              output_dir: Path, device: torch.device, run_config: dict) -> dict:
    """Train one fresh model, select by validation AUROC, and freeze thresholds."""
    seed_everything(config.seed)
    name = f"train_{percent}"
    output_dir.mkdir(exist_ok=False)
    mean, std = training_statistics(data.signals, data.indices[name])
    pd.DataFrame({"lead": data.leads, "mean": mean, "std": std}).to_csv(output_dir / "normalization.csv", index=False)
    subset_config = {**run_config, "train_percent": percent,
                     "training_recordings": len(data.frames[name]),
                     "training_patients": int(data.frames[name].patient_id.nunique())}
    save_json(output_dir / "config.json", subset_config)
    model = create_model(config.model_version, mean, std, **config.model_config).to(device)
    if config.initialize_bias:
        prevalence = torch.as_tensor(data.labels[name].mean(axis=0), device=device).clamp(1e-4, 1 - 1e-4)
        with torch.no_grad():
            model.classifier.bias.copy_(torch.logit(prevalence))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=config.min_learning_rate,
    ) if config.schedule == "cosine" else None)
    train_loader = data.loader(name, config.batch_size, config.seed, training=True, pin_memory=device.type == "cuda")
    val_loader = data.loader("val", config.batch_size, config.seed, pin_memory=device.type == "cuda")
    history, best_auc, patience_reference, stale = [], -np.inf, -np.inf, 0
    print(f"\nTraining {config.model_version}, {percent}%: {len(train_loader.dataset):,} ECGs", flush=True)
    for epoch in range(1, config.max_epochs + 1):
        started = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, targets, probabilities = evaluate(model, val_loader, device)
        table = classification_metrics(targets, probabilities)
        score = float(table.AUROC.mean())
        if not np.isfinite(score):
            raise RuntimeError("Validation macro AUROC is undefined")
        if score > best_auc:
            best_auc = score
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state_dict": state, "config": subset_config,
                        "epoch": epoch, "val_macro_AUROC": score}, output_dir / "best_model.pt")
            del state
        if score > patience_reference + config.min_delta:
            patience_reference, stale = score, 0
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step()
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_macro_AUROC": score, "val_macro_AUPRC": float(table.AUPRC.mean()),
            "val_macro_F1_at_0.5": float(table["F1_at_0.5"].mean()),
            "learning_rate": learning_rate, "seconds": time.perf_counter() - started,
        })
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(f"{percent}% epoch {epoch:02d}/{config.max_epochs}: train={train_loss:.4f} "
              f"val={val_loss:.4f} AUROC={score:.4f} AP={table.AUPRC.mean():.4f} "
              f"({history[-1]['seconds']:.1f}s)", flush=True)
        if config.patience and stale >= config.patience:
            print("Early stopping; restoring the best validation checkpoint.", flush=True)
            break
    checkpoint = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, targets, probabilities = evaluate(model, val_loader, device)
    thresholds = (choose_validation_thresholds(targets, probabilities) if config.tune_thresholds
                  else np.full(len(CLASSES), 0.5))
    table = classification_metrics(targets, probabilities, thresholds)
    if not np.isclose(table.AUROC.mean(), best_auc):
        raise RuntimeError("Restored checkpoint score differs from selection score")
    table.to_csv(output_dir / "validation_metrics.csv")
    save_predictions(output_dir / "validation_predictions.csv", data.frames["val"], targets, probabilities)
    save_json(output_dir / "thresholds.json", dict(zip(CLASSES, thresholds.tolist())))
    checkpoint["thresholds"] = thresholds.tolist()
    torch.save(checkpoint, output_dir / "best_model.pt")
    summary = {
        "train_percent": percent, "train_recordings": len(data.frames[name]),
        "best_epoch": checkpoint["epoch"], "epochs_run": len(history),
        "training_seconds": float(sum(row["seconds"] for row in history)),
        "validation_loss": val_loss,
        **{f"val_{key}": value for key, value in summarize_metrics(targets, probabilities, table, thresholds).items()},
    }
    save_json(output_dir / "training_summary.json", summary)
    return summary


@dataclass
class ExperimentResult:
    run_dir: Path
    training_summary: pd.DataFrame
    test_summary: pd.DataFrame | None
    classwise_results: pd.DataFrame | None


def run_experiment(config: TrainConfig | None = None) -> ExperimentResult:
    """Run all training subsets, freeze models, then optionally evaluate fold 10."""
    config = TrainConfig() if config is None else config
    device = resolve_device(config.device)
    torch.set_num_threads(config.threads)
    data_root = resolve_data_root(config.data_root)
    split_dir = project_path(config.split_dir)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_seed{config.seed}"
    run_dir = project_path(config.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Device: {device}; artifacts: {run_dir}", flush=True)
    data = prepare_data(data_root, split_dir, run_dir, config.train_percents, include_test=config.run_test)
    run_config = {
        **asdict(config), "model": config.model_config, "classes": CLASSES,
        "leads": data.leads, "sample_rate": SAMPLE_RATE, "signal_shape": [SIGNAL_LENGTH, N_LEADS],
        "resolved_data_root": str(data_root), "provenance": data.provenance,
        "torch_version": str(torch.__version__), "device": str(device),
        "selection_metric": "validation macro AUROC",
        "f1_threshold_selection": "maximum per-class validation F1" if config.tune_thresholds else "fixed 0.5",
        "auprc_definition": "sklearn average_precision_score",
    }
    save_json(run_dir / "config.json", run_config)
    summaries = []
    for percent in config.train_percents:
        summaries.append(fit_model(data, config, percent, run_dir / f"train_{percent}", device, run_config))
        pd.DataFrame(summaries).to_csv(run_dir / "training_summary.csv", index=False)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    training = pd.DataFrame(summaries).set_index("train_percent")
    test, classwise = None, None
    print("All checkpoints and thresholds frozen.", flush=True)
    if config.run_test:
        loader = data.loader("test", config.batch_size, config.seed, pin_memory=device.type == "cuda")
        results, tables = [], []
        for percent in config.train_percents:
            directory = run_dir / f"train_{percent}"
            model, checkpoint = load_checkpoint(directory / "best_model.pt", str(device))
            loss, targets, probabilities = evaluate(model, loader, device)
            if not np.array_equal(targets, data.labels["test"]):
                raise RuntimeError("Test row order changed")
            thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float64)
            table = classification_metrics(targets, probabilities, thresholds)
            table.to_csv(directory / "test_metrics.csv")
            save_predictions(directory / "test_predictions.csv", data.frames["test"], targets, probabilities)
            summary = {
                "train_percent": percent, "train_recordings": len(data.frames[f"train_{percent}"]),
                "best_epoch": int(training.loc[percent, "best_epoch"]),
                "epochs_run": int(training.loc[percent, "epochs_run"]), "test_loss": loss,
                **summarize_metrics(targets, probabilities, table, thresholds),
            }
            save_json(directory / "test_summary.json", summary)
            results.append(summary)
            tables.append(table.reset_index().assign(train_percent=percent))
            del model
        test = pd.DataFrame(results).set_index("train_percent")
        classwise = pd.concat(tables, ignore_index=True).set_index(["train_percent", "class"])
        test.to_csv(run_dir / "test_summary.csv")
        classwise.to_csv(run_dir / "test_classwise.csv")
    result = ExperimentResult(run_dir, training, test, classwise)
    if config.plots:
        from .plotting import save_figures
        save_figures(result)
    return result
