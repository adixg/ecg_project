"""PTB-XL Transformer models and reusable experiment functions.

Importing this package does not read recordings, create output directories, or
start training. Use run_experiment() explicitly or the separate run.py CLI.
"""

from .data import (
    CLASSES, prepare_data, resolve_data_root, resolve_split_dir,
    training_statistics,
)
from .metrics import choose_validation_thresholds, classification_metrics
from .transformer_v1 import ECGTransformerV1
from .transformer_v2 import ECGTransformerV2
from .training import (
    ExperimentResult, TrainConfig, create_model, default_config, evaluate,
    fit_model, load_checkpoint, predict_probabilities, run_experiment, train_one_epoch,
)

__all__ = [
    "CLASSES", "ECGTransformerV1", "ECGTransformerV2", "ExperimentResult", "TrainConfig",
    "choose_validation_thresholds", "classification_metrics", "create_model", "default_config",
    "evaluate", "fit_model", "load_checkpoint", "predict_probabilities", "prepare_data",
    "resolve_data_root", "resolve_split_dir", "run_experiment", "train_one_epoch",
    "training_statistics",
]
