"""Run a PTB-XL Transformer experiment with repository-relative default paths.

Examples:
    python -m src.transformers.run
    python -m src.transformers.run --model v1
    python src/transformers/run.py --model v2 --train-percents 10 25 50
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

# Support direct execution from any cwd as well as `python -m` from the repo.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.transformers.training import TrainConfig, default_config, run_experiment


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("v1", "v2"), default="v2",
                        help="Version preset: v2 uses 10/25/50%%, 30 epochs; v1 uses 5%%, 5 epochs.")
    parser.add_argument("--train-percents", type=int, nargs="+", choices=(5, 10, 25, 50, 100), default=argparse.SUPPRESS)
    for flag in ("data-root", "split-dir", "output-dir"):
        parser.add_argument(f"--{flag}", default=argparse.SUPPRESS, help="Relative paths resolve from the repository root.")
    for flag in ("seed", "batch-size", "max-epochs", "patience", "d-model", "n-heads", "n-layers", "patch-size", "threads"):
        parser.add_argument(f"--{flag}", type=int, default=argparse.SUPPRESS)
    for flag in ("min-delta", "learning-rate", "min-learning-rate", "weight-decay", "dropout"):
        parser.add_argument(f"--{flag}", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--schedule", choices=("constant", "cosine"), default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS, help="auto, cpu, cuda, or cuda:0; auto prefers CUDA.")
    for flag in ("initialize-bias", "tune-thresholds", "run-test", "plots"):
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    values = vars(parser.parse_args(argv))
    preset = default_config(values.pop("model"))
    if "train_percents" in values:
        values["train_percents"] = tuple(values["train_percents"])
    try:
        return replace(preset, **values)
    except ValueError as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> None:
    result = run_experiment(parse_args(argv))
    if result.test_summary is not None:
        print("\nFinal test metrics")
        print(result.test_summary[["macro_AUROC", "macro_AUPRC", "macro_F1_at_0.5", "macro_F1_val_threshold"]].round(4).to_string())
    else:
        print("\nValidation metrics (test evaluation disabled)")
        print(result.training_summary[["val_macro_AUROC", "val_macro_AUPRC", "val_macro_F1_at_0.5", "val_macro_F1_val_threshold"]].round(4).to_string())
    print(f"\nSaved experiment: {result.run_dir}")


if __name__ == "__main__":
    main()
