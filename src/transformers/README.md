# PTB-XL Transformers

Importable versions of the two notebook models, plus a separate training runner.
Inputs are raw mV recordings shaped `(batch, 1000, 12)` at 100 Hz. Outputs are
five logits in the order `NORM`, `MI`, `STTC`, `CD`, `HYP`; apply sigmoid for
probabilities. Both models store training-set normalization in checkpoint buffers.

## Files

| File | Purpose |
|---|---|
| `transformer_v1.py` | `ECGTransformerV1`: linear temporal patches and learned positions |
| `transformer_v2.py` | `ECGTransformerV2`: convolutional tokens and sinusoidal positions |
| `run.py` | CLI argument parsing and the main entry point |
| `training.py` | Configurations, training, checkpoint loading, and inference |
| `data.py` | Split auditing, WFDB loading, memory-mapped datasets, normalization |
| `metrics.py` | Classwise AUROC, AUPRC, F1, and validation threshold selection |
| `plotting.py` | Training and test learning curves saved as PNGs |
| `scratch.md` | Verified two-model, three-seed, seven-fraction benchmark summary |

Importing the package does not start training or read the dataset. The original
notebooks remain under `notebooks/` as executed experiment records. This package
does not require Jupyter or the removed notebook execution script.

## Run

From the repository root, using the project Python environment:

```bash
# Default v2 experiment: independent models on 10%, 25%, and 50% training subsets.
.venv/bin/python -m src.transformers.run

# Original v1 preset: 5% training, five epochs, validation only.
.venv/bin/python -m src.transformers.run --model v1

# Choose subsets, budget, and device explicitly.
.venv/bin/python -m src.transformers.run --model v2 \
  --train-percents 10 25 50 --max-epochs 30 --device cpu

# Run a short training check without evaluating the test set.
.venv/bin/python -m src.transformers.run --model v2 \
  --train-percents 5 --max-epochs 1 --no-run-test --no-plots

.venv/bin/python -m src.transformers.run --help
```

For the complete architecture comparison, run both models with the same training
and evaluation policy over all 3 seeds and all 7 nested training fractions. This
trains 42 fresh models; validation selects checkpoints and thresholds before test
evaluation begins.

```bash
for model in v1 v2; do
  for seed in 0 1 42; do
    .venv/bin/python -m src.transformers.run \
      --model "$model" --seed "$seed" \
      --train-percents 1 3 5 10 25 50 100 \
      --max-epochs 30 --batch-size 64 --patience 8 \
      --schedule cosine --initialize-bias --tune-thresholds --run-test \
      --device cuda --output-dir "artifacts/transformers/$model"
  done
done
```

The explicit policy overrides are important: the original V1 and V2 presets
have different budgets and evaluation behavior. Use the loop above for a direct
architecture comparison; use the shorter preset commands to reproduce the
historical notebook protocols.

Direct execution also works:

```bash
.venv/bin/python src/transformers/run.py --model v2
```

When using the absolute path to `run.py`, direct execution works from any working
directory. Relative **data, split, and output arguments always resolve from the
repository root**, rather than from the shell's current directory.

## Defaults

| Setting | v1 | v2 |
|---|---|---|
| Training percentages | `5` | `10 25 50` |
| Epoch limit | 5 | 30 |
| Batch size | 32 | 64 |
| Early-stopping patience | Disabled (`0`) | 8 |
| Learning-rate schedule | Constant | Cosine, minimum `3e-5` |
| Classifier bias initialization | Standard PyTorch initialization | Training class log-odds |
| F1 thresholds | Fixed 0.5 | Per-class maximum validation F1 |
| Final test evaluation | Disabled | Enabled |
| Output base | `artifacts/transformers/v1` | `artifacts/transformers/v2` |

Shared defaults are seed 42, AdamW with learning rate `3e-4` and weight decay
`1e-2`, unweighted binary cross-entropy, gradient clipping at 1.0, 128-dimensional
embeddings, four attention heads, three pre-norm encoder layers, a 512-wide
feed-forward block, and dropout 0.1. The device is CUDA when available and CPU
otherwise, with four CPU threads. V1 uses 20-sample nonoverlapping patches; v2
uses 50 overlapping convolutional tokens with a 31-sample receptive field.

Data defaults to `data/ptbxl`, or the value of `PTBXL_ROOT` when set. Pass
`--data-root` to override it. The resolver accepts either a folder containing
`records100/` or a folder with one extracted PTB-XL subdirectory. Splits default
to `splits/`. For seed `S`, the runner automatically selects
`splits/seed_<S>/`; passing that seed directory explicitly is also supported.
Available seeds are 0, 1, and 42, and available training percentages are 1, 3,
5, 10, 25, 50, and 100. A legacy flat split directory containing the CSV files
is still accepted.

Use `--run-test` to enable v1 test evaluation, `--no-run-test` to disable it for
v2, and `--tune-thresholds` / `--no-tune-thresholds` to override threshold policy.
`--patience 0` disables early stopping. `--patch-size` controls the v1 linear
embedding; v2 has a fixed convolutional stem. All defaults are exposed in
`TrainConfig` and `default_config()`; CLI arguments override the selected preset.

## CLI flags

Main entry point: `python -m src.transformers.run`. Relative paths resolve from
the repository root.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--model` | `v1` or `v2` | `v2` | Select the architecture and its historical preset |
| `--train-percents` | one or more choices | V1: `5`; V2: `10 25 50` | Fractions chosen from `1 3 5 10 25 50 100` |
| `--data-root` | path | `PTBXL_ROOT` or `data/ptbxl` | Dataset directory containing `records100/` |
| `--split-dir` | path | `splits` | Seeded split root or an explicit seed/legacy flat directory |
| `--output-dir` | path | `artifacts/transformers/<model>` | Base for timestamped runs |
| `--seed` | choice | `42` | Model/data-loader seed and split selector: `0`, `1`, or `42` |
| `--batch-size` | integer | V1: `32`; V2: `64` | Training, validation, and test batch size |
| `--max-epochs` | integer | V1: `5`; V2: `30` | Maximum epochs for each fraction |
| `--patience` | nonnegative integer | V1: `0`; V2: `8` | Stale epochs before stopping; `0` disables early stopping |
| `--min-delta` | float | `0.0001` | Validation-AUROC improvement needed to reset patience |
| `--learning-rate` | float | `0.0003` | Initial AdamW learning rate |
| `--min-learning-rate` | float | `0.00003` | Final cosine-schedule learning rate |
| `--weight-decay` | float | `0.01` | AdamW weight decay |
| `--schedule` | `constant` or `cosine` | V1: `constant`; V2: `cosine` | Learning-rate schedule |
| `--d-model` | integer | `128` | Token embedding width |
| `--n-heads` | integer | `4` | Attention heads |
| `--n-layers` | integer | `3` | Transformer encoder layers |
| `--dropout` | float | `0.1` | Encoder and classifier dropout |
| `--patch-size` | integer | `20` | V1 patch length; must divide 1,000; ignored by V2 |
| `--threads` | positive integer | `4` | PyTorch CPU threads |
| `--device` | string | `auto` | `auto`, `cpu`, `cuda`, or a device such as `cuda:0` |
| `--[no-]initialize-bias` | boolean | V1: off; V2: on | Initialize classifier bias from training prevalence |
| `--[no-]tune-thresholds` | boolean | V1: off; V2: on | Select per-class F1 thresholds on validation |
| `--[no-]run-test` | boolean | V1: off; V2: on | Evaluate the frozen checkpoint and thresholds on test |
| `--[no-]plots` | boolean | on | Save training and learning-curve figures |
| `-h`, `--help` | flag | off | Show CLI help without running an experiment |

## Python API

```python
from dataclasses import replace
from src.transformers import default_config, run_experiment

config = replace(
    default_config("v2"),
    train_percents=(10,),
    max_epochs=5,
    run_test=False,
)
result = run_experiment(config)
print(result.run_dir)
print(result.training_summary)
```

To load either an original notebook checkpoint or a new runner checkpoint:

```python
import numpy as np
from src.transformers import load_checkpoint, predict_probabilities

model, metadata = load_checkpoint(
    "artifacts/transformer_v2/20261027T020054956545Z_seed42/train_50/best_model.pt",
    device="cpu",
)
# raw_signals is a float32 NumPy array of raw mV ECGs, shaped (N, 1000, 12).
probabilities = predict_probabilities(model, raw_signals, batch_size=64)
predictions = probabilities >= np.asarray(metadata["thresholds"])[None, :]
```

`ECGTransformerV1`, `ECGTransformerV2`, `create_model`, `prepare_data`,
`training_statistics`, `train_one_epoch`, `evaluate`, `fit_model`,
`classification_metrics`, and `choose_validation_thresholds` are also public.
Model constructors take the training per-lead mean and standard deviation;
`load_checkpoint` restores those statistics automatically. Checkpoint state keys
match the original notebook implementations.

## Selection, metrics, and saved files

CSV targets are used unchanged. The runner checks official folds, patient
separation, nested training subsets, record identity, signal shape/rate, lead
order, and finite values. Differences from local metadata are written to
`label_audit.csv`. The known unique differences are validation ECG 3799 (MI=0 in
the CSV versus MI=1 under code-presence mapping) and training ECG 9822 (CD=0 in
the CSV versus CD=1); ECG 9822 appears in `train_100` and may also appear in a
smaller seeded subset. Normalization is fitted separately to each requested
training subset.

Each model starts from a fresh seed. The highest validation macro AUROC selects
the checkpoint; for early stopping, an improvement must exceed `min_delta=1e-4`
to reset patience. When threshold tuning is enabled, the selected checkpoint's
validation predictions determine class thresholds. All requested models and
thresholds are frozen before any test inference. Larger subsets receive more
optimizer steps under the same epoch limit.

AUROC and AUPRC use continuous probabilities. AUPRC is scikit-learn average
precision, rather than trapezoidal integration. Reports include classwise and
macro/micro scores, with F1 at both 0.5 and the frozen thresholds. Undefined
class AUROC/AP is represented as NaN; undefined F1 is zero. Prediction CSVs retain
full-precision model probabilities.

Every invocation creates a timestamped run directory. It contains configuration,
source hashes, split and label audits, the signal cache, normalization statistics,
epoch histories, best checkpoints, frozen thresholds, validation predictions and
metrics, optional test predictions and metrics, aggregate CSVs, and PNG curves.
The large cache and weights stay in the ignored `artifacts/` directory;
`scratch.md` preserves the verified aggregate, individual-run, and classwise
results alongside the source code.

The current benchmark covers both architectures, seeds 0, 1, and 42, and all
seven training fractions under a shared protocol. See `scratch.md` for artifact
paths, selected epochs, validation/test metrics, thresholds, timing, and
reproducibility limitations.
