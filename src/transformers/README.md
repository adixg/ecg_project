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
| `scratch.md` | Recorded metadata and best metrics from the completed notebook runs |

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
| Output base | `artifacts/transformer_v1` | `artifacts/transformer_v2` |

Shared defaults are seed 42, AdamW with learning rate `3e-4` and weight decay
`1e-2`, unweighted binary cross-entropy, gradient clipping at 1.0, 128-dimensional
embeddings, four attention heads, three pre-norm encoder layers, a 512-wide
feed-forward block, and dropout 0.1. The device is CUDA when available and CPU
otherwise, with four CPU threads. V1 uses 20-sample nonoverlapping patches; v2
uses 50 overlapping convolutional tokens with a 31-sample receptive field.

Data defaults to `data/ptbxl`, or the value of `PTBXL_ROOT` when set. Pass
`--data-root` to override it. The resolver accepts either a folder containing
`records100/` or a folder with one extracted PTB-XL subdirectory. Splits default
to `splits/`. Available training percentages are 5, 10, 25, 50, and 100.

Use `--run-test` to enable v1 test evaluation, `--no-run-test` to disable it for
v2, and `--tune-thresholds` / `--no-tune-thresholds` to override threshold policy.
`--patience 0` disables early stopping. `--patch-size` controls the v1 linear
embedding; v2 has a fixed convolutional stem. All defaults are exposed in
`TrainConfig` and `default_config()`; CLI arguments override the selected preset.

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
`label_audit.csv`; the known difference is validation ECG 3799, MI=0 in the CSV
versus MI=1 under code-presence mapping. Normalization is fitted separately to
each requested training subset.

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
`scratch.md` preserves the historical best metrics alongside the source code.

The historical results are single-seed experiments and do not establish
superiority over a CNN baseline. See `scratch.md` for the original budgets,
selected epochs, and measured outcomes.
