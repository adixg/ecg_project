# PTB-XL CNN-BiLSTM

## 1. Overview

This package trains a compact multi-label CNN-BiLSTM on the 100 Hz PTB-XL
signals. Three one-dimensional convolution blocks reduce each 1,000-sample ECG
to 125 time steps. A single bidirectional LSTM, temporal mean pooling, dropout,
and a linear head produce logits for the five diagnostic superclasses.

## 2. Files

| File | Purpose |
|---|---|
| `__init__.py` | Exposes `CNNBiLSTM` without importing PyTorch during package discovery |
| `data.py` | Loads PTB-XL waveforms and builds reproducibly seeded data loaders |
| `model.py` | Defines the convolution blocks and `CNNBiLSTM` model |
| `run.py` | Main training, checkpoint-selection, evaluation, and prediction entry point |
| `README.md` | Setup, commands, inputs, outputs, and conventions |
| `scratch.md` | Summary of all 21 verified runs and reproducibility notes |

The approach also uses the shared `src/metrics.py` implementation.

## 3. Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No CNN-only dependency is needed beyond the root requirements. CUDA is optional;
the runner also supports Apple MPS and CPU.

The recorded 21-run experiment used PyTorch 2.14.1+cu126 and an NVIDIA A40 on a
Georgia Tech PACE compute node.

## 4. Data and preprocessing

- Put PTB-XL under `data/ptbxl`, or set `PTBXL_ROOT` to the directory containing
  `records100/`.
- Each input is a 10-second, 100 Hz, 12-lead ECG with shape `(12, 1000)` after
  loading and transposition.
- Labels are multi-hot values for `NORM`, `MI`, `STTC`, `CD`, and `HYP`.
- Patient-level split definitions live in
  `splits/seed_<seed>/{train_<fraction>,val,test}.csv`. Available seeds are 0, 1,
  and 42; available training percentages are 1, 3, 5, 10, 25, 50, and 100.
  Training uses folds 1–8, validation is fold 9 (2,146 ECGs), and test is fold
  10 (2,158 ECGs). Validation, test, and `train_100` are identical across the
  three seeds; only the smaller training subsets differ.
- By default the WFDB physical-unit signal is used without per-record
  normalization. `--normalize` applies a per-lead, per-record z-score.
- Class weighting is off by default. `--use-pos-weight` derives one positive
  weight per class from the selected seed's `train_100.csv`, so it remains fixed
  across fractions.

## 5. Commands

All commands run from the repository root.

```bash
# Show every option without starting a run.
python -m src.cnn.run --help

# Train on the seed-42 100% split, select a checkpoint on validation AUROC,
# evaluate validation and test, and save both prediction files.
python -m src.cnn.run --seed 42 --train-fraction 100

# CUDA example with data stored elsewhere.
PTBXL_ROOT=/path/to/ptbxl python -m src.cnn.run \
  --seed 42 --train-fraction 25 --device cuda

# One-epoch smoke run. This still evaluates the full validation and test sets.
python -m src.cnn.run --seed 42 --train-fraction 1 --epochs 1 \
  --batch-size 32 --num-workers 0 \
  --artifacts-dir artifacts/cnn_smoke

# Complete recorded matrix: 3 seeds x 7 fractions = 21 sequential runs.
for seed in 0 1 42; do
  for fraction in 1 3 5 10 25 50 100; do
    python -m src.cnn.run \
      --seed "$seed" --train-fraction "$fraction" --device cuda \
      --artifacts-dir artifacts/cnn_restructured
  done
done
```

Training, evaluation, and labeled prediction are deliberately one reproducible
command: after early stopping, `run.py` reloads the validation-selected
checkpoint, tunes thresholds on validation only, evaluates test once, and writes
validation and test probabilities. There is no separate unlabeled-inference CLI
or result-table generator; the verified aggregate tables are in `scratch.md`.

## 6. CLI flags

Main entry point: `python -m src.cnn.run`.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--data-root` | path | `PTBXL_ROOT` or `data/ptbxl` | Directory containing `records100/` |
| `--splits-dir` | path | `splits` | Base directory containing `seed_<seed>/`; `--split-dir` is a compatibility alias |
| `--artifacts-dir` | path | `artifacts/cnn` | Root for generated runs; `--output-dir` is a compatibility alias |
| `--train-fraction` | choice | `100` | One of 1, 3, 5, 10, 25, 50, or 100 percent |
| `--seed` | choice | `42` | One of 0, 1, or 42; selects `seed_<seed>` and seeds Python, NumPy, PyTorch, and the data loader |
| `--epochs` | integer | `50` | Maximum training epochs |
| `--batch-size` | integer | `64` | Batch size for all splits |
| `--learning-rate` | float | `0.001` | AdamW learning rate |
| `--weight-decay` | float | `0.0001` | AdamW weight decay |
| `--dropout` | float | `0.3` | Dropout before the classifier |
| `--lstm-hidden-size` | integer | `64` | Hidden units in each LSTM direction |
| `--patience` | integer | `8` | Epochs without validation-AUROC improvement before stopping |
| `--num-workers` | integer | `4` | Data-loader worker processes |
| `--normalize` | flag | off | Apply per-record, per-lead z-score normalization |
| `--use-pos-weight` | flag | off | Use full-training-set positive-class weights in BCE loss |
| `--device` | string | `auto` | `auto`, `cpu`, `mps`, `cuda`, or a device such as `cuda:0` |

Relative paths resolve from the repository root, independent of the caller's
working directory.

## 7. Outputs

Generated files are ignored by Git and stored under:

```text
artifacts/cnn/cnn_bilstm_seed<seed>_fraction<fraction>/
├── config.json
├── epoch_metrics.json
├── validation_metrics.json
├── test_metrics.json
├── thresholds.json
├── best_model.pt
├── validation_predictions.csv
└── test_predictions.csv
```

`artifacts/cnn` is the CLI default. The completed three-seed experiment described
in `scratch.md` was deliberately written to `artifacts/cnn_restructured` so it
would remain separate from smoke tests and legacy runs.

`config.json` includes resolved paths, resolved device, and elapsed training
seconds. `epoch_metrics.json` contains training loss and validation metrics for
each epoch. During training, a progress bar identifies the active seed and
fraction and displays the latest training loss, validation loss, and validation
macro AUROC. The metric files contain macro and classwise AUROC, AUPRC, and F1.
Prediction CSVs contain the ECG ID, targets, and probabilities for each class.
Rerunning the same seed and fraction writes to the same run directory and
replaces files from that run.

Runs produced before the restructure may retain the legacy names `history.json`
and `metrics.json`; they remain locally archived below `artifacts/cnn/`.

## 8. Selection and metric conventions

- Loss is `BCEWithLogitsLoss`, optionally with full-training-set positive weights.
- Optimization uses AdamW.
- The best checkpoint is the epoch with the highest validation macro AUROC.
- Early stopping occurs after `--patience` consecutive epochs without a higher
  validation macro AUROC.
- One decision threshold per class is selected on validation by maximizing F1
  over 0.05 through 0.95 in steps of 0.05.
- The selected checkpoint and frozen validation thresholds are then used for the
  test set. Test results never select a checkpoint or threshold.
- Macro and per-class AUROC, AUPRC (average precision), and F1 are computed by
  the shared `src/metrics.py`, the same implementation used by classical ML.

## 9. Limitations and reproducibility

- The reference results cover seeds 0, 1, and 42. Three seeds provide a useful
  descriptive spread but are too few for precise confidence intervals.
- Below 100%, seed-to-seed spread combines patient-subset variation with model
  initialization and optimization variation. At 100%, the patient subset is
  identical across seeds, so the spread is primarily optimization variation.
- CUDA kernels and platform/library changes can still introduce small numerical
  differences despite deterministic seeds and deterministic cuDNN settings.
- The baseline does not use a learning-rate schedule, augmentation, calibration,
  or lead-quality filtering.
- Validation data controls checkpoint selection, early stopping, and thresholds;
  validation scores are therefore optimistic relative to test scores.
- The approach currently combines training, evaluation, and prediction in one
  command and does not provide an unlabeled deployment-inference interface.
