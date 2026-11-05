# PTB-XL Classical ML

Random forest, XGBoost, and RBF-SVM baselines on hand-built wavelet features, for
the same five-superclass task and the same splits as the CNN-BiLSTM and the
Transformers. They give the "no deep learning" end of the learning curve.

## Files

| File | Purpose |
|---|---|
| `features.py` | Extracts a 300-d wavelet feature vector per ECG and saves `features/features.npz` |
| `classical.py` | Tunes and evaluates RF / XGBoost / SVM per training fraction and writes the reports |
| `scratch.md` | Recorded results and notes from the completed seed-42 runs |

Both scripts run from the repository root as modules, for example
`python -m src.classical_ml.classical`. Relative paths resolve from the repository
root, so the working directory does not matter.

## Pipeline

**1. Features** (`features.py`). Each 10 s, 100 Hz, 12-lead record is band-pass
filtered (3rd-order Butterworth, 0.5–40 Hz, zero-phase), then z-scored per lead
within the record. Each lead gets a 4-level `db4` wavelet decomposition, giving 5
bands (cA4, cD4, cD3, cD2, cD1). Each band is summarised by 5 statistics: energy,
standard deviation, mean absolute value, Shannon entropy of the normalised
energy, and kurtosis. That is 12 × 5 × 5 = 300 features. Features are computed
once for `train_100`, `val`, and `test` (21,388 records); smaller training
fractions reuse the same rows, so the features never depend on the fraction.

```bash
python -m src.classical_ml.features        # -> features/features.npz (git-ignored)
```

**2. Models** (`classical.py`). The task is multi-label, so each model is trained
one-vs-rest: five independent binary classifiers, one per class.

| Model | Class imbalance | Score used for ranking |
|---|---|---|
| `rf` | `class_weight="balanced"` | `predict_proba` |
| `xgb` | `scale_pos_weight = negatives / positives` | `predict_proba` |
| `svm` | `class_weight="balanced"`, features standardised, RBF kernel | `sigmoid(decision_function)` |

SVMs have no native probability. The sigmoid of the margin distance is monotone,
so AUROC and AUPRC are identical to those of the raw margin, but the value is
**not calibrated**. Only the validation-tuned F1 threshold gives it meaning.

**3. Tuning.** Each model has a small grid (see `GRIDS` in `classical.py`):
RF 4 settings, XGBoost 8, SVM 6. Every setting is fitted on the training
fraction only and scored by **validation macro AUROC**; the best setting is kept.
Validation is never trained on, and the test set is touched only for the chosen
setting's final predictions. Because the grid is searched afresh for each
fraction, the chosen hyperparameters can differ across fractions.

**4. Metrics.** Metrics and thresholds come from `src/metrics.py`, the same code
the CNN-BiLSTM uses: one F1 threshold per class, chosen on validation from the
0.05–0.95 grid in steps of 0.05, then frozen for test. Macro AUROC, macro AUPRC
(average precision), and macro F1 are reported, plus per-class values.

## Run

```bash
# Everything: 3 models x 5 fractions, seed 42 (the SVM at 100% is the slow part).
python -m src.classical_ml.classical

# A quick check on the smallest cut.
python -m src.classical_ml.classical --fractions 5 --models rf

python -m src.classical_ml.classical --help
```

Options: `--fractions` (5 10 25 50 100), `--models` (rf xgb svm), `--seeds`
(default 42), and `--features`, `--splits-dir`, `--reports-dir`, `--outputs-dir`.

## Outputs

The layout mirrors `reports/cnn_bilstm/`:

```text
reports/<rf|xgb|svm>/seed_42/
  summary.csv                    one row per fraction
  fraction_<N>/config.json       model, grid, scoring, training size, run time
  fraction_<N>/metrics.json      best_params, thresholds, validation and test
                                 macro + per-class AUROC / AUPRC / F1
outputs/<model>/fraction_<N>_seed_42/{validation,test}_predictions.csv   (git-ignored)
```

`summary.csv` has the CNN's columns, minus `best_epoch`. The JSON files carry
`best_params` instead of `best_epoch` and have no `loss`, because these models have
no epoch loop or loss curve, so there is no `history.json`.

## Notes and limits

- Single seed (42). RF and XGBoost depend on the seed; the SVM is deterministic.
  No variability or confidence intervals were measured.
- Inputs are normalised differently from the deep models: classical features use
  a per-record z-score, the CNN-BiLSTM uses raw mV, and the Transformers use
  train-subset per-lead statistics. This is kept as is.
- The classical models are tuned on validation with a grid search, whereas the
  deep models use fixed hyperparameters and early stopping on validation.
  Validation scores are therefore optimistic for both and test scores are the
  ones to compare.
- `import torch` before `import xgboost` crashes XGBoost on macOS (an OpenMP
  clash). `src/__init__.py` therefore imports the CNN lazily; keep it that way.
