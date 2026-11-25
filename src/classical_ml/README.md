# PTB-XL Classical ML

## 1. Overview

Random forest, XGBoost, and RBF-SVM baselines on hand-built wavelet features, for
the same five-superclass task and split folds as the CNN-BiLSTM and the
Transformers. They give the "no deep learning" end of the learning curve.

The task is multi-label, so each model is trained one-vs-rest: five independent
binary classifiers, one per class.

| Model | Library | Class imbalance | Score used for ranking |
|---|---|---|---|
| `rf` | scikit-learn `RandomForestClassifier` | `class_weight="balanced"` | `predict_proba` |
| `xgb` | `XGBClassifier` | `scale_pos_weight = negatives / positives` | `predict_proba` |
| `svm` | scikit-learn `SVC`, RBF kernel, standardised features | `class_weight="balanced"` | `sigmoid(decision_function)` |

## 2. Files

| File | Purpose |
|---|---|
| `__init__.py` | Marks the package |
| `features.py` | Extracts a 300-d wavelet feature vector per ECG |
| `run.py` | Main entry point: tunes and evaluates each model, seed, and training fraction |
| `summarize.py` | Prints markdown result tables from the run directories (read-only) |
| `README.md` | This guide |
| `scratch.md` | Summary of the verified runs |

`run.py` also uses the shared `src/metrics.py`.

## 3. Environment

From the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No dependencies beyond the root `requirements.txt`. CPU only. The recorded runs
used Python 3.14.6, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, xgboost 3.4.1,
scipy 1.18.1, PyWavelets 1.8.0, wfdb 4.3.1, and joblib 1.6.0.

`import torch` before `import xgboost` crashes XGBoost on macOS (an OpenMP
clash). `src/__init__.py` therefore imports the CNN lazily; keep it that way.

## 4. Data

- **Location:** PTB-XL `records100/` under `data/ptbxl`, or set `PTBXL_ROOT` to the
  directory that contains `records100/`. Only `features.py` reads the signals.
- **Input:** 10 s, 100 Hz, 12-lead records, shape (1000, 12).
- **Labels:** five superclasses, multi-label: NORM, MI, STTC, CD, HYP.
- **Splits:** `splits/seed_<S>/{train_<N>,val,test}.csv`, patient-level. Train is
  from folds 1–8, validation is fold 9 (2,146 ECGs), test is fold 10 (2,158 ECGs).
  Available seeds are 0, 1, and 42, with N in 1, 3, 5, 10, 25, 50, 100.
  `val`, `test`, and `train_100` are identical across the three seeds; the seed
  changes only the smaller training subsets.
- **Preprocessing:** 3rd-order Butterworth band-pass (0.5–40 Hz, zero-phase), then
  a per-record, per-lead z-score. Each lead gets a 4-level `db4` wavelet
  decomposition (cA4, cD4, cD3, cD2, cD1). Each band is summarised by energy,
  standard deviation, mean absolute value, Shannon entropy of the normalised
  energy, and kurtosis: 12 × 5 × 5 = 300 features.
- Features are computed once from `splits/seed_42` for `train_100`, `val`, and
  `test` (21,388 records). Every training subset reuses those rows.

## 5. Commands

All commands run from the repository root.

```bash
# Features (needed once, before any training).
python -m src.classical_ml.features

# Smoke test (a few seconds). Writes to a separate directory.
python -m src.classical_ml.run --models rf --fractions 1 --seeds 42 \
  --artifacts-dir artifacts/classical_ml_smoke

# All recorded runs: 3 models x 3 seeds x 7 fractions.
python -m src.classical_ml.run --seeds 0 1 42 --fractions 1 3 5 10 25 50 100

# Result tables from the stored runs.
python -m src.classical_ml.summarize
```

Training and evaluation happen in the same command: each run scores validation and
test and saves the predictions. There is no separate prediction command, and
fitted models are not saved.

The smoke test should report a test AUROC of 0.7484.

## 6. CLI flags

Main entry point: `python -m src.classical_ml.run`. Relative paths resolve from the
repository root.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--fractions` | one or more int | `5 10 25 50 100` | Training percentages; each needs `train_<N>.csv`. Pass `1 3` explicitly |
| `--seeds` | one or more int | `42` | Split seeds; selects `seed_<S>/` and sets the RF and XGBoost `random_state` |
| `--models` | one or more of `rf`, `xgb`, `svm` | all three | Models to run |
| `--splits-dir` | path | `splits` | Base directory holding the `seed_<S>/` folders |
| `--features` | path | `artifacts/classical_ml/features/features.npz` | Feature file to load |
| `--artifacts-dir` | path | `artifacts/classical_ml` | Root for the run directories |
| `-h`, `--help` | flag | off | Show help |

`summarize.py` takes `--artifacts-dir` (path, default `artifacts/classical_ml`).
`features.py` takes only `--help`; it reads `PTBXL_ROOT` from the environment.

## 7. Outputs

`artifacts/` is git-ignored.

```text
artifacts/classical_ml/
  features/features.npz               ecg_id and the 300-d feature matrix X
  <model>_seed<S>_frac<N>/
    config.json                       inputs, grid, best_params, n_train, run time
    grid_results.json                 validation macro AUROC of every grid setting
    thresholds.json                   per-class F1 thresholds
    validation_metrics.json           macro and per-class AUROC / AUPRC / F1
    test_metrics.json                 same, for test
    validation_predictions.csv        ecg_id, targets, probabilities
    test_predictions.csv
```

Running a model, seed, and fraction again overwrites that run directory.

There is no `epoch_metrics.json`, loss, or checkpoint, because these models have no
epoch loop. Runs migrated from the earlier `reports/` layout carry a
`migrated_from` field in `config.json` and have no `grid_results.json`.

## 8. Selection and metric conventions

- **Model selection:** each model has a small grid (`GRIDS` in `run.py`): RF 4
  settings, XGBoost 8, SVM 6. Every setting is fitted on the training subset and
  scored by **validation macro AUROC**; the best is kept. The grid is searched
  again for each seed and fraction. Test data takes no part in selection.
- **Early stopping:** none.
- **Thresholds:** one F1 threshold per class, chosen on validation from the
  0.05–0.95 grid in steps of 0.05, then frozen for test.
- **Metrics:** macro AUROC, macro AUPRC (average precision), and macro F1, plus
  per-class values, all from `src/metrics.py`, the same code the CNN-BiLSTM uses.

## 9. Limitations and reproducibility

- Three seeds. The spread at 100% is not training-sample variance, because
  `train_100` is the same for every seed. XGBoost uses no row or column
  subsampling, so its 100% runs are identical; the SVM is deterministic; only RF
  varies there.
- RF scores can differ between repeat runs at floating-point precision (about
  1e-16), because trees are summed in parallel.
- SVM scores are **not calibrated** probabilities. The sigmoid is monotone, so
  AUROC and AUPRC equal those of the raw margin.
- The classical models are grid-searched on validation, while the deep models use
  fixed hyperparameters with early stopping. Validation scores are optimistic;
  compare test scores.
- Inputs are normalised differently across families: per-record z-score here, raw
  mV for the CNN-BiLSTM, train-subset per-lead statistics for the Transformers.
- SVM time grows steeply with training size; 100% is the slow case.
