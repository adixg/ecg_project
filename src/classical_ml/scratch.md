# Scratch: recorded classical-ML run metadata

Snapshot of the completed seed-42 runs. Settings were selected by **validation
macro AUROC**, not by test metrics. Full per-class numbers are in
`reports/<rf|xgb|svm>/seed_42/fraction_<N>/metrics.json`.

## Shared dataset and settings

- Dataset: PTB-XL records100; 100 Hz, 10 seconds, 12 leads; five superclass labels
  (NORM, MI, STTC, CD, HYP), one-vs-rest.
- Features: 300-d db4 wavelet statistics (`features.py`), computed once for all splits.
- Validation: fold 9, 2,146 ECGs. Test: fold 10, 2,158 ECGs.
- Seed 42. Run locally on a 10-core Mac, CPU only.
- Metrics and F1 thresholds from `src/metrics.py`: one threshold per class from the
  0.05–0.95 grid, chosen on validation and frozen for test (same as the CNN-BiLSTM).
- SVM scores are `sigmoid(decision_function)`: AUROC/AUPRC are unaffected, but the
  value is not a calibrated probability.

## Test results (macro, seed 42)

| Training % | ECGs | RF AUROC | XGB AUROC | SVM AUROC | RF AUPRC | XGB AUPRC | SVM AUPRC | RF F1 | XGB F1 | SVM F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 841 | 0.7911 | 0.7848 | 0.7892 | 0.5789 | 0.5690 | 0.5756 | 0.5543 | 0.5471 | 0.5557 |
| 10 | 1687 | 0.8049 | 0.8064 | 0.8113 | 0.5963 | 0.5966 | 0.5973 | 0.5721 | 0.5685 | 0.5812 |
| 25 | 4252 | 0.8243 | 0.8230 | 0.8317 | 0.6198 | 0.6256 | 0.6249 | 0.5930 | 0.5952 | 0.5971 |
| 50 | 8521 | 0.8311 | 0.8329 | 0.8430 | 0.6280 | 0.6406 | 0.6408 | 0.5943 | 0.6009 | 0.6103 |
| 100 | 17084 | 0.8380 | 0.8453 | 0.8532 | 0.6430 | 0.6635 | 0.6586 | 0.6058 | 0.6172 | 0.6235 |

For reference, the CNN-BiLSTM seed-42 test results (`reports/cnn_bilstm/seed_42/summary.csv`):

| Training % | AUROC | AUPRC | F1 |
|---|---:|---:|---:|
| 5 | 0.8551 | 0.6892 | 0.6411 |
| 10 | 0.8798 | 0.7333 | 0.6746 |
| 25 | 0.9006 | 0.7734 | 0.6968 |
| 50 | 0.9126 | 0.7981 | 0.7247 |
| 100 | 0.9236 | 0.8174 | 0.7464 |

## Per-class test AUROC at 5% and 100%

| Model | Training % | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|---:|
| RF | 5 | 0.846 | 0.788 | 0.834 | 0.793 | 0.695 |
| RF | 100 | 0.880 | 0.823 | 0.874 | 0.831 | 0.783 |
| XGB | 5 | 0.839 | 0.781 | 0.819 | 0.792 | 0.694 |
| XGB | 100 | 0.894 | 0.834 | 0.883 | 0.837 | 0.779 |
| SVM | 5 | 0.851 | 0.794 | 0.828 | 0.798 | 0.675 |
| SVM | 100 | 0.893 | 0.845 | 0.885 | 0.846 | 0.797 |

## Selected hyperparameters

| Model | Training % | Selected setting |
|---|---|---|
| RF | 5, 10, 25, 100 | 300 trees, max_depth 20, min_samples_leaf 3 |
| RF | 50 | 300 trees, max_depth 20, min_samples_leaf 1 |
| XGB | 5, 25 | 400 trees, max_depth 6, learning_rate 0.05 |
| XGB | 10, 50, 100 | 400 trees, max_depth 6, learning_rate 0.1 |
| SVM | all | C = 1, gamma = "scale" |

## Run time (seconds, full grid plus final predictions)

| Training % | RF | XGB | SVM |
|---|---:|---:|---:|
| 5 | 10 | 59 | 9 |
| 10 | 16 | 101 | 13 |
| 25 | 35 | 111 | 49 |
| 50 | 82 | 144 | 165 |
| 100 | 192 | 183 | 1075 |

SVM time grows roughly quadratically with training size, so it is the only model
where the 100% fraction is expensive.

## Interpretation

- All three models improve monotonically in AUROC and AUPRC as the training set
  grows. From 5% to 100%, AUROC rises by about 0.05–0.06 for each model.
- The three models are close at every fraction (within about 0.015 AUROC). The SVM
  is highest at 10% and above; at 5% RF is highest in AUROC, by 0.002 over the SVM.
  Differences this small come from a single seed and should not be ranked.
- The CNN-BiLSTM is ahead of every classical model at every fraction. The gap is
  about 0.065 AUROC at 5% and about 0.07 at 100%, so the curves are roughly
  parallel and the gap is not closing within this range.
- HYP is the weakest class for every model (AUROC 0.68–0.80, F1 0.29–0.41), and
  NORM is the strongest.
- Validation scores are slightly above test scores in every run, as expected when
  the setting and thresholds are chosen on validation.

## Caveats

- Single seed. RF and XGBoost depend on it; the SVM is deterministic. No variability
  or confidence intervals were measured.
- Grid edges: XGBoost always selected the largest `max_depth` (6) and `n_estimators`
  (400) offered, and learning rate 0.1 in three of five fractions; RF always selected
  `max_depth` 20, the larger value offered. Larger values might do better, so the
  XGBoost and RF numbers could be slightly under-tuned. The SVM chose the middle `C`.
- The classical models are grid-searched on validation, while the deep models use
  fixed hyperparameters with early stopping on validation. Inputs are normalised
  differently across model families (per-record z-score here, raw mV for the CNN).
  Both differences are left as is.
- `import torch` before `import xgboost` crashes XGBoost on macOS; `src/__init__.py`
  imports the CNN lazily because of this.
