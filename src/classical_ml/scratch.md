# Classical ML experiment summary

Verified runs: RF, XGBoost, and SVM at 1, 3, 5, 10, 25, 50, and 100% of the
training data, seeds 0, 1, and 42 (63 runs). Settings were selected by
**validation macro AUROC**, never by test metrics.

## Provenance and settings

- Artifacts: `artifacts/classical_ml/<model>_seed<S>_frac<N>/` (git-ignored).
- Dataset: PTB-XL records100; 100 Hz, 10 s, 12 leads; NORM, MI, STTC, CD, HYP.
  Validation is fold 9 (2,146 ECGs), test is fold 10 (2,158 ECGs).
- Features: 300-d db4 wavelet statistics after a 0.5–40 Hz band-pass and per-lead,
  per-record z-score.
- Class weighting: balanced (RF, SVM) or `scale_pos_weight` (XGBoost), per class.
- Grids: RF `n_estimators` 300, `max_depth` None/20, `min_samples_leaf` 1/3;
  XGBoost `n_estimators` 200/400, `max_depth` 3/6, `learning_rate` 0.05/0.1;
  SVM `C` 0.1/1/10, `gamma` scale/0.001.
- Device: 10-core Mac, CPU only. Times below cover the whole grid and depend on
  machine load.
- Selected epoch: not applicable (no epoch loop).
- Thresholds: per class, chosen on validation and frozen for test.
- Code state: seed 42 at 5–100% comes from the results committed in `07498ef`
  (`reports/`). All other runs were made at commit `9d3d560`. All were converted
  to the current layout; macro metrics were checked against the original
  summaries and recomputed from the saved predictions. These runs have no
  `grid_results.json`.

## Observations (test set)

- Mean AUROC rises with training size for all three models.
- At 1% the seed standard deviation (0.011–0.020 AUROC) is as large as the gaps
  between models. Three seeds give only a rough estimate of spread.
- At 100% the SVM has the highest AUROC (0.853), then XGBoost (0.845) and RF
  (0.839). On AUPRC, XGBoost is highest (0.664).
- The 100% spread is not training-sample variance: all seeds share `train_100`.
- In the 100% seed-42 runs, HYP is the weakest class by AUROC, AUPRC, and F1.
- For reference, CNN-BiLSTM seed-42 test AUROC is 0.855 at 5% and 0.924 at 100%
  (`reports/cnn_bilstm/seed_42/summary.csv`), about 0.06–0.07 above the best
  classical model at those sizes.

## Caveats

- XGBoost always selected 400 trees, and depth 6 at every fraction above 1%. Both
  are the largest values offered, so XGBoost may be under-tuned.
- Classical models are grid-searched on validation; the deep models are not tuned
  this way, and inputs are normalised differently across families.
- SVM scores are not calibrated probabilities.
- Regenerate the tables below with `python -m src.classical_ml.summarize`.

## Cross-seed results (seeds 0, 1, 42)

Mean ± sample standard deviation over seeds; [n] is the number of seeds.

### Validation macro AUROC

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| rf | 0.751 ± 0.006 [3] | 0.788 ± 0.004 [3] | 0.802 ± 0.009 [3] | 0.816 ± 0.002 [3] | 0.828 ± 0.002 [3] | 0.836 ± 0.003 [3] | 0.843 ± 0.000 [3] |
| xgb | 0.725 ± 0.011 [3] | 0.776 ± 0.006 [3] | 0.793 ± 0.005 [3] | 0.815 ± 0.001 [3] | 0.832 ± 0.003 [3] | 0.840 ± 0.002 [3] | 0.850 ± 0.000 [3] |
| svm | 0.756 ± 0.013 [3] | 0.786 ± 0.003 [3] | 0.802 ± 0.007 [3] | 0.820 ± 0.001 [3] | 0.839 ± 0.001 [3] | 0.849 ± 0.002 [3] | 0.859 ± 0.000 [3] |

### Validation macro AUPRC

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| rf | 0.514 ± 0.005 [3] | 0.562 ± 0.004 [3] | 0.582 ± 0.011 [3] | 0.603 ± 0.002 [3] | 0.623 ± 0.002 [3] | 0.634 ± 0.004 [3] | 0.645 ± 0.002 [3] |
| xgb | 0.485 ± 0.010 [3] | 0.547 ± 0.006 [3] | 0.570 ± 0.002 [3] | 0.606 ± 0.004 [3] | 0.634 ± 0.006 [3] | 0.653 ± 0.004 [3] | 0.666 ± 0.000 [3] |
| svm | 0.513 ± 0.008 [3] | 0.558 ± 0.007 [3] | 0.577 ± 0.002 [3] | 0.606 ± 0.001 [3] | 0.632 ± 0.003 [3] | 0.649 ± 0.002 [3] | 0.662 ± 0.000 [3] |

### Validation macro F1

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| rf | 0.530 ± 0.003 [3] | 0.561 ± 0.001 [3] | 0.574 ± 0.008 [3] | 0.588 ± 0.005 [3] | 0.604 ± 0.001 [3] | 0.613 ± 0.004 [3] | 0.620 ± 0.001 [3] |
| xgb | 0.504 ± 0.010 [3] | 0.548 ± 0.005 [3] | 0.563 ± 0.007 [3] | 0.589 ± 0.001 [3] | 0.612 ± 0.006 [3] | 0.626 ± 0.003 [3] | 0.645 ± 0.000 [3] |
| svm | 0.533 ± 0.010 [3] | 0.561 ± 0.006 [3] | 0.576 ± 0.008 [3] | 0.601 ± 0.001 [3] | 0.622 ± 0.003 [3] | 0.634 ± 0.002 [3] | 0.645 ± 0.000 [3] |

### Test macro AUROC

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| rf | 0.743 ± 0.011 [3] | 0.783 ± 0.000 [3] | 0.796 ± 0.005 [3] | 0.809 ± 0.004 [3] | 0.824 ± 0.001 [3] | 0.832 ± 0.001 [3] | 0.839 ± 0.001 [3] |
| xgb | 0.722 ± 0.015 [3] | 0.774 ± 0.006 [3] | 0.790 ± 0.006 [3] | 0.809 ± 0.006 [3] | 0.828 ± 0.005 [3] | 0.839 ± 0.005 [3] | 0.845 ± 0.000 [3] |
| svm | 0.746 ± 0.020 [3] | 0.781 ± 0.001 [3] | 0.795 ± 0.005 [3] | 0.812 ± 0.001 [3] | 0.831 ± 0.003 [3] | 0.843 ± 0.001 [3] | 0.853 ± 0.000 [3] |

### Test macro AUPRC

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| rf | 0.505 ± 0.011 [3] | 0.555 ± 0.004 [3] | 0.575 ± 0.006 [3] | 0.595 ± 0.004 [3] | 0.619 ± 0.001 [3] | 0.629 ± 0.004 [3] | 0.643 ± 0.002 [3] |
| xgb | 0.480 ± 0.016 [3] | 0.541 ± 0.009 [3] | 0.568 ± 0.004 [3] | 0.599 ± 0.011 [3] | 0.631 ± 0.005 [3] | 0.649 ± 0.007 [3] | 0.664 ± 0.000 [3] |
| svm | 0.503 ± 0.015 [3] | 0.551 ± 0.003 [3] | 0.572 ± 0.004 [3] | 0.598 ± 0.002 [3] | 0.623 ± 0.002 [3] | 0.642 ± 0.002 [3] | 0.659 ± 0.000 [3] |

### Test macro F1

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| rf | 0.509 ± 0.005 [3] | 0.548 ± 0.005 [3] | 0.559 ± 0.004 [3] | 0.570 ± 0.003 [3] | 0.588 ± 0.008 [3] | 0.596 ± 0.002 [3] | 0.602 ± 0.006 [3] |
| xgb | 0.490 ± 0.009 [3] | 0.535 ± 0.004 [3] | 0.549 ± 0.003 [3] | 0.567 ± 0.005 [3] | 0.594 ± 0.001 [3] | 0.608 ± 0.006 [3] | 0.617 ± 0.000 [3] |
| svm | 0.515 ± 0.019 [3] | 0.544 ± 0.007 [3] | 0.563 ± 0.009 [3] | 0.580 ± 0.002 [3] | 0.595 ± 0.006 [3] | 0.611 ± 0.000 [3] | 0.623 ± 0.000 [3] |

## Every run

Selected setting is the grid entry with the highest validation macro AUROC. Time covers the whole grid.

| Run | ECGs | Selected setting | Seconds | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|---|---|---|---|---|---|---|---|---|---|
| `rf_seed0_frac1` | 171 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 6.3 | 0.7458 | 0.5084 | 0.5271 | 0.7307 | 0.4920 | 0.5036 |
| `rf_seed0_frac3` | 502 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 7.9 | 0.7887 | 0.5626 | 0.5599 | 0.7829 | 0.5528 | 0.5525 |
| `rf_seed0_frac5` | 844 | n_estimators=300, max_depth=20, min_samples_leaf=1 | 11.0 | 0.7987 | 0.5725 | 0.5695 | 0.7956 | 0.5684 | 0.5621 |
| `rf_seed0_frac10` | 1719 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 15.9 | 0.8171 | 0.6054 | 0.5867 | 0.8129 | 0.5986 | 0.5725 |
| `rf_seed0_frac25` | 4241 | n_estimators=300, max_depth=20, min_samples_leaf=3 | 37.6 | 0.8285 | 0.6233 | 0.6049 | 0.8250 | 0.6194 | 0.5934 |
| `rf_seed0_frac50` | 8535 | n_estimators=300, max_depth=20, min_samples_leaf=1 | 84.9 | 0.8349 | 0.6324 | 0.6171 | 0.8320 | 0.6254 | 0.5953 |
| `rf_seed0_frac100` | 17084 | n_estimators=300, max_depth=20, min_samples_leaf=3 | 208.6 | 0.8426 | 0.6430 | 0.6191 | 0.8384 | 0.6403 | 0.5954 |
| `rf_seed1_frac1` | 165 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 3.9 | 0.7570 | 0.5164 | 0.5323 | 0.7509 | 0.5087 | 0.5086 |
| `rf_seed1_frac3` | 517 | n_estimators=300, max_depth=20, min_samples_leaf=1 | 4.9 | 0.7924 | 0.5661 | 0.5610 | 0.7824 | 0.5526 | 0.5426 |
| `rf_seed1_frac5` | 852 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 6.9 | 0.8112 | 0.5935 | 0.5834 | 0.8012 | 0.5775 | 0.5596 |
| `rf_seed1_frac10` | 1699 | n_estimators=300, max_depth=20, min_samples_leaf=3 | 10.0 | 0.8163 | 0.6025 | 0.5833 | 0.8089 | 0.5902 | 0.5664 |
| `rf_seed1_frac25` | 4284 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 25.0 | 0.8294 | 0.6244 | 0.6041 | 0.8229 | 0.6182 | 0.5787 |
| `rf_seed1_frac50` | 8575 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 56.7 | 0.8390 | 0.6385 | 0.6144 | 0.8323 | 0.6328 | 0.5974 |
| `rf_seed1_frac100` | 17084 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 128.7 | 0.8426 | 0.6441 | 0.6191 | 0.8395 | 0.6449 | 0.6042 |
| `rf_seed42_frac1` | 164 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 6.3 | 0.7506 | 0.5174 | 0.5297 | 0.7484 | 0.5140 | 0.5134 |
| `rf_seed42_frac3` | 498 | n_estimators=300, max_depth=None, min_samples_leaf=3 | 7.8 | 0.7835 | 0.5575 | 0.5611 | 0.7828 | 0.5601 | 0.5475 |
| `rf_seed42_frac5` | 841 | n_estimators=300, max_depth=20, min_samples_leaf=3 | 10.3 | 0.7948 | 0.5803 | 0.5687 | 0.7911 | 0.5789 | 0.5543 |
| `rf_seed42_frac10` | 1687 | n_estimators=300, max_depth=20, min_samples_leaf=3 | 16.1 | 0.8133 | 0.6024 | 0.5940 | 0.8049 | 0.5963 | 0.5721 |
| `rf_seed42_frac25` | 4252 | n_estimators=300, max_depth=20, min_samples_leaf=3 | 35.3 | 0.8262 | 0.6208 | 0.6032 | 0.8243 | 0.6198 | 0.5930 |
| `rf_seed42_frac50` | 8521 | n_estimators=300, max_depth=20, min_samples_leaf=1 | 81.7 | 0.8344 | 0.6319 | 0.6085 | 0.8311 | 0.6280 | 0.5943 |
| `rf_seed42_frac100` | 17084 | n_estimators=300, max_depth=20, min_samples_leaf=3 | 191.7 | 0.8427 | 0.6466 | 0.6213 | 0.8380 | 0.6430 | 0.6058 |
| `xgb_seed0_frac1` | 171 | n_estimators=400, max_depth=3, learning_rate=0.05 | 19.4 | 0.7122 | 0.4743 | 0.4930 | 0.7052 | 0.4615 | 0.4790 |
| `xgb_seed0_frac3` | 502 | n_estimators=400, max_depth=6, learning_rate=0.05 | 46.0 | 0.7762 | 0.5469 | 0.5505 | 0.7689 | 0.5311 | 0.5324 |
| `xgb_seed0_frac5` | 844 | n_estimators=400, max_depth=6, learning_rate=0.1 | 58.0 | 0.7888 | 0.5672 | 0.5574 | 0.7961 | 0.5711 | 0.5524 |
| `xgb_seed0_frac10` | 1719 | n_estimators=400, max_depth=6, learning_rate=0.05 | 79.7 | 0.8137 | 0.6091 | 0.5898 | 0.8163 | 0.6107 | 0.5705 |
| `xgb_seed0_frac25` | 4241 | n_estimators=400, max_depth=6, learning_rate=0.1 | 117.1 | 0.8354 | 0.6392 | 0.6189 | 0.8320 | 0.6330 | 0.5924 |
| `xgb_seed0_frac50` | 8535 | n_estimators=400, max_depth=6, learning_rate=0.1 | 150.2 | 0.8382 | 0.6536 | 0.6288 | 0.8418 | 0.6543 | 0.6090 |
| `xgb_seed0_frac100` | 17084 | n_estimators=400, max_depth=6, learning_rate=0.1 | 198.1 | 0.8502 | 0.6660 | 0.6449 | 0.8453 | 0.6635 | 0.6172 |
| `xgb_seed1_frac1` | 165 | n_estimators=400, max_depth=3, learning_rate=0.05 | 11.7 | 0.7317 | 0.4922 | 0.5122 | 0.7275 | 0.4872 | 0.4957 |
| `xgb_seed1_frac3` | 517 | n_estimators=400, max_depth=6, learning_rate=0.1 | 27.6 | 0.7815 | 0.5532 | 0.5510 | 0.7809 | 0.5475 | 0.5391 |
| `xgb_seed1_frac5` | 852 | n_estimators=400, max_depth=6, learning_rate=0.05 | 36.3 | 0.7981 | 0.5711 | 0.5705 | 0.7891 | 0.5641 | 0.5469 |
| `xgb_seed1_frac10` | 1699 | n_estimators=400, max_depth=6, learning_rate=0.1 | 49.6 | 0.8159 | 0.6023 | 0.5871 | 0.8049 | 0.5901 | 0.5619 |
| `xgb_seed1_frac25` | 4284 | n_estimators=400, max_depth=6, learning_rate=0.1 | 71.9 | 0.8306 | 0.6347 | 0.6114 | 0.8304 | 0.6350 | 0.5931 |
| `xgb_seed1_frac50` | 8575 | n_estimators=400, max_depth=6, learning_rate=0.1 | 89.3 | 0.8425 | 0.6561 | 0.6273 | 0.8417 | 0.6517 | 0.6129 |
| `xgb_seed1_frac100` | 17084 | n_estimators=400, max_depth=6, learning_rate=0.1 | 117.2 | 0.8502 | 0.6660 | 0.6449 | 0.8453 | 0.6635 | 0.6172 |
| `xgb_seed42_frac1` | 164 | n_estimators=400, max_depth=3, learning_rate=0.05 | 18.8 | 0.7309 | 0.4900 | 0.5079 | 0.7326 | 0.4912 | 0.4945 |
| `xgb_seed42_frac3` | 498 | n_estimators=400, max_depth=6, learning_rate=0.05 | 47.2 | 0.7699 | 0.5414 | 0.5419 | 0.7709 | 0.5442 | 0.5337 |
| `xgb_seed42_frac5` | 841 | n_estimators=400, max_depth=6, learning_rate=0.05 | 58.5 | 0.7907 | 0.5707 | 0.5624 | 0.7848 | 0.5690 | 0.5471 |
| `xgb_seed42_frac10` | 1687 | n_estimators=400, max_depth=6, learning_rate=0.1 | 100.8 | 0.8147 | 0.6080 | 0.5893 | 0.8064 | 0.5966 | 0.5685 |
| `xgb_seed42_frac25` | 4252 | n_estimators=400, max_depth=6, learning_rate=0.05 | 111.3 | 0.8291 | 0.6268 | 0.6063 | 0.8230 | 0.6256 | 0.5952 |
| `xgb_seed42_frac50` | 8521 | n_estimators=400, max_depth=6, learning_rate=0.1 | 143.6 | 0.8386 | 0.6478 | 0.6222 | 0.8329 | 0.6406 | 0.6009 |
| `xgb_seed42_frac100` | 17084 | n_estimators=400, max_depth=6, learning_rate=0.1 | 183.0 | 0.8502 | 0.6660 | 0.6449 | 0.8453 | 0.6635 | 0.6172 |
| `svm_seed0_frac1` | 171 | C=1, gamma=0.001 | 5.6 | 0.7408 | 0.5048 | 0.5216 | 0.7225 | 0.4868 | 0.4939 |
| `svm_seed0_frac3` | 502 | C=1, gamma=0.001 | 4.0 | 0.7871 | 0.5642 | 0.5668 | 0.7809 | 0.5480 | 0.5461 |
| `svm_seed0_frac5` | 844 | C=1, gamma=0.001 | 6.7 | 0.8015 | 0.5771 | 0.5811 | 0.7965 | 0.5731 | 0.5727 |
| `svm_seed0_frac10` | 1719 | C=1, gamma=0.001 | 15.3 | 0.8203 | 0.6047 | 0.6011 | 0.8136 | 0.5996 | 0.5799 |
| `svm_seed0_frac25` | 4241 | C=1, gamma=scale | 52.3 | 0.8378 | 0.6349 | 0.6248 | 0.8330 | 0.6244 | 0.5995 |
| `svm_seed0_frac50` | 8535 | C=1, gamma=scale | 156.1 | 0.8475 | 0.6473 | 0.6324 | 0.8412 | 0.6402 | 0.6107 |
| `svm_seed0_frac100` | 17084 | C=1, gamma=scale | 545.1 | 0.8586 | 0.6624 | 0.6452 | 0.8532 | 0.6586 | 0.6235 |
| `svm_seed1_frac1` | 165 | C=1, gamma=0.001 | 3.1 | 0.7636 | 0.5192 | 0.5399 | 0.7558 | 0.5040 | 0.5230 |
| `svm_seed1_frac3` | 517 | C=1, gamma=scale | 2.8 | 0.7879 | 0.5600 | 0.5612 | 0.7801 | 0.5503 | 0.5362 |
| `svm_seed1_frac5` | 852 | C=1, gamma=0.001 | 4.4 | 0.8084 | 0.5799 | 0.5796 | 0.7997 | 0.5671 | 0.5595 |
| `svm_seed1_frac10` | 1699 | C=1, gamma=0.001 | 8.7 | 0.8218 | 0.6052 | 0.5996 | 0.8123 | 0.5965 | 0.5777 |
| `svm_seed1_frac25` | 4284 | C=1, gamma=scale | 33.2 | 0.8392 | 0.6291 | 0.6202 | 0.8279 | 0.6210 | 0.5882 |
| `svm_seed1_frac50` | 8575 | C=1, gamma=scale | 108.1 | 0.8505 | 0.6503 | 0.6359 | 0.8434 | 0.6445 | 0.6111 |
| `svm_seed1_frac100` | 17084 | C=1, gamma=scale | 582.1 | 0.8586 | 0.6624 | 0.6452 | 0.8532 | 0.6586 | 0.6235 |
| `svm_seed42_frac1` | 164 | C=1, gamma=scale | 4.6 | 0.7632 | 0.5159 | 0.5373 | 0.7587 | 0.5170 | 0.5286 |
| `svm_seed42_frac3` | 498 | C=1, gamma=scale | 3.9 | 0.7825 | 0.5508 | 0.5553 | 0.7829 | 0.5539 | 0.5505 |
| `svm_seed42_frac5` | 841 | C=1, gamma=scale | 8.9 | 0.7950 | 0.5752 | 0.5659 | 0.7892 | 0.5756 | 0.5557 |
| `svm_seed42_frac10` | 1687 | C=1, gamma=scale | 13.0 | 0.8189 | 0.6071 | 0.6011 | 0.8113 | 0.5973 | 0.5812 |
| `svm_seed42_frac25` | 4252 | C=1, gamma=scale | 49.2 | 0.8396 | 0.6331 | 0.6197 | 0.8317 | 0.6249 | 0.5971 |
| `svm_seed42_frac50` | 8521 | C=1, gamma=scale | 164.7 | 0.8500 | 0.6481 | 0.6333 | 0.8430 | 0.6408 | 0.6103 |
| `svm_seed42_frac100` | 17084 | C=1, gamma=scale | 1074.6 | 0.8586 | 0.6624 | 0.6452 | 0.8532 | 0.6586 | 0.6235 |

## Classwise results (100% training data, seed 42)

Thresholds are chosen per class on validation and frozen for test.

### rf_seed42_frac100

| Split | Class | AUROC | AUPRC | F1 | Threshold |
|---|---|---|---|---|---|
| Validation | NORM | 0.8877 | 0.8472 | 0.7965 | 0.45 |
| Validation | MI | 0.8345 | 0.6233 | 0.6021 | 0.40 |
| Validation | STTC | 0.8803 | 0.6911 | 0.6639 | 0.45 |
| Validation | CD | 0.8527 | 0.7219 | 0.6500 | 0.45 |
| Validation | HYP | 0.7582 | 0.3494 | 0.3939 | 0.35 |
| Test | NORM | 0.8799 | 0.8381 | 0.7842 | 0.45 |
| Test | MI | 0.8225 | 0.6319 | 0.6055 | 0.40 |
| Test | STTC | 0.8743 | 0.6826 | 0.6432 | 0.45 |
| Test | CD | 0.8305 | 0.6911 | 0.5940 | 0.45 |
| Test | HYP | 0.7826 | 0.3713 | 0.4022 | 0.35 |

### xgb_seed42_frac100

| Split | Class | AUROC | AUPRC | F1 | Threshold |
|---|---|---|---|---|---|
| Validation | NORM | 0.8989 | 0.8539 | 0.8096 | 0.45 |
| Validation | MI | 0.8468 | 0.6766 | 0.6311 | 0.35 |
| Validation | STTC | 0.8879 | 0.7040 | 0.7032 | 0.40 |
| Validation | CD | 0.8595 | 0.7423 | 0.6667 | 0.40 |
| Validation | HYP | 0.7578 | 0.3533 | 0.4140 | 0.15 |
| Test | NORM | 0.8941 | 0.8576 | 0.7932 | 0.45 |
| Test | MI | 0.8339 | 0.6713 | 0.6210 | 0.35 |
| Test | STTC | 0.8832 | 0.6939 | 0.6606 | 0.40 |
| Test | CD | 0.8366 | 0.7097 | 0.6175 | 0.40 |
| Test | HYP | 0.7788 | 0.3851 | 0.3935 | 0.15 |

### svm_seed42_frac100

| Split | Class | AUROC | AUPRC | F1 | Threshold |
|---|---|---|---|---|---|
| Validation | NORM | 0.8986 | 0.8584 | 0.8111 | 0.50 |
| Validation | MI | 0.8573 | 0.6776 | 0.6363 | 0.45 |
| Validation | STTC | 0.8888 | 0.7017 | 0.6814 | 0.55 |
| Validation | CD | 0.8755 | 0.7259 | 0.6890 | 0.50 |
| Validation | HYP | 0.7726 | 0.3482 | 0.4082 | 0.45 |
| Test | NORM | 0.8933 | 0.8469 | 0.8029 | 0.50 |
| Test | MI | 0.8450 | 0.6686 | 0.6287 | 0.45 |
| Test | STTC | 0.8850 | 0.6915 | 0.6547 | 0.55 |
| Test | CD | 0.8458 | 0.6889 | 0.6184 | 0.50 |
| Test | HYP | 0.7970 | 0.3971 | 0.4125 | 0.45 |

