# Transformer experiment summary

Verified benchmark: both Transformer variants at 1, 3, 5, 10, 25, 50, and
100% of the training data, using seeds 0, 1, and 42 (42 trained models).
Checkpoints were selected by **validation macro AUROC**. Per-class F1 thresholds
were selected on validation after checkpoint selection and frozen before test
inference.

## Run provenance

- V1 seed 0: `artifacts/transformers/v1/20261128T101602722080Z_seed0`
- V1 seed 1: `artifacts/transformers/v1/20261128T102405256192Z_seed1`
- V1 seed 42: `artifacts/transformers/v1/20261128T103222909967Z_seed42`
- V2 seed 0: `artifacts/transformers/v2/20261128T103859720841Z_seed0`
- V2 seed 1: `artifacts/transformers/v2/20261128T104638174521Z_seed1`
- V2 seed 42: `artifacts/transformers/v2/20261128T105446971156Z_seed42`
- Console logs: `artifacts/transformers/logs/{v1,v2}_seed{0,1,42}.log`
- Dataset: PTB-XL 1.0.1, `records100`, 100 Hz, 10 seconds, 12 leads.
- Metadata SHA256: `cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156`.
- SCP mapping SHA256: `ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f`.
- Splits: `splits/seed_<seed>/`; folds 1–8 train, fold 9 validation, fold
  10 test. Fractions are nested within each seed. Validation, test, and 100%
  training data are identical across seeds.
- CSV labels remained authoritative. The audits record ECG 3799 as MI=0 versus
  MI=1 under diagnostic-code presence mapping, and ECG 9822 as CD=0 versus CD=1.
- Code state: repository base `2f32623` plus the Transformer runner changes in
  this worktree that add seeded split resolution, 1%/3% fractions, and canonical
  artifact paths.

## Shared training protocol

- Models: V1 linear 20-sample patches with learned positions (632,965 trainable
  parameters); V2 convolutional tokens with sinusoidal positions (650,309).
- Both: 128-dimensional embeddings, four attention heads, three pre-norm
  encoder layers, 512-wide feed-forward blocks, dropout 0.1.
- Fresh model per fraction; seed controls the patient subset, Python, NumPy,
  PyTorch, and data-loader order.
- Per-lead population mean/std fitted separately on each training fraction and
  stored in the checkpoint.
- AdamW, learning rate 0.0003, weight decay 0.01, batch size 64, unweighted BCE,
  gradient clipping at 1.0, cosine schedule to 0.00003.
- Maximum 30 epochs; patience 8; a validation macro-AUROC improvement greater
  than 0.0001 resets patience. Classifier bias starts at training log-odds.
- Thresholds maximize validation F1 independently by class. Test is evaluated
  only after all seven checkpoints and thresholds in a model-seed invocation
  are frozen.
- Environment: Python 3.11.14, PyTorch 2.10.0+cu128, CUDA 12.8, NVIDIA GeForce
  GTX 1650 (4 GB), numpy 2.4.6, pandas 3.1.1, scipy 1.17.1,
  scikit-learn 1.9.1, WFDB 5.5.1, matplotlib 3.11.2.
- Epoch timers total 896.7 seconds for V1 and 1,045.5 seconds for V2 (32.4
  minutes overall). These timers exclude waveform-cache construction, final
  inference, plots, and artifact writing.

## Cross-seed results

Mean ± sample standard deviation over seeds 0, 1, and 42. F1 uses the frozen
validation-selected thresholds.

### Validation macro AUROC

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| V1 | 0.753 ± 0.021 | 0.814 ± 0.011 | 0.833 ± 0.003 | 0.853 ± 0.003 | 0.880 ± 0.002 | 0.895 ± 0.001 | 0.908 ± 0.001 |
| V2 | 0.799 ± 0.015 | 0.834 ± 0.010 | 0.849 ± 0.003 | 0.869 ± 0.004 | 0.890 ± 0.000 | 0.903 ± 0.001 | 0.915 ± 0.001 |

### Validation macro AUPRC

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| V1 | 0.491 ± 0.031 | 0.585 ± 0.017 | 0.621 ± 0.012 | 0.657 ± 0.006 | 0.720 ± 0.000 | 0.748 ± 0.006 | 0.778 ± 0.002 |
| V2 | 0.551 ± 0.025 | 0.617 ± 0.022 | 0.647 ± 0.015 | 0.692 ± 0.010 | 0.735 ± 0.005 | 0.762 ± 0.009 | 0.786 ± 0.002 |

### Validation macro F1

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| V1 | 0.532 ± 0.021 | 0.591 ± 0.011 | 0.610 ± 0.009 | 0.638 ± 0.001 | 0.681 ± 0.003 | 0.699 ± 0.004 | 0.719 ± 0.001 |
| V2 | 0.573 ± 0.019 | 0.616 ± 0.016 | 0.630 ± 0.005 | 0.665 ± 0.004 | 0.696 ± 0.007 | 0.711 ± 0.006 | 0.732 ± 0.006 |

### Test macro AUROC

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| V1 | 0.749 ± 0.021 | 0.805 ± 0.007 | 0.823 ± 0.001 | 0.841 ± 0.006 | 0.869 ± 0.002 | 0.887 ± 0.002 | 0.899 ± 0.003 |
| V2 | 0.785 ± 0.019 | 0.820 ± 0.010 | 0.835 ± 0.003 | 0.856 ± 0.002 | 0.880 ± 0.002 | 0.893 ± 0.002 | 0.908 ± 0.002 |

### Test macro AUPRC

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| V1 | 0.486 ± 0.035 | 0.578 ± 0.020 | 0.619 ± 0.005 | 0.648 ± 0.014 | 0.709 ± 0.005 | 0.744 ± 0.002 | 0.769 ± 0.007 |
| V2 | 0.541 ± 0.037 | 0.613 ± 0.030 | 0.649 ± 0.017 | 0.691 ± 0.007 | 0.732 ± 0.010 | 0.755 ± 0.006 | 0.786 ± 0.006 |

### Test macro F1

| Model | 1% | 3% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|---|
| V1 | 0.514 ± 0.013 | 0.572 ± 0.008 | 0.593 ± 0.007 | 0.619 ± 0.008 | 0.661 ± 0.005 | 0.684 ± 0.002 | 0.706 ± 0.005 |
| V2 | 0.551 ± 0.018 | 0.589 ± 0.012 | 0.607 ± 0.009 | 0.640 ± 0.001 | 0.674 ± 0.005 | 0.691 ± 0.008 | 0.714 ± 0.007 |

## Individual runs

`Best/ran` is selected epoch / epochs executed. `Train s` is the sum of epoch
timers. Validation and test F1 use the validation-selected thresholds.

| Run | ECGs | Best/ran | Train s | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1_seed0_frac1` | 171 | 17/25 | 13.9 | 0.7300 | 0.4595 | 0.5089 | 0.7256 | 0.4460 | 0.4995 |
| `v1_seed0_frac3` | 502 | 8/16 | 9.7 | 0.8046 | 0.5687 | 0.5814 | 0.7973 | 0.5548 | 0.5633 |
| `v1_seed0_frac5` | 844 | 7/15 | 11.8 | 0.8332 | 0.6236 | 0.6072 | 0.8236 | 0.6136 | 0.5892 |
| `v1_seed0_frac10` | 1,719 | 7/15 | 17.2 | 0.8563 | 0.6644 | 0.6390 | 0.8452 | 0.6569 | 0.6218 |
| `v1_seed0_frac25` | 4,241 | 10/18 | 43.5 | 0.8813 | 0.7191 | 0.6817 | 0.8692 | 0.7037 | 0.6583 |
| `v1_seed0_frac50` | 8,535 | 7/15 | 66.0 | 0.8954 | 0.7528 | 0.7003 | 0.8849 | 0.7414 | 0.6854 |
| `v1_seed0_frac100` | 17,084 | 6/14 | 119.6 | 0.9063 | 0.7760 | 0.7192 | 0.8970 | 0.7646 | 0.7037 |
| `v1_seed1_frac1` | 165 | 11/19 | 16.2 | 0.7572 | 0.4927 | 0.5352 | 0.7639 | 0.5118 | 0.5228 |
| `v1_seed1_frac3` | 517 | 9/17 | 10.6 | 0.8117 | 0.5822 | 0.5888 | 0.8070 | 0.5858 | 0.5789 |
| `v1_seed1_frac5` | 852 | 7/15 | 11.7 | 0.8306 | 0.6079 | 0.6020 | 0.8225 | 0.6182 | 0.5898 |
| `v1_seed1_frac10` | 1,699 | 9/17 | 20.1 | 0.8512 | 0.6516 | 0.6387 | 0.8434 | 0.6546 | 0.6248 |
| `v1_seed1_frac25` | 4,284 | 8/16 | 37.7 | 0.8808 | 0.7200 | 0.6825 | 0.8708 | 0.7143 | 0.6667 |
| `v1_seed1_frac50` | 8,575 | 10/18 | 78.0 | 0.8930 | 0.7418 | 0.6938 | 0.8873 | 0.7454 | 0.6812 |
| `v1_seed1_frac100` | 17,084 | 11/19 | 156.7 | 0.9085 | 0.7803 | 0.7186 | 0.9025 | 0.7774 | 0.7112 |
| `v1_seed42_frac1` | 164 | 25/30 | 15.7 | 0.7715 | 0.5215 | 0.5507 | 0.7588 | 0.5013 | 0.5192 |
| `v1_seed42_frac3` | 498 | 7/15 | 9.1 | 0.8269 | 0.6031 | 0.6029 | 0.8106 | 0.5933 | 0.5746 |
| `v1_seed42_frac5` | 841 | 11/19 | 14.9 | 0.8359 | 0.6306 | 0.6198 | 0.8220 | 0.6238 | 0.6008 |
| `v1_seed42_frac10` | 1,687 | 9/17 | 19.8 | 0.8521 | 0.6561 | 0.6370 | 0.8332 | 0.6322 | 0.6095 |
| `v1_seed42_frac25` | 4,252 | 10/18 | 40.8 | 0.8781 | 0.7196 | 0.6777 | 0.8678 | 0.7082 | 0.6570 |
| `v1_seed42_frac50` | 8,521 | 7/15 | 63.4 | 0.8951 | 0.7500 | 0.7014 | 0.8883 | 0.7439 | 0.6855 |
| `v1_seed42_frac100` | 17,084 | 7/15 | 120.2 | 0.9090 | 0.7790 | 0.7179 | 0.8981 | 0.7662 | 0.7027 |
| `v2_seed0_frac1` | 171 | 30/30 | 17.6 | 0.7870 | 0.5253 | 0.5586 | 0.7668 | 0.4999 | 0.5353 |
| `v2_seed0_frac3` | 502 | 12/20 | 13.5 | 0.8265 | 0.5934 | 0.6019 | 0.8105 | 0.5815 | 0.5779 |
| `v2_seed0_frac5` | 844 | 8/16 | 14.6 | 0.8469 | 0.6336 | 0.6319 | 0.8332 | 0.6333 | 0.5987 |
| `v2_seed0_frac10` | 1,719 | 10/18 | 24.6 | 0.8697 | 0.6865 | 0.6606 | 0.8577 | 0.6829 | 0.6417 |
| `v2_seed0_frac25` | 4,241 | 6/14 | 39.0 | 0.8900 | 0.7295 | 0.6979 | 0.8781 | 0.7201 | 0.6688 |
| `v2_seed0_frac50` | 8,535 | 7/15 | 80.7 | 0.9015 | 0.7621 | 0.7049 | 0.8914 | 0.7491 | 0.6825 |
| `v2_seed0_frac100` | 17,084 | 6/14 | 153.3 | 0.9157 | 0.7874 | 0.7360 | 0.9059 | 0.7800 | 0.7059 |
| `v2_seed1_frac1` | 165 | 30/30 | 19.9 | 0.7938 | 0.5532 | 0.5667 | 0.7829 | 0.5538 | 0.5474 |
| `v2_seed1_frac3` | 517 | 17/25 | 19.0 | 0.8315 | 0.6193 | 0.6136 | 0.8189 | 0.6174 | 0.5886 |
| `v2_seed1_frac5` | 852 | 9/17 | 16.9 | 0.8484 | 0.6449 | 0.6249 | 0.8340 | 0.6479 | 0.6056 |
| `v2_seed1_frac10` | 1,699 | 10/18 | 26.2 | 0.8651 | 0.6860 | 0.6662 | 0.8567 | 0.6962 | 0.6400 |
| `v2_seed1_frac25` | 4,284 | 7/15 | 45.6 | 0.8899 | 0.7347 | 0.7028 | 0.8796 | 0.7401 | 0.6756 |
| `v2_seed1_frac50` | 8,575 | 6/14 | 80.7 | 0.9033 | 0.7536 | 0.7114 | 0.8955 | 0.7613 | 0.6933 |
| `v2_seed1_frac100` | 17,084 | 7/15 | 165.4 | 0.9145 | 0.7877 | 0.7348 | 0.9096 | 0.7919 | 0.7142 |
| `v2_seed42_frac1` | 164 | 30/30 | 19.0 | 0.8158 | 0.5741 | 0.5944 | 0.8044 | 0.5708 | 0.5713 |
| `v2_seed42_frac3` | 498 | 13/21 | 15.8 | 0.8454 | 0.6376 | 0.6331 | 0.8312 | 0.6401 | 0.6018 |
| `v2_seed42_frac5` | 841 | 10/18 | 17.3 | 0.8525 | 0.6639 | 0.6345 | 0.8390 | 0.6669 | 0.6161 |
| `v2_seed42_frac10` | 1,687 | 8/16 | 23.2 | 0.8729 | 0.7028 | 0.6674 | 0.8536 | 0.6934 | 0.6398 |
| `v2_seed42_frac25` | 4,252 | 7/15 | 41.5 | 0.8900 | 0.7402 | 0.6882 | 0.8819 | 0.7353 | 0.6777 |
| `v2_seed42_frac50` | 8,521 | 6/14 | 69.6 | 0.9031 | 0.7712 | 0.7169 | 0.8917 | 0.7557 | 0.6978 |
| `v2_seed42_frac100` | 17,084 | 6/14 | 142.2 | 0.9133 | 0.7842 | 0.7251 | 0.9077 | 0.7853 | 0.7205 |

## Classwise results at 100% training data

Mean ± sample standard deviation across the three seeds. Thresholds are selected
on validation and reused unchanged for test, so each model/class threshold is
shown on both split rows.

| Model | Split | Class | AUROC | AUPRC | F1 | Threshold |
|---|---|---|---|---|---|---|
| V1 | Validation | NORM | 0.9352 ± 0.0018 | 0.9156 ± 0.0025 | 0.8470 ± 0.0035 | 0.4294 ± 0.1095 |
| V1 | Validation | MI | 0.8966 ± 0.0023 | 0.7748 ± 0.0083 | 0.6990 ± 0.0072 | 0.3800 ± 0.1081 |
| V1 | Validation | STTC | 0.9225 ± 0.0019 | 0.7705 ± 0.0069 | 0.7413 ± 0.0027 | 0.2930 ± 0.0263 |
| V1 | Validation | CD | 0.8986 ± 0.0106 | 0.8085 ± 0.0135 | 0.7265 ± 0.0073 | 0.3189 ± 0.0848 |
| V1 | Validation | HYP | 0.8867 ± 0.0035 | 0.6226 ± 0.0038 | 0.5791 ± 0.0070 | 0.2127 ± 0.0648 |
| V1 | Test | NORM | 0.9264 ± 0.0017 | 0.8974 ± 0.0065 | 0.8361 ± 0.0060 | 0.4294 ± 0.1095 |
| V1 | Test | MI | 0.8860 ± 0.0055 | 0.7596 ± 0.0040 | 0.6887 ± 0.0090 | 0.3800 ± 0.1081 |
| V1 | Test | STTC | 0.9196 ± 0.0029 | 0.7872 ± 0.0016 | 0.7344 ± 0.0051 | 0.2930 ± 0.0263 |
| V1 | Test | CD | 0.8844 ± 0.0073 | 0.7916 ± 0.0066 | 0.6978 ± 0.0210 | 0.3189 ± 0.0848 |
| V1 | Test | HYP | 0.8796 ± 0.0071 | 0.6112 ± 0.0231 | 0.5725 ± 0.0128 | 0.2127 ± 0.0648 |
| V2 | Validation | NORM | 0.9377 ± 0.0024 | 0.9168 ± 0.0032 | 0.8543 ± 0.0043 | 0.3952 ± 0.0866 |
| V2 | Validation | MI | 0.9057 ± 0.0056 | 0.7881 ± 0.0138 | 0.7198 ± 0.0055 | 0.3874 ± 0.1914 |
| V2 | Validation | STTC | 0.9277 ± 0.0022 | 0.7807 ± 0.0047 | 0.7499 ± 0.0054 | 0.4111 ± 0.1891 |
| V2 | Validation | CD | 0.9127 ± 0.0053 | 0.8271 ± 0.0087 | 0.7404 ± 0.0160 | 0.4599 ± 0.1681 |
| V2 | Validation | HYP | 0.8887 ± 0.0050 | 0.6195 ± 0.0128 | 0.5954 ± 0.0149 | 0.4842 ± 0.2035 |
| V2 | Test | NORM | 0.9340 ± 0.0009 | 0.9103 ± 0.0019 | 0.8454 ± 0.0043 | 0.3952 ± 0.0866 |
| V2 | Test | MI | 0.8919 ± 0.0096 | 0.7679 ± 0.0176 | 0.6804 ± 0.0189 | 0.3874 ± 0.1914 |
| V2 | Test | STTC | 0.9259 ± 0.0026 | 0.8102 ± 0.0063 | 0.7405 ± 0.0087 | 0.4111 ± 0.1891 |
| V2 | Test | CD | 0.8979 ± 0.0040 | 0.8002 ± 0.0057 | 0.7087 ± 0.0131 | 0.4599 ± 0.1681 |
| V2 | Test | HYP | 0.8888 ± 0.0044 | 0.6401 ± 0.0181 | 0.5927 ± 0.0301 | 0.4842 ± 0.2035 |

## Interpretation

- Cross-seed mean validation and test metrics increase with training fraction
  for both architectures.
- V2 has higher cross-seed mean test AUROC, AUPRC, and validation-threshold F1
  than V1 at every fraction. The advantage is largest at 1% (AUROC +0.035,
  AUPRC +0.055, F1 +0.038) and smaller at 100% (+0.009, +0.016, +0.008).
- At 100%, V2 reaches test AUROC 0.908 ± 0.002, AUPRC 0.786 ± 0.006, and F1
  0.714 ± 0.007. V1 reaches 0.899 ± 0.003, 0.769 ± 0.007, and 0.706 ± 0.005.
- NORM is the strongest 100% class for both architectures. HYP remains the
  weakest by AUPRC and F1.
- Threshold spread is substantial for several V2 classes, especially MI, STTC,
  CD, and HYP. Thresholded F1 is therefore less stable than ranking metrics and
  should not be treated as a calibrated operating point.

## Limitations and reproducibility

- Three seeds provide a descriptive spread, not a precise confidence interval.
  Below 100%, seed changes both the sampled patients and optimization. At 100%,
  the training patients are identical and spread primarily reflects optimization.
- PyTorch emitted a warning that CUDA memory-efficient attention used a
  nondeterministic backward algorithm. Seeds and deterministic flags were set,
  but exact bitwise reproduction is not guaranteed on this GPU/software stack.
- Validation controls checkpoint selection, early stopping, and thresholds, so
  validation metrics are optimistic relative to test metrics.
- AUROC/AUPRC are threshold-free; F1 uses per-class thresholds selected on the
  same validation set used for checkpoint selection.
- The six self-contained run directories occupy about 5.9 GB because each
  stores its own waveform cache. `artifacts/` is Git-ignored and must be backed
  up separately; this tracked file is only the high-level summary.
- Earlier single-seed notebook runs used different V1/V2 budgets and evaluation
  policies. They remain historical provenance, but the common-protocol matrix
  above supersedes them for architecture comparisons.
