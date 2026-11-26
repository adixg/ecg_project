# CNN-BiLSTM experiment summary

Verified runs: CNN-BiLSTM at 1, 3, 5, 10, 25, 50, and 100% of the training
data, seeds 0, 1, and 42 (21 runs). Checkpoints were selected by **validation
macro AUROC**, and class thresholds were selected by validation F1; test metrics
were never used for selection.

## Provenance and settings

- Artifacts: `artifacts/cnn_restructured/cnn_bilstm_seed<seed>_fraction<fraction>/`
- Seeds: 0, 1, and 42
- Training fractions: 1%, 3%, 5%, 10%, 25%, 50%, and 100%
- Dataset: PTB-XL `records100`, 12 leads, 100 Hz, 1,000 samples, and five
  diagnostic superclasses. Validation is fold 9 (2,146 ECGs), and test is fold
  10 (2,158 ECGs).
- Model: three 1D convolution blocks `(32, 64, 96)`, one bidirectional LSTM
  with 64 hidden units per direction, mean pooling, dropout 0.3, and a
  five-logit head
- Runtime: CUDA on a Georgia Tech PACE node with an NVIDIA A40; PyTorch
  2.14.1+cu126
- Checkpoint: epoch with highest validation macro AUROC; patience 8 and maximum
  50 epochs
- Thresholds: selected independently for each class on validation F1, then
  frozen for test evaluation
- Inputs: physical-unit WFDB signals without per-record normalization
- Loss: unweighted `BCEWithLogitsLoss`
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001
- Batch size: 64; data-loader workers: 8; no learning-rate schedule

All summary values below are the mean plus or minus the sample standard
deviation across the three seeds (`n = 3`). The recorded training time excludes
final validation/test inference and artifact writing. Across all 21 runs, the
recorded training portion took 3,123 seconds, or approximately 52 minutes. All
21 run directories and their saved configuration, validation, test, threshold,
epoch-history, and checkpoint artifacts were checked before aggregation.

### Aggregate results

| Fraction | Best epoch | Training time | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 25.0 ± 10.4 | 69.9 ± 19.9 s | 0.7994 ± 0.0165 | 0.5535 ± 0.0326 | 0.5729 ± 0.0186 | 0.7885 ± 0.0148 | 0.5471 ± 0.0374 | 0.5582 ± 0.0184 |
| 3% | 22.3 ± 6.4 | 65.8 ± 12.7 s | 0.8496 ± 0.0060 | 0.6503 ± 0.0206 | 0.6304 ± 0.0086 | 0.8360 ± 0.0046 | 0.6396 ± 0.0161 | 0.6077 ± 0.0073 |
| 5% | 21.7 ± 4.0 | 73.3 ± 9.7 s | 0.8702 ± 0.0018 | 0.6934 ± 0.0041 | 0.6616 ± 0.0017 | 0.8600 ± 0.0020 | 0.6926 ± 0.0098 | 0.6376 ± 0.0043 |
| 10% | 23.0 ± 4.6 | 98.7 ± 14.4 s | 0.8939 ± 0.0037 | 0.7383 ± 0.0083 | 0.7016 ± 0.0087 | 0.8839 ± 0.0051 | 0.7361 ± 0.0105 | 0.6798 ± 0.0056 |
| 25% | 17.0 ± 5.0 | 125.0 ± 18.2 s | 0.9118 ± 0.0018 | 0.7803 ± 0.0043 | 0.7206 ± 0.0019 | 0.9018 ± 0.0010 | 0.7743 ± 0.0043 | 0.7024 ± 0.0048 |
| 50% | 22.7 ± 1.5 | 242.3 ± 4.2 s | 0.9230 ± 0.0008 | 0.8046 ± 0.0035 | 0.7389 ± 0.0042 | 0.9163 ± 0.0009 | 0.8023 ± 0.0024 | 0.7260 ± 0.0023 |
| 100% | 17.7 ± 3.2 | 366.1 ± 49.5 s | 0.9302 ± 0.0008 | 0.8170 ± 0.0014 | 0.7518 ± 0.0026 | 0.9243 ± 0.0018 | 0.8148 ± 0.0026 | 0.7397 ± 0.0079 |

### Individual runs

| Seed | Fraction | ECGs | Best epoch | Time | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1% | 171 | 19 | 59.9s | 0.7872 | 0.5228 | 0.5579 | 0.7726 | 0.5042 | 0.5377 |
| 0 | 3% | 502 | 15 | 51.2s | 0.8462 | 0.6332 | 0.6270 | 0.8354 | 0.6217 | 0.6024 |
| 0 | 5% | 844 | 21 | 73.6s | 0.8712 | 0.6901 | 0.6599 | 0.8621 | 0.6839 | 0.6389 |
| 0 | 10% | 1,719 | 22 | 99.8s | 0.8982 | 0.7471 | 0.7076 | 0.8882 | 0.7475 | 0.6852 |
| 0 | 25% | 4,241 | 12 | 109.1s | 0.9137 | 0.7830 | 0.7223 | 0.9016 | 0.7705 | 0.6995 |
| 0 | 50% | 8,535 | 21 | 241.8s | 0.9222 | 0.8043 | 0.7388 | 0.9161 | 0.8019 | 0.7239 |
| 0 | 100% | 17,084 | 20 | 409.0s | 0.9305 | 0.8180 | 0.7538 | 0.9247 | 0.8157 | 0.7387 |
| 1 | 1% | 165 | 37 | 92.7s | 0.7928 | 0.5499 | 0.5669 | 0.7912 | 0.5730 | 0.5633 |
| 1 | 3% | 517 | 25 | 71.8s | 0.8460 | 0.6446 | 0.6240 | 0.8318 | 0.6440 | 0.6047 |
| 1 | 5% | 852 | 26 | 82.9s | 0.8681 | 0.6922 | 0.6632 | 0.8596 | 0.7032 | 0.6411 |
| 1 | 10% | 1,699 | 28 | 112.5s | 0.8916 | 0.7372 | 0.7055 | 0.8852 | 0.7339 | 0.6803 |
| 1 | 25% | 4,284 | 17 | 120.9s | 0.9100 | 0.7754 | 0.7210 | 0.9009 | 0.7789 | 0.7079 |
| 1 | 50% | 8,575 | 23 | 238.3s | 0.9230 | 0.8013 | 0.7348 | 0.9173 | 0.8002 | 0.7285 |
| 1 | 100% | 17,084 | 14 | 312.0s | 0.9293 | 0.8154 | 0.7527 | 0.9223 | 0.8119 | 0.7322 |
| 42 | 1% | 164 | 19 | 57.0s | 0.8182 | 0.5878 | 0.5937 | 0.8018 | 0.5640 | 0.5734 |
| 42 | 3% | 498 | 27 | 74.4s | 0.8565 | 0.6733 | 0.6401 | 0.8409 | 0.6530 | 0.6160 |
| 42 | 5% | 841 | 18 | 63.4s | 0.8713 | 0.6981 | 0.6618 | 0.8583 | 0.6908 | 0.6328 |
| 42 | 10% | 1,687 | 19 | 83.8s | 0.8920 | 0.7307 | 0.6916 | 0.8783 | 0.7267 | 0.6739 |
| 42 | 25% | 4,252 | 22 | 144.8s | 0.9116 | 0.7825 | 0.7186 | 0.9028 | 0.7736 | 0.6998 |
| 42 | 50% | 8,521 | 24 | 246.7s | 0.9237 | 0.8083 | 0.7432 | 0.9155 | 0.8049 | 0.7257 |
| 42 | 100% | 17,084 | 19 | 377.3s | 0.9308 | 0.8176 | 0.7488 | 0.9258 | 0.8168 | 0.7480 |

### Validation classwise results

Thresholds are validation-selected and therefore identical in this table and
the test table. Their mean and standard deviation summarize variation across
the three seeds; no threshold was fit on test data.

| Fraction | Class | AUROC | AUPRC | F1 | Threshold |
|---:|---|---:|---:|---:|---:|
| 1% | NORM | 0.8612 ± 0.0173 | 0.8117 ± 0.0161 | 0.7714 ± 0.0215 | 0.2833 ± 0.0289 |
| 1% | MI | 0.7900 ± 0.0107 | 0.5138 ± 0.0075 | 0.5700 ± 0.0169 | 0.2333 ± 0.1607 |
| 1% | STTC | 0.8056 ± 0.0525 | 0.5497 ± 0.1169 | 0.5731 ± 0.0501 | 0.2667 ± 0.0764 |
| 1% | CD | 0.8136 ± 0.0149 | 0.6042 ± 0.0278 | 0.5871 ± 0.0154 | 0.3667 ± 0.1041 |
| 1% | HYP | 0.7267 ± 0.0138 | 0.2881 ± 0.0144 | 0.3627 ± 0.0134 | 0.2000 ± 0.0866 |
| 3% | NORM | 0.8960 ± 0.0021 | 0.8590 ± 0.0026 | 0.8084 ± 0.0022 | 0.3500 ± 0.2000 |
| 3% | MI | 0.8178 ± 0.0046 | 0.6029 ± 0.0123 | 0.5950 ± 0.0060 | 0.2500 ± 0.0500 |
| 3% | STTC | 0.8680 ± 0.0167 | 0.6670 ± 0.0368 | 0.6479 ± 0.0249 | 0.2167 ± 0.0289 |
| 3% | CD | 0.8412 ± 0.0087 | 0.6957 ± 0.0224 | 0.6427 ± 0.0318 | 0.3833 ± 0.0289 |
| 3% | HYP | 0.8248 ± 0.0255 | 0.4271 ± 0.0693 | 0.4579 ± 0.0290 | 0.1500 ± 0.0500 |
| 5% | NORM | 0.9080 ± 0.0049 | 0.8719 ± 0.0063 | 0.8218 ± 0.0083 | 0.4667 ± 0.2363 |
| 5% | MI | 0.8396 ± 0.0034 | 0.6573 ± 0.0211 | 0.6280 ± 0.0108 | 0.3167 ± 0.1258 |
| 5% | STTC | 0.8897 ± 0.0069 | 0.7018 ± 0.0267 | 0.6759 ± 0.0100 | 0.1833 ± 0.0577 |
| 5% | CD | 0.8605 ± 0.0088 | 0.7365 ± 0.0123 | 0.6751 ± 0.0103 | 0.4333 ± 0.0764 |
| 5% | HYP | 0.8533 ± 0.0102 | 0.4997 ± 0.0042 | 0.5073 ± 0.0119 | 0.1500 ± 0.1000 |
| 10% | NORM | 0.9244 ± 0.0017 | 0.8941 ± 0.0025 | 0.8394 ± 0.0042 | 0.4000 ± 0.2179 |
| 10% | MI | 0.8847 ± 0.0010 | 0.7478 ± 0.0046 | 0.6893 ± 0.0100 | 0.2500 ± 0.1000 |
| 10% | STTC | 0.9125 ± 0.0023 | 0.7429 ± 0.0117 | 0.7298 ± 0.0023 | 0.3833 ± 0.0764 |
| 10% | CD | 0.8876 ± 0.0042 | 0.7808 ± 0.0068 | 0.7109 ± 0.0148 | 0.4500 ± 0.1323 |
| 10% | HYP | 0.8606 ± 0.0131 | 0.5259 ± 0.0282 | 0.5383 ± 0.0189 | 0.3000 ± 0.2000 |
| 25% | NORM | 0.9363 ± 0.0014 | 0.9175 ± 0.0012 | 0.8459 ± 0.0042 | 0.2667 ± 0.0764 |
| 25% | MI | 0.9018 ± 0.0060 | 0.7822 ± 0.0098 | 0.7122 ± 0.0136 | 0.4333 ± 0.1258 |
| 25% | STTC | 0.9242 ± 0.0014 | 0.7721 ± 0.0039 | 0.7371 ± 0.0067 | 0.2667 ± 0.1041 |
| 25% | CD | 0.9085 ± 0.0016 | 0.8158 ± 0.0017 | 0.7398 ± 0.0099 | 0.5333 ± 0.2363 |
| 25% | HYP | 0.8880 ± 0.0065 | 0.6140 ± 0.0115 | 0.5681 ± 0.0167 | 0.3000 ± 0.1323 |
| 50% | NORM | 0.9409 ± 0.0023 | 0.9246 ± 0.0031 | 0.8525 ± 0.0037 | 0.3000 ± 0.0866 |
| 50% | MI | 0.9224 ± 0.0049 | 0.8197 ± 0.0087 | 0.7333 ± 0.0084 | 0.5667 ± 0.1041 |
| 50% | STTC | 0.9280 ± 0.0019 | 0.7886 ± 0.0012 | 0.7493 ± 0.0069 | 0.3000 ± 0.0500 |
| 50% | CD | 0.9238 ± 0.0028 | 0.8459 ± 0.0051 | 0.7595 ± 0.0073 | 0.3833 ± 0.1041 |
| 50% | HYP | 0.8997 ± 0.0046 | 0.6444 ± 0.0045 | 0.6001 ± 0.0116 | 0.2833 ± 0.0577 |
| 100% | NORM | 0.9472 ± 0.0004 | 0.9307 ± 0.0025 | 0.8628 ± 0.0049 | 0.3833 ± 0.1041 |
| 100% | MI | 0.9299 ± 0.0069 | 0.8372 ± 0.0131 | 0.7553 ± 0.0124 | 0.5167 ± 0.0764 |
| 100% | STTC | 0.9336 ± 0.0010 | 0.7981 ± 0.0056 | 0.7637 ± 0.0054 | 0.3333 ± 0.0577 |
| 100% | CD | 0.9346 ± 0.0008 | 0.8597 ± 0.0028 | 0.7716 ± 0.0070 | 0.4833 ± 0.0577 |
| 100% | HYP | 0.9057 ± 0.0028 | 0.6593 ± 0.0032 | 0.6054 ± 0.0113 | 0.3000 ± 0.1803 |

### Test classwise results

| Fraction | Class | AUROC | AUPRC | F1 | Threshold |
|---:|---|---:|---:|---:|---:|
| 1% | NORM | 0.8491 ± 0.0138 | 0.7977 ± 0.0123 | 0.7619 ± 0.0092 | 0.2833 ± 0.0289 |
| 1% | MI | 0.7627 ± 0.0142 | 0.4938 ± 0.0158 | 0.5364 ± 0.0174 | 0.2333 ± 0.1607 |
| 1% | STTC | 0.7993 ± 0.0484 | 0.5549 ± 0.1249 | 0.5711 ± 0.0460 | 0.2667 ± 0.0764 |
| 1% | CD | 0.7872 ± 0.0072 | 0.5988 ± 0.0125 | 0.5636 ± 0.0225 | 0.3667 ± 0.1041 |
| 1% | HYP | 0.7442 ± 0.0052 | 0.2901 ± 0.0416 | 0.3578 ± 0.0095 | 0.2000 ± 0.0866 |
| 3% | NORM | 0.8823 ± 0.0035 | 0.8384 ± 0.0098 | 0.7911 ± 0.0011 | 0.3500 ± 0.2000 |
| 3% | MI | 0.7922 ± 0.0038 | 0.5734 ± 0.0034 | 0.5552 ± 0.0047 | 0.2500 ± 0.0500 |
| 3% | STTC | 0.8614 ± 0.0138 | 0.6653 ± 0.0207 | 0.6265 ± 0.0245 | 0.2167 ± 0.0289 |
| 3% | CD | 0.8196 ± 0.0119 | 0.6803 ± 0.0116 | 0.6139 ± 0.0254 | 0.3833 ± 0.0289 |
| 3% | HYP | 0.8246 ± 0.0204 | 0.4406 ± 0.0821 | 0.4518 ± 0.0288 | 0.1500 ± 0.0500 |
| 5% | NORM | 0.8977 ± 0.0038 | 0.8604 ± 0.0101 | 0.8024 ± 0.0066 | 0.4667 ± 0.2363 |
| 5% | MI | 0.8233 ± 0.0033 | 0.6363 ± 0.0195 | 0.5869 ± 0.0124 | 0.3167 ± 0.1258 |
| 5% | STTC | 0.8868 ± 0.0064 | 0.7287 ± 0.0210 | 0.6684 ± 0.0113 | 0.1833 ± 0.0577 |
| 5% | CD | 0.8419 ± 0.0113 | 0.7089 ± 0.0043 | 0.6319 ± 0.0193 | 0.4333 ± 0.0764 |
| 5% | HYP | 0.8503 ± 0.0013 | 0.5288 ± 0.0237 | 0.4984 ± 0.0111 | 0.1500 ± 0.1000 |
| 10% | NORM | 0.9159 ± 0.0029 | 0.8838 ± 0.0089 | 0.8267 ± 0.0053 | 0.4000 ± 0.2179 |
| 10% | MI | 0.8692 ± 0.0061 | 0.7223 ± 0.0107 | 0.6624 ± 0.0107 | 0.2500 ± 0.1000 |
| 10% | STTC | 0.9094 ± 0.0039 | 0.7549 ± 0.0230 | 0.7233 ± 0.0139 | 0.3833 ± 0.0764 |
| 10% | CD | 0.8687 ± 0.0080 | 0.7585 ± 0.0095 | 0.6742 ± 0.0182 | 0.4500 ± 0.1323 |
| 10% | HYP | 0.8563 ± 0.0062 | 0.5608 ± 0.0171 | 0.5124 ± 0.0114 | 0.3000 ± 0.2000 |
| 25% | NORM | 0.9288 ± 0.0019 | 0.9038 ± 0.0033 | 0.8388 ± 0.0101 | 0.2667 ± 0.0764 |
| 25% | MI | 0.8917 ± 0.0027 | 0.7691 ± 0.0059 | 0.6836 ± 0.0067 | 0.4333 ± 0.1258 |
| 25% | STTC | 0.9227 ± 0.0017 | 0.7972 ± 0.0133 | 0.7344 ± 0.0126 | 0.2667 ± 0.1041 |
| 25% | CD | 0.8898 ± 0.0053 | 0.7873 ± 0.0106 | 0.7054 ± 0.0139 | 0.5333 ± 0.2363 |
| 25% | HYP | 0.8758 ± 0.0044 | 0.6144 ± 0.0159 | 0.5499 ± 0.0312 | 0.3000 ± 0.1323 |
| 50% | NORM | 0.9409 ± 0.0012 | 0.9215 ± 0.0012 | 0.8492 ± 0.0037 | 0.3000 ± 0.0866 |
| 50% | MI | 0.9166 ± 0.0028 | 0.8144 ± 0.0060 | 0.7239 ± 0.0008 | 0.5667 ± 0.1041 |
| 50% | STTC | 0.9283 ± 0.0004 | 0.8076 ± 0.0066 | 0.7489 ± 0.0108 | 0.3000 ± 0.0500 |
| 50% | CD | 0.9043 ± 0.0032 | 0.8227 ± 0.0083 | 0.7272 ± 0.0149 | 0.3833 ± 0.1041 |
| 50% | HYP | 0.8914 ± 0.0029 | 0.6454 ± 0.0137 | 0.5810 ± 0.0170 | 0.2833 ± 0.0577 |
| 100% | NORM | 0.9453 ± 0.0012 | 0.9256 ± 0.0026 | 0.8592 ± 0.0024 | 0.3833 ± 0.1041 |
| 100% | MI | 0.9225 ± 0.0068 | 0.8297 ± 0.0093 | 0.7256 ± 0.0110 | 0.5167 ± 0.0764 |
| 100% | STTC | 0.9343 ± 0.0008 | 0.8193 ± 0.0048 | 0.7718 ± 0.0027 | 0.3333 ± 0.0577 |
| 100% | CD | 0.9177 ± 0.0018 | 0.8406 ± 0.0038 | 0.7458 ± 0.0053 | 0.4833 ± 0.0577 |
| 100% | HYP | 0.9015 ± 0.0032 | 0.6589 ± 0.0060 | 0.5958 ± 0.0246 | 0.3000 ± 0.1803 |

## Findings

- Performance improved monotonically with training fraction for macro AUROC,
  AUPRC, and F1 on every individual seed, not only after averaging.
- The largest gains occurred in the low-data regime. Mean test AUPRC increased
  by 0.0925 from 1% to 3%, by 0.0530 from 3% to 5%, and by only 0.0125 from
  50% to 100%.
- Seed sensitivity was greatest at 1%. Test macro AUPRC had a standard
  deviation of 0.0374 at 1%, falling below 0.011 from 5% onward.
- STTC was especially unstable at 1% (test AUPRC 0.5549 ± 0.1249), but its
  variability dropped sharply as the training set grew.
- At 100%, the mean test metrics were 0.9243 AUROC, 0.8148 AUPRC, and 0.7397
  F1. NORM was the strongest class by AUPRC (0.9256), while HYP remained the
  most difficult (0.6589).
- Despite being the weakest class, HYP's 100% AUPRC was approximately 5.4 times
  its test prevalence (0.1214), well above a random ranking baseline.
- For context, at 100% the CNN-BiLSTM's mean test AUROC (0.9243) exceeded the
  best classical mean, SVM at 0.8532, by 0.0711. Its mean test AUPRC (0.8148)
  exceeded the best classical mean, XGBoost at 0.6635, by 0.1513. These are
  descriptive comparisons, not controlled ablations.

## Limitations

- Three seeds quantify variability better than a single run, but they are still
  a small sample; the reported standard deviations are descriptive rather than
  precise confidence intervals.
- For fractions below 100%, changing the seed changes both the sampled patient
  subset and the model optimization. At 100%, the training patients are the
  same, so variation is primarily from initialization, shuffling, and CUDA
  execution.
- The fractions are nested within each seed but not necessarily across seeds.
- The model uses raw physical-unit signals. Other model families may use
  different normalization and selection procedures, so cross-family comparisons
  should account for preprocessing and tuning differences. In particular, the
  classical models use per-record z-scored wavelet features and validation grid
  searches, whereas the CNN uses fixed hyperparameters and early stopping.
- Threshold-based F1 depends on the validation-selected operating point. AUROC
  and AUPRC are threshold-independent and are better suited to comparing the
  learning curves directly.
