# Scratch: recorded Transformer run metadata

Snapshot of the completed notebook runs. Checkpoints were selected by **validation
macro AUROC**, not by maximizing test metrics. These are historical results;
the refactor checks do not replace or extend these benchmark runs.

## Shared dataset and settings

- Dataset: PTB-XL records100; 100 Hz, 10 seconds, 12 ordered leads; raw mV input.
- Task: five independent labels, ordered NORM, MI, STTC, CD, HYP.
- Validation: fold 9, 2,146 ECGs / 1,917 patients.
- Test: fold 10, 2,158 ECGs / 1,877 patients.
- Seed: 42; device: CPU; PyTorch 2.14.1+cpu.
- Embedding dimension 128; four heads; three encoder layers; dropout 0.1.
- AdamW: learning rate 0.0003, weight decay 0.01; unweighted BCE.
- Normalization: per-lead population mean/std fitted on each training subset.
- AUROC/AP use probabilities. AUPRC means average precision.
- Frozen CSV labels retained. Validation ECG 3799 has MI=0 in the split,
  while local diagnostic-code presence maps to MI=1.

## V1: linear patch Transformer

- Run: `artifacts/transformer_v1/train5_seed42_20261024T184430993193Z`.
- Model: `ECGTransformerV1`; 632,965 trainable parameters.
- Embedding: 20-sample patches, 50 tokens, learned positional embedding.
- Training: 5%, 841 ECGs / 742 patients; batch 32; five epochs; constant LR.
- Selected epoch: 5.
- F1 threshold: fixed 0.5. **Test evaluation was not run.**

| Validation macro AUROC | Validation macro AUPRC | Validation macro F1 at 0.5 |
|---:|---:|---:|
| 0.835023 | 0.638649 | 0.521232 |

| Class | Validation AUROC | Validation AUPRC | Validation F1 at 0.5 |
|---|---:|---:|---:|
| NORM | 0.890017 | 0.843872 | 0.804243 |
| MI | 0.803180 | 0.613051 | 0.523044 |
| STTC | 0.874204 | 0.675280 | 0.657706 |
| CD | 0.809273 | 0.656108 | 0.497878 |
| HYP | 0.798442 | 0.404934 | 0.123288 |

## V2: convolutional Transformer

- Run: `artifacts/transformer_v2/20261027T020054956545Z_seed42`.
- Model: `ECGTransformerV2`; 650,309 trainable parameters.
- Embedding: 50 convolutional tokens, 31-sample receptive field,
  20-sample step, sinusoidal positional encoding.
- Batch 64; maximum 30 epochs; patience 8; minimum improvement 0.0001.
- Cosine LR schedule to 0.00003; classifier bias initialized to training log-odds.
- Thresholds: one per class, selected by validation F1 and frozen for test.
- All three checkpoints were selected before test inference.

| Training % | ECGs | Patients | Epochs run | Selected epoch | Validation AUROC | Validation AUPRC | Test AUROC | Test AUPRC | Test F1 at 0.5 | Test F1, validation thresholds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1687 | 1483 | 16 | 8 | 0.872354 | 0.700326 | 0.855199 | 0.694983 | 0.599032 | 0.639854 |
| 25 | 4252 | 3706 | 15 | 7 | 0.888793 | 0.738247 | 0.882454 | 0.732783 | 0.637265 | 0.679040 |
| 50 | 8521 | 7412 | 14 | 6 | 0.904205 | 0.772517 | 0.893620 | 0.758127 | 0.677487 | 0.694319 |

### Classwise test metrics at the selected checkpoints

| Training % | Class | AUROC | AUPRC | F1 at 0.5 | F1, validation threshold | Threshold |
|---|---|---:|---:|---:|---:|---:|
| 10 | NORM | 0.899002 | 0.869874 | 0.804173 | 0.803411 | 0.499738127 |
| 10 | MI | 0.829594 | 0.661109 | 0.601156 | 0.604385 | 0.490601867 |
| 10 | STTC | 0.888061 | 0.725913 | 0.605442 | 0.666048 | 0.236475632 |
| 10 | CD | 0.836230 | 0.721579 | 0.627451 | 0.629787 | 0.458847523 |
| 10 | HYP | 0.823109 | 0.496439 | 0.356941 | 0.495637 | 0.182218134 |
| 25 | NORM | 0.920152 | 0.890855 | 0.826370 | 0.827688 | 0.699363172 |
| 25 | MI | 0.860570 | 0.708932 | 0.643510 | 0.647931 | 0.270526677 |
| 25 | STTC | 0.910142 | 0.772757 | 0.692063 | 0.715613 | 0.304442763 |
| 25 | CD | 0.869636 | 0.760994 | 0.617347 | 0.680988 | 0.220123097 |
| 25 | HYP | 0.851773 | 0.530376 | 0.407035 | 0.522979 | 0.210293442 |
| 50 | NORM | 0.923350 | 0.893697 | 0.832932 | 0.833173 | 0.506014407 |
| 50 | MI | 0.871735 | 0.719987 | 0.609442 | 0.676617 | 0.206316099 |
| 50 | STTC | 0.920119 | 0.796918 | 0.735382 | 0.742601 | 0.327233493 |
| 50 | CD | 0.882230 | 0.783954 | 0.690632 | 0.689441 | 0.434698582 |
| 50 | HYP | 0.870668 | 0.596078 | 0.519048 | 0.529762 | 0.155859366 |

### Source provenance

- Local data path: `data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1`.
- Metadata SHA256: `cd9bf4e6445c5e022605857c4d6f90c90ff0e62df847da52efaebe94de786156`.
- SCP mapping SHA256: `ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f`.

| Split | SHA256 |
|---|---|
| train_10 | `8288568bd73ea37689361a14d5c99b9b85f070e1fa073660d9568108bd23c22b` |
| train_25 | `ebe304eb8fccbef69f21945ab68fedfd93b7144b4f3bcb75716c9f18f43f2b16` |
| train_50 | `740872d80edbbb71a485b96981351c4d195231d0ee56a648b13622c5e966ea84` |
| val | `c74e76fc396a22f32fecd2913d8f7392a3ec220657570b3aa195d80592d89341` |
| test | `6db1571d74074dc196f22d7ee16f66555601b46e8325134173a6de83049ccf3e` |

### Interpretation

More training patients improved the three test macro metrics in this seed.
The 50% test scores fell below the original expected ranges (AUROC 0.90–0.94,
AUPRC 0.77–0.86, thresholded F1 0.72–0.81). NORM was strongest; HYP remained
the weakest by AUPRC and F1. Validation BCE rose after the early epochs,
supporting early checkpoint selection. These results do not establish that
adding layers or heads would help, or that a Transformer beats a 1D CNN.
No seed variability or confidence intervals were measured.
