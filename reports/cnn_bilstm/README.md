# CNN-BiLSTM experiment results

This directory contains the lightweight, reproducibility-focused artifacts from
the seed 42 PTB-XL learning-curve rerun on PACE. Each training fraction retains
the exact run configuration, epoch history, and final validation/test metrics.

The consolidated results are in `seed_42/summary.csv`. Per-class AUROC, AUPRC,
F1, validation-selected thresholds, positive counts, and prevalence are stored
inside each fraction's `metrics.json`.

Model checkpoints and validation/test prediction CSVs remain under the ignored
`outputs/` directory and are intentionally not tracked in Git.
