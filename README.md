# ECG learning curves on PTB-XL

## Setup

```bash
git clone https://github.com/adixg/ecg_project.git
cd ecg_project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

gh release download v1.0-data   # needs GitHub CLI, logged in (gh auth login)
mkdir -p data/ptbxl && tar -xzf ptbxl_records100.tar.gz -C data/ptbxl
```

To use data stored elsewhere, set `PTBXL_ROOT` to the directory containing `records100/`.

## Splits

`splits/` holds patient-level PTB-XL splits: `train_{5,10,25,50,100}.csv` (folds 1–8, nested), `val.csv` (fold 9), `test.csv` (fold 10).

## CNN-BiLSTM baseline

The initial model is intentionally compact: three 1D convolution blocks reduce
the sequence from 1,000 to 125 time steps, followed by a single bidirectional
LSTM, temporal mean pooling, dropout, and a five-logit classification head.

Run training from the repository root. For example, on a PACE compute node:

```bash
export PTBXL_ROOT=/path/to/ptbxl_records100
python -m src.train \
  --train-fraction 5 \
  --seed 42 \
  --device cuda \
  --output-dir outputs/cnn_bilstm
```

The default input is the physical-unit signal returned by WFDB, without
per-record normalization. Pass `--normalize` to enable per-lead, per-record
standardization. Class weighting is disabled by default; `--use-pos-weight`
uses weights computed from `train_100.csv` so the weights remain constant
across training fractions.

Each run writes its configuration, best checkpoint, epoch history, validation
and test metrics, selected validation thresholds, and prediction CSVs under a
fraction-and-seed-specific output directory. The final metrics include both the
overall macro AUROC, AUPRC, and F1 and class-specific results for NORM, MI,
STTC, CD, and HYP.
