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

`splits/seed_<seed>/` holds patient-level PTB-XL splits for seeds 0, 1, and 42:
`train_{1,3,5,10,25,50,100}.csv` (folds 1–8, nested), `val.csv` (fold 9),
and `test.csv` (fold 10).

## CNN-BiLSTM baseline

The compact CNN-BiLSTM lives in `src/cnn/`. See the
[CNN README](src/cnn/README.md) for its architecture, commands, CLI, and output
format, and the [CNN experiment summary](src/cnn/scratch.md) for verified
three-seed results. From the repository root:

```bash
python -m src.cnn.run --help
python -m src.cnn.run --seed 42 --train-fraction 5
```

Generated CNN runs go to `artifacts/cnn/` and are ignored by Git.

## Classical ML baselines

Random forest, XGBoost, and RBF-SVM on 300-d wavelet features live in
`src/classical_ml/`. See the [classical ML README](src/classical_ml/README.md) for
commands, flags, and outputs, and its
[experiment summary](src/classical_ml/scratch.md) for results. From the repository
root:

```bash
python -m src.classical_ml.features     # builds the feature file
python -m src.classical_ml.run --help
```

Outputs go to `artifacts/classical_ml/` (git-ignored).
