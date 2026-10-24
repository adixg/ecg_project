# ECG learning curves on PTB-XL

## Setup

```bash
git clone https://github.com/adixg/ecg_project.git
cd ecg_project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

gh release download v1.0-data   # needs GitHub CLI, logged in (gh auth login)
mkdir -p data/ptbxl && tar -xzf ptbxl_records100.tar.gz -C data/ptbxl

python check_data.py   # add --full to read every record
```

To use data stored elsewhere, set `PTBXL_ROOT` to the directory containing `records100/`.

## Splits

`splits/` holds patient-level PTB-XL splits: `train_{5,10,25,50,100}.csv` (folds 1–8, nested), `val.csv` (fold 9), `test.csv` (fold 10).
