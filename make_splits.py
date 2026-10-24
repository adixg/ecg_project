
import os
import ast
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["PTBXL_ROOT"])
OUT = Path("data/splits")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SEED = 42

# Load metadata
df = pd.read_csv(ROOT / "ptbxl_database.csv")
scp = pd.read_csv(ROOT / "scp_statements.csv", index_col=0)

# Map diagnostic codes to five superclasses
mapping = (
    scp[scp["diagnostic"] == 1]["diagnostic_class"]
    .dropna()
    .to_dict()
)

def get_labels(raw):
    codes = ast.literal_eval(raw)
    labels = {mapping[c] for c in codes if c in mapping}
    return [int(c in labels) for c in CLASSES]

df[CLASSES] = pd.DataFrame(
    df["scp_codes"].map(get_labels).tolist(),
    index=df.index,
    columns=CLASSES,
)

# Exclude records without mapped diagnostic superclass labels
df = df[df[CLASSES].sum(axis=1) > 0].copy()

# Official PTB-XL folds
train = df[df.strat_fold <= 8]
val = df[df.strat_fold == 9]
test = df[df.strat_fold == 10]

# Check for patient leakage
assert set(train.patient_id).isdisjoint(val.patient_id)
assert set(train.patient_id).isdisjoint(test.patient_id)
assert set(val.patient_id).isdisjoint(test.patient_id)

# Deterministic patient ordering
patients = np.random.default_rng(SEED).permutation(
    np.sort(train.patient_id.unique())
)

columns = [
    "ecg_id", "patient_id", "filename_lr",
    "strat_fold", *CLASSES
]

def save(data, name):
    path = OUT / f"{name}.csv"
    data[columns].sort_values("ecg_id").to_csv(
        path, index=False
    )
    print(f"{name}: {len(data)} ECGs")
    print(data[CLASSES].sum().to_dict())

# Nested patient subsets
for fraction in [5, 10, 25, 50, 100]:
    n = int(np.ceil(len(patients) * fraction / 100))
    subset = train[train.patient_id.isin(patients[:n])]
    save(subset, f"train_{fraction}")

# Fixed validation and test sets
save(val, "val")
save(test, "test")

