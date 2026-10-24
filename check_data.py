import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

REPO = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PTBXL_ROOT", REPO / "data" / "ptbxl"))
SPLITS = REPO / "splits"

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SPLIT_NAMES = [
    "train_5", "train_10", "train_25", "train_50", "train_100",
    "val", "test",
]

# Pass --full to read every record instead of a sample
FULL = "--full" in sys.argv

if not (ROOT / "records100").is_dir():
    sys.exit(
        f"records100/ not found under {ROOT}\n"
        "Extract ptbxl_records100.tar.gz into data/ptbxl/ "
        "or set PTBXL_ROOT."
    )

splits = {name: pd.read_csv(SPLITS / f"{name}.csv") for name in SPLIT_NAMES}

# Every referenced record must exist on disk
missing = [
    path
    for df in splits.values()
    for path in df["filename_lr"]
    for ext in [".dat", ".hea"]
    if not (ROOT / (path + ext)).is_file()
]
if missing:
    sys.exit(f"{len(missing)} files missing, e.g. {missing[:3]}")

# No patient may appear in more than one of train/val/test
patients = {
    name: set(splits[name].patient_id)
    for name in ["train_100", "val", "test"]
}
assert patients["train_100"].isdisjoint(patients["val"])
assert patients["train_100"].isdisjoint(patients["test"])
assert patients["val"].isdisjoint(patients["test"])

# Training subsets must be nested
for small, large in zip(SPLIT_NAMES[:4], SPLIT_NAMES[1:5]):
    assert set(splits[small].patient_id) <= set(splits[large].patient_id)

# Signals must load as 1000 samples x 12 leads at 100 Hz
paths = pd.concat(splits.values()).filename_lr.drop_duplicates()
if not FULL:
    paths = paths.sample(50, random_state=0)
for path in paths:
    signal, meta = wfdb.rdsamp(str(ROOT / path))
    assert signal.shape == (1000, 12), (path, signal.shape)
    assert meta["fs"] == 100, (path, meta["fs"])
    assert np.isfinite(signal).all(), path

for name, df in splits.items():
    print(f"{name}: {len(df)} ECGs, {df.patient_id.nunique()} patients")
print(f"Loaded {len(paths)} records OK. Data is ready.")
