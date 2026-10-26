"""Extract 300-d DWT features (12 leads * 5 bands * 5 stats) for every split record.

Usage: python features.py            -> features/features.npz
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import wfdb
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis

REPO = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PTBXL_ROOT", REPO / "data" / "ptbxl"))
OUT = REPO / "features" / "features.npz"

FS = 100
WAVELET = "db4"
LEVEL = 4  # -> 5 bands: cA4, cD4, cD3, cD2, cD1
SOS = butter(3, [0.5, 40], btype="band", fs=FS, output="sos")


def band_stats(c):
    energy = np.sum(c ** 2)
    p = c ** 2 / (energy + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return [energy, c.std(), np.abs(c).mean(), entropy, kurtosis(c)]


def extract(filename_lr):
    sig, _ = wfdb.rdsamp(str(ROOT / filename_lr))  # (1000, 12)
    sig = sosfiltfilt(SOS, sig.T, axis=1)
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)
    feats = []
    for lead in sig:
        for band in pywt.wavedec(lead, WAVELET, level=LEVEL):
            feats.extend(band_stats(band))
    return np.asarray(feats, dtype=np.float32)


if __name__ == "__main__":
    frames = [pd.read_csv(REPO / "splits" / f"{n}.csv") for n in ["train_100", "val", "test"]]
    df = pd.concat(frames).drop_duplicates("ecg_id")
    X = Parallel(n_jobs=-1)(delayed(extract)(f) for f in df.filename_lr)
    X = np.nan_to_num(np.stack(X))
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, ecg_id=df.ecg_id.to_numpy(), X=X)
    print(f"Saved {X.shape} to {OUT}")
