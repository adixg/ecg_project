"""PTB-XL datasets and data-loader helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")
EXPECTED_LEADS = 12
EXPECTED_SAMPLES = 1000


class PTBXLDataset(Dataset):
    """Loads 100 Hz PTB-XL waveforms referenced by a project split CSV."""

    def __init__(
        self,
        split_csv: str | Path,
        data_root: str | Path,
        *,
        normalize: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.data_root = Path(data_root)
        self.normalize = normalize
        self.frame = pd.read_csv(self.split_csv)

        required = {"ecg_id", "filename_lr", *CLASS_NAMES}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(
                f"{self.split_csv} is missing columns: {sorted(missing)}"
            )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        record_path = self.data_root / row["filename_lr"]
        signal, _ = wfdb.rdsamp(str(record_path))

        if signal.shape != (EXPECTED_SAMPLES, EXPECTED_LEADS):
            raise ValueError(
                f"Unexpected waveform shape for {record_path}: {signal.shape}; "
                f"expected {(EXPECTED_SAMPLES, EXPECTED_LEADS)}"
            )

        signal = signal.astype(np.float32, copy=False).T
        if self.normalize:
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            signal = (signal - mean) / np.maximum(std, 1e-6)

        labels = row[list(CLASS_NAMES)].to_numpy(dtype=np.float32)
        return {
            "signal": torch.from_numpy(np.ascontiguousarray(signal)),
            "labels": torch.from_numpy(labels),
            "ecg_id": torch.tensor(int(row["ecg_id"]), dtype=torch.long),
        }


def make_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader:
    """Creates a reproducibly seeded PyTorch data loader."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )
