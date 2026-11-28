"""Patient-level splits, raw WFDB signals, and train-only normalization."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import wfdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
SAMPLE_RATE, SIGNAL_LENGTH, N_LEADS = 100, 1000, 12


def project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository, independently of the cwd."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def resolve_data_root(path: str | Path = "data/ptbxl") -> Path:
    """Accept the records100 parent or a directory with one extracted dataset."""
    base = project_path(path)
    if (base / "records100").is_dir():
        return base
    candidates = sorted(p.parent for p in base.glob("*/records100") if p.is_dir())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected records100/ in {base} or one dataset subdirectory. "
            "Use --data-root or PTBXL_ROOT to select the dataset."
        )
    return candidates[0]


def resolve_split_dir(path: str | Path = "splits", seed: int = 42) -> Path:
    """Resolve either the seeded split root or a legacy flat split directory.

    The current repository layout is ``splits/seed_<seed>/``. Accepting a flat
    directory keeps archived experiments and external split sets usable, while
    rejecting a mismatched explicit ``seed_*`` directory prevents a run from
    silently training on one subset and recording another seed.
    """
    base = project_path(path)
    seed_name = f"seed_{seed}"
    if base.name == seed_name:
        selected = base
    elif base.name.startswith("seed_"):
        raise ValueError(
            f"Split directory {base} does not match configured seed {seed}."
        )
    elif (base / seed_name).is_dir():
        selected = base / seed_name
    elif (base / "val.csv").is_file() and (base / "test.csv").is_file():
        selected = base
    else:
        raise FileNotFoundError(
            f"Expected {base / seed_name} or a flat split directory containing "
            "val.csv and test.csv."
        )
    return selected.resolve()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_splits(
    split_dir: Path, data_root: Path, train_percents: tuple[int, ...]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Check patient separation, nested subsets, identity, and metadata labels.

    Labels from the committed split CSVs are authoritative. Metadata differences
    are returned for reporting, rather than silently changing experiment targets.
    """
    frames = {
        f"train_{p}": pd.read_csv(split_dir / f"train_{p}.csv")
        for p in train_percents
    }
    frames.update({name: pd.read_csv(split_dir / f"{name}.csv") for name in ("val", "test")})
    metadata = pd.read_csv(data_root / "ptbxl_database.csv").set_index("ecg_id")
    scp = pd.read_csv(data_root / "scp_statements.csv", index_col=0)
    mapping = scp.loc[scp.diagnostic == 1, "diagnostic_class"].dropna().to_dict()
    patients = {name: set(frame.patient_id) for name, frame in frames.items()}
    differences = []
    for name, frame in frames.items():
        if frame.empty or not frame.ecg_id.is_unique or frame.patient_id.isna().any():
            raise ValueError(f"Empty split, duplicate ECG IDs, or missing patients: {name}")
        if not frame[CLASSES].isin([0, 1]).all().all() or not (frame[CLASSES].sum(axis=1) > 0).all():
            raise ValueError(f"Invalid binary diagnostic labels: {name}")
        if not frame.filename_lr.str.startswith("records100/").all():
            raise ValueError(f"Expected records100 filenames: {name}")
        source = metadata.loc[frame.ecg_id]
        for column in ("patient_id", "filename_lr", "strat_fold"):
            if not np.array_equal(frame[column].to_numpy(), source[column].to_numpy()):
                raise ValueError(f"Metadata {column} mismatch: {name}")
        expected = []
        for raw in source.scp_codes:
            classes = {mapping[c] for c in ast.literal_eval(raw) if c in mapping}
            expected.append([int(c in classes) for c in CLASSES])
        actual = frame[CLASSES].to_numpy()
        expected = np.asarray(expected)
        for row, column in np.argwhere(actual != expected):
            differences.append({
                "split": name, "ecg_id": int(frame.iloc[row].ecg_id),
                "class": CLASSES[column], "csv_label": int(actual[row, column]),
                "metadata_label": int(expected[row, column]),
            })
        if name.startswith("train_"):
            if not frame.strat_fold.between(1, 8).all():
                raise ValueError(f"Unexpected training folds: {name}")
            for held_out in ("val", "test"):
                if not patients[name].isdisjoint(patients[held_out]):
                    raise ValueError(f"Patient overlap: {name}, {held_out}")
        elif not (frame.strat_fold == {"val": 9, "test": 10}[name]).all():
            raise ValueError(f"Unexpected held-out fold: {name}")
    if not patients["val"].isdisjoint(patients["test"]):
        raise ValueError("Validation/test patient overlap")
    for small, large in zip(train_percents[:-1], train_percents[1:]):
        a, b = frames[f"train_{small}"], frames[f"train_{large}"]
        included = b.patient_id.isin(patients[f"train_{small}"])
        if not patients[f"train_{small}"] <= patients[f"train_{large}"] or set(b.loc[included, "ecg_id"]) != set(a.ecg_id):
            raise ValueError(f"Training subsets are not nested: {small}, {large}")
    audit = pd.DataFrame(differences, columns=["split", "ecg_id", "class", "csv_label", "metadata_label"])
    provenance = {
        "data_root": str(data_root),
        "metadata_sha256": file_hash(data_root / "ptbxl_database.csv"),
        "scp_sha256": file_hash(data_root / "scp_statements.csv"),
        "split_sha256": {name: file_hash(split_dir / f"{name}.csv") for name in frames},
    }
    return frames, audit, provenance


class ECGDataset(Dataset):
    """Return raw signals as (1000, 12) tensors and five binary labels."""

    def __init__(self, signals: np.ndarray, indices: np.ndarray, targets: np.ndarray):
        self.signals = signals
        self.indices = indices
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        signal = np.array(self.signals[self.indices[index]], copy=True)
        return torch.from_numpy(signal), self.targets[index]


@dataclass
class PreparedData:
    signals: np.ndarray
    frames: dict[str, pd.DataFrame]
    indices: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    leads: list[str]
    provenance: dict

    def loader(self, name: str, batch_size: int, seed: int, *, training=False, pin_memory=False):
        return DataLoader(
            ECGDataset(self.signals, self.indices[name], self.labels[name]),
            batch_size=batch_size, shuffle=training, num_workers=0,
            generator=torch.Generator().manual_seed(seed), pin_memory=pin_memory,
        )


def prepare_data(
    data_root: Path, split_dir: Path, output_dir: Path,
    train_percents: tuple[int, ...], *, include_test: bool = True,
) -> PreparedData:
    """Audit CSVs and cache each required raw recording once in a memmap."""
    frames, audit, provenance = audit_splits(split_dir, data_root, train_percents)
    audit.to_csv(output_dir / "label_audit.csv", index=False)
    save_json(output_dir / "provenance.json", provenance)
    summary = pd.DataFrame({
        name: {"recordings": len(frame), "patients": frame.patient_id.nunique(),
               **frame[CLASSES].sum().to_dict()}
        for name, frame in frames.items()
    }).T
    summary.to_csv(output_dir / "split_summary.csv")
    print(summary.to_string(), flush=True)
    if len(audit):
        print("Metadata label differences; retaining the supplied CSV labels:\n" + audit.to_string(index=False), flush=True)
    cached_names = [f"train_{max(train_percents)}", "val"] + (["test"] if include_test else [])
    records = (pd.concat([frames[name] for name in cached_names], ignore_index=True)
               .sort_values("ecg_id").reset_index(drop=True))
    if not records.ecg_id.is_unique:
        raise ValueError("Overlapping recording IDs across splits")
    records.to_csv(output_dir / "signal_index.csv", index=False)
    path = output_dir / "signals.npy"
    signals = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(records), SIGNAL_LENGTH, N_LEADS),
    )
    leads = None
    for i, filename in enumerate(records.filename_lr):
        signal, fields = wfdb.rdsamp(str(data_root / filename))
        if signal.shape != (SIGNAL_LENGTH, N_LEADS) or fields["fs"] != SAMPLE_RATE or not np.isfinite(signal).all():
            raise ValueError(f"Unexpected signal shape, rate, or values: {filename}")
        if leads is None:
            leads = fields["sig_name"]
        if fields["sig_name"] != leads or any(unit != "mV" for unit in fields["units"]):
            raise ValueError(f"Unexpected lead order or units: {filename}")
        signals[i] = signal
        if (i + 1) % 2000 == 0:
            print(f"Loaded {i + 1:,}/{len(records):,} ECGs", flush=True)
    signals.flush()
    del signals
    signals = np.load(path, mmap_mode="r")
    lookup = pd.Series(np.arange(len(records)), index=records.ecg_id)
    loaded_names = [f"train_{p}" for p in train_percents] + ["val"] + (["test"] if include_test else [])
    indices = {name: lookup.loc[frames[name].ecg_id].to_numpy(dtype=np.int64) for name in loaded_names}
    labels = {name: frame[CLASSES].to_numpy(dtype=np.float32) for name, frame in frames.items()}
    save_json(output_dir / "signal_manifest.json", {
        "shape": list(signals.shape), "dtype": str(signals.dtype), "leads": leads,
        "sample_rate": SAMPLE_RATE, "units": "mV", "source": provenance,
    })
    return PreparedData(signals, frames, indices, labels, leads, provenance)


def training_statistics(signals: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-lead population mean/std using only the supplied training rows."""
    count = 0
    mean, m2 = np.zeros(N_LEADS), np.zeros(N_LEADS)
    for start in range(0, len(indices), 128):
        batch = np.asarray(signals[indices[start:start + 128]], dtype=np.float64)
        batch_count = batch.shape[0] * batch.shape[1]
        batch_mean = batch.mean(axis=(0, 1))
        batch_m2 = batch.var(axis=(0, 1)) * batch_count
        delta = batch_mean - mean
        new_count = count + batch_count
        m2 += batch_m2 + delta**2 * count * batch_count / new_count
        mean += delta * batch_count / new_count
        count = new_count
    return mean.astype(np.float32), np.maximum(np.sqrt(m2 / count), 1e-6).astype(np.float32)
