
import os
import tarfile
from pathlib import Path
import pandas as pd

repo = Path(__file__).resolve().parent
root = Path(os.environ.get("PTBXL_ROOT", repo / "data" / "ptbxl"))
out = Path.home() / "scratch" / "ptbxl_mvp_5pct.tar.gz"
out.parent.mkdir(parents=True, exist_ok=True)

paths = set()

for split in ["train_5", "val"]:
    df = pd.read_csv(repo / "splits" / f"{split}.csv")
    paths.update(df["filename_lr"])

with tarfile.open(out, "w:gz") as archive:
    for path in sorted(paths):
        for ext in [".dat", ".hea"]:
            relative = Path(path + ext)
            archive.add(
                root / relative,
                arcname=str(relative)
            )

print(f"Packaged {len(paths)} ECG recordings")
print(f"Archive size: {out.stat().st_size / 1e6:.1f} MB")

