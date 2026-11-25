"""Print markdown result tables from artifacts/classical_ml/ run directories.

Usage (from the repo root): python -m src.classical_ml.summarize [--artifacts-dir DIR]
Read-only: it writes nothing; redirect stdout to keep the tables.
"""
import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
MODELS = ["rf", "xgb", "svm"]
DETAIL_SEED, DETAIL_FRACTION = 42, 100


def load_runs(root):
    """Return {(model, seed, fraction): {file stem: parsed JSON}}."""
    runs = {}
    for path in sorted(root.glob("*_seed*_frac*/config.json")):
        config = json.loads(path.read_text())
        run = {"config": config}
        for stem in ("validation_metrics", "test_metrics", "thresholds"):
            run[stem] = json.loads((path.parent / f"{stem}.json").read_text())
        runs[config["model"], config["seed"], config["train_fraction"]] = run
    return runs


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", type=lambda v: REPO / Path(v).expanduser(),
                    default=REPO / "artifacts" / "classical_ml",
                    help="root of the run directories (default: artifacts/classical_ml)")
    args = ap.parse_args()
    runs = load_runs(args.artifacts_dir)
    if not runs:
        ap.error(f"no run directories under {args.artifacts_dir}")
    seeds = sorted({k[1] for k in runs})
    fractions = sorted({k[2] for k in runs})
    models = [m for m in MODELS if any(k[0] == m for k in runs)]

    print(f"## Cross-seed results (seeds {', '.join(map(str, seeds))})\n")
    print("Mean ± sample standard deviation over seeds; [n] is the number of seeds.\n")
    for split in ("validation", "test"):
        for metric, label in (("macro_auroc", "AUROC"), ("macro_auprc", "AUPRC"), ("macro_f1", "F1")):
            rows = []
            for m in models:
                cells = [m]
                for f in fractions:
                    v = [runs[m, s, f][f"{split}_metrics"][metric] for s in seeds if (m, s, f) in runs]
                    spread = f" ± {statistics.stdev(v):.3f}" if len(v) > 1 else ""
                    cells.append(f"{statistics.mean(v):.3f}{spread} [{len(v)}]" if v else "n/a")
                rows.append(cells)
            print(f"### {split.title()} macro {label}\n")
            print(table(["Model"] + [f"{f}%" for f in fractions], rows))

    print("## Every run\n")
    print("Selected setting is the grid entry with the highest validation macro AUROC. "
          "Time covers the whole grid.\n")
    rows = []
    for m in models:
        for s in seeds:
            for f in fractions:
                if (m, s, f) not in runs:
                    continue
                r = runs[m, s, f]
                c, va, te = r["config"], r["validation_metrics"], r["test_metrics"]
                rows.append([f"`{m}_seed{s}_frac{f}`", c["n_train"],
                             ", ".join(f"{k}={v}" for k, v in c["best_params"].items()), c["seconds"]]
                            + [f"{d[k]:.4f}" for d in (va, te)
                               for k in ("macro_auroc", "macro_auprc", "macro_f1")])
    print(table(["Run", "ECGs", "Selected setting", "Seconds", "Val AUROC", "Val AUPRC", "Val F1",
                 "Test AUROC", "Test AUPRC", "Test F1"], rows))

    print(f"## Classwise results ({DETAIL_FRACTION}% training data, seed {DETAIL_SEED})\n")
    print("Thresholds are chosen per class on validation and frozen for test.\n")
    for m in models:
        key = (m, DETAIL_SEED, DETAIL_FRACTION)
        if key not in runs:
            continue
        r = runs[key]
        rows = []
        for split in ("validation", "test"):
            for c in CLASSES:
                v = r[f"{split}_metrics"]["per_class"][c]
                rows.append([split.title(), c, f"{v['auroc']:.4f}", f"{v['auprc']:.4f}",
                             f"{v['f1']:.4f}", f"{r['thresholds'][c]:.2f}"])
        print(f"### {m}_seed{DETAIL_SEED}_frac{DETAIL_FRACTION}\n")
        print(table(["Split", "Class", "AUROC", "AUPRC", "F1", "Threshold"], rows))


if __name__ == "__main__":
    main()
