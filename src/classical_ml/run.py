"""Train RF / XGBoost / SVM on DWT features and write results in the shared report layout.

Usage (from the repo root):
    python -m src.classical_ml.run [--fractions 5 10 ...] [--seeds 42] [--models rf xgb svm]

Every (model, fraction, seed) reads <splits-dir>/seed_<seed>/ and writes
    <artifacts-dir>/<model>_seed<seed>_frac<fraction>/
        config.json  grid_results.json  thresholds.json
        validation_metrics.json  test_metrics.json
        validation_predictions.csv  test_predictions.csv
An existing run directory of the same name is overwritten.
Metrics and F1 thresholds come from src/metrics.py, the same code the CNN-BiLSTM uses.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.metrics import compute_metrics, compute_per_class_metrics, find_f1_thresholds

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "artifacts" / "classical_ml"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

GRIDS = {
    "rf": {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 3]},
    "xgb": {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
    "svm": {"C": [0.1, 1, 10], "gamma": ["scale", 0.001]},
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fit_one_class(name, params, Xtr, y, Xeval_list, seed):
    """Fit one binary classifier; return a score in [0, 1] for each X in Xeval_list."""
    if name == "rf":
        m = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **params)
        score = lambda X: m.predict_proba(X)[:, 1]
    elif name == "xgb":
        spw = (len(y) - y.sum()) / max(y.sum(), 1)
        m = XGBClassifier(scale_pos_weight=spw, n_jobs=-1, random_state=seed,
                          eval_metric="logloss", **params)
        score = lambda X: m.predict_proba(X)[:, 1]
    else:  # SVM margin distance, squashed by a sigmoid (monotone: AUROC/AUPRC unchanged)
        m = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", **params))
        score = lambda X: _sigmoid(m.decision_function(X))
    m.fit(Xtr, y)
    return [score(X) for X in Xeval_list]


def fit_predict(name, params, Xtr, Ytr, Xeval_list, seed):
    """One-vs-rest, class-balanced. Returns an (n, 5) probability-like array per X."""
    # SVC is single-threaded, so run the five classes side by side; RF/XGB parallelize inside.
    jobs = 5 if name == "svm" else 1
    per_class = Parallel(n_jobs=jobs)(
        delayed(_fit_one_class)(name, params, Xtr, Ytr[:, k], Xeval_list, seed)
        for k in range(len(CLASSES))
    )
    return [np.stack([per_class[k][i] for k in range(len(CLASSES))], axis=1)
            for i in range(len(Xeval_list))]


def macro_auroc(Y, S):
    return compute_metrics(Y, S)["macro_auroc"]


def save_predictions(path, ecg_ids, Y, S):
    out = {"ecg_id": ecg_ids}
    for k, c in enumerate(CLASSES):
        out[f"target_{c}"] = Y[:, k].astype(np.int8)
        out[f"probability_{c}"] = S[:, k]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)


def repo_path(value):
    """Resolve a CLI path; relative paths are taken from the repository root."""
    return REPO / Path(value).expanduser()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fractions", nargs="+", type=int, default=[5, 10, 25, 50, 100],
                    help="training percentages; each needs train_<N>.csv (default: %(default)s)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="split seeds, also the model random_state (default: %(default)s)")
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "svm"], choices=list(GRIDS),
                    help="models to run (default: %(default)s)")
    ap.add_argument("--splits-dir", type=repo_path, default=REPO / "splits",
                    help="base directory holding seed_<S>/ folders (default: splits)")
    ap.add_argument("--features", type=repo_path, default=ARTIFACTS / "features" / "features.npz",
                    help="feature file from features.py (default: artifacts/classical_ml/features/features.npz)")
    ap.add_argument("--artifacts-dir", type=repo_path, default=ARTIFACTS,
                    help="root for run directories (default: artifacts/classical_ml)")
    args = ap.parse_args()

    # Fail before any training if a requested split file is absent.
    for seed in args.seeds:
        for name in ["val", "test"] + [f"train_{f}" for f in args.fractions]:
            path = args.splits_dir / f"seed_{seed}" / f"{name}.csv"
            if not path.is_file():
                ap.error(f"missing split file: {path}")

    z = np.load(args.features)
    index = {e: i for i, e in enumerate(z["ecg_id"])}

    def load(seed, name):
        df = pd.read_csv(args.splits_dir / f"seed_{seed}" / f"{name}.csv")
        return z["X"][[index[e] for e in df.ecg_id]], df[CLASSES].to_numpy(), df.ecg_id.to_numpy()

    for name, seed in itertools.product(args.models, args.seeds):
        Xva, Yva, ids_va = load(seed, "val")
        Xte, Yte, ids_te = load(seed, "test")
        for frac in args.fractions:
            started = time.perf_counter()
            Xtr, Ytr, _ = load(seed, f"train_{frac}")
            grid = GRIDS[name]
            best = None
            grid_results = []
            for combo in itertools.product(*grid.values()):
                params = dict(zip(grid, combo))
                Sva, Ste = fit_predict(name, params, Xtr, Ytr, [Xva, Xte], seed)
                auc = macro_auroc(Yva, Sva)  # selection uses validation only
                grid_results.append({"params": params, "validation_macro_auroc": auc})
                if best is None or auc > best[0]:
                    best = (auc, params, Sva, Ste)
            _, params, Sva, Ste = best

            th = find_f1_thresholds(Yva, Sva)
            val_m = compute_metrics(Yva, Sva, thresholds=th)
            test_m = compute_metrics(Yte, Ste, thresholds=th)
            val_full = {**val_m, "per_class": compute_per_class_metrics(
                Yva, Sva, thresholds=th, class_names=CLASSES)}
            test_full = {**test_m, "per_class": compute_per_class_metrics(
                Yte, Ste, thresholds=th, class_names=CLASSES)}
            config = {
                "model": name, "train_fraction": frac, "seed": seed, "n_train": len(Xtr),
                "features": "12 leads x 5 db4 DWT bands x 5 stats = 300-d (src/classical_ml/features.py)",
                "grid": grid, "best_params": params,
                "selection_metric": "validation macro AUROC",
                "class_weighting": "balanced (RF/SVM) or scale_pos_weight (XGB), per class, one-vs-rest",
                "score": "sigmoid(decision_function)" if name == "svm" else "predict_proba",
                "seconds": round(time.perf_counter() - started, 1),
            }
            run_dir = args.artifacts_dir / f"{name}_seed{seed}_frac{frac}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "config.json", config)
            write_json(run_dir / "grid_results.json", grid_results)
            write_json(run_dir / "thresholds.json", dict(zip(CLASSES, th.tolist())))
            write_json(run_dir / "validation_metrics.json", val_full)
            write_json(run_dir / "test_metrics.json", test_full)
            save_predictions(run_dir / "validation_predictions.csv", ids_va, Yva, Sva)
            save_predictions(run_dir / "test_predictions.csv", ids_te, Yte, Ste)

            print(f"{name} {frac}% seed {seed} ({config['seconds']}s) params={params} "
                  f"test AUROC={test_m['macro_auroc']:.4f} AUPRC={test_m['macro_auprc']:.4f} "
                  f"F1={test_m['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
