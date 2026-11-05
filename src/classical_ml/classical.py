"""Train RF / XGBoost / SVM on DWT features and write results in the shared report layout.

Usage (from the repo root):
    python -m src.classical_ml.classical [--fractions 5 10 ...] [--seeds 42] [--models rf xgb svm]

For every (model, fraction, seed) this writes
    reports/<model>/seed_<seed>/fraction_<N>/{config,metrics}.json
and refreshes reports/<model>/seed_<seed>/summary.csv, mirroring reports/cnn_bilstm/.
Validation/test probabilities go to the git-ignored outputs/ directory.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fractions", nargs="+", type=int, default=[5, 10, 25, 50, 100])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "svm"], choices=list(GRIDS))
    ap.add_argument("--splits-dir", default=str(REPO / "splits"))
    ap.add_argument("--features", default=str(REPO / "features" / "features.npz"))
    ap.add_argument("--reports-dir", default=str(REPO / "reports"))
    ap.add_argument("--outputs-dir", default=str(REPO / "outputs"))
    args = ap.parse_args()

    z = np.load(args.features)
    index = {e: i for i, e in enumerate(z["ecg_id"])}

    def load(name):
        df = pd.read_csv(Path(args.splits_dir) / f"{name}.csv")
        return z["X"][[index[e] for e in df.ecg_id]], df[CLASSES].to_numpy(), df.ecg_id.to_numpy()

    Xva, Yva, ids_va = load("val")
    Xte, Yte, ids_te = load("test")

    for name, seed in itertools.product(args.models, args.seeds):
        seed_dir = Path(args.reports_dir) / name / f"seed_{seed}"
        summary_rows = []
        for frac in args.fractions:
            started = time.perf_counter()
            Xtr, Ytr, _ = load(f"train_{frac}")
            grid = GRIDS[name]
            best = None
            for combo in itertools.product(*grid.values()):
                params = dict(zip(grid, combo))
                Sva, Ste = fit_predict(name, params, Xtr, Ytr, [Xva, Xte], seed)
                auc = macro_auroc(Yva, Sva)  # selection uses validation only
                if best is None or auc > best[0]:
                    best = (auc, params, Sva, Ste)
            _, params, Sva, Ste = best

            th = find_f1_thresholds(Yva, Sva)
            val_m = compute_metrics(Yva, Sva, thresholds=th)
            test_m = compute_metrics(Yte, Ste, thresholds=th)
            results = {
                "best_params": params,
                "thresholds": dict(zip(CLASSES, th.tolist())),
                "validation": {**val_m, "per_class": compute_per_class_metrics(
                    Yva, Sva, thresholds=th, class_names=CLASSES)},
                "test": {**test_m, "per_class": compute_per_class_metrics(
                    Yte, Ste, thresholds=th, class_names=CLASSES)},
            }
            config = {
                "model": name, "train_fraction": frac, "seed": seed, "n_train": len(Xtr),
                "features": "12 leads x 5 db4 DWT bands x 5 stats = 300-d (src/classical_ml/features.py)",
                "grid": grid, "selection_metric": "validation macro AUROC",
                "class_weighting": "balanced (RF/SVM) or scale_pos_weight (XGB), per class, one-vs-rest",
                "score": "sigmoid(decision_function)" if name == "svm" else "predict_proba",
                "seconds": round(time.perf_counter() - started, 1),
            }
            run_dir = seed_dir / f"fraction_{frac}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (run_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
            pred_dir = Path(args.outputs_dir) / name / f"fraction_{frac}_seed_{seed}"
            save_predictions(pred_dir / "validation_predictions.csv", ids_va, Yva, Sva)
            save_predictions(pred_dir / "test_predictions.csv", ids_te, Yte, Ste)

            summary_rows.append({
                "train_fraction": frac, "seed": seed,
                "val_macro_auroc": val_m["macro_auroc"], "val_macro_auprc": val_m["macro_auprc"],
                "val_macro_f1": val_m["macro_f1"], "test_macro_auroc": test_m["macro_auroc"],
                "test_macro_auprc": test_m["macro_auprc"], "test_macro_f1": test_m["macro_f1"],
            })
            pd.DataFrame(summary_rows).to_csv(seed_dir / "summary.csv", index=False)
            print(f"{name} {frac}% seed {seed} ({config['seconds']}s) params={params} "
                  f"test AUROC={test_m['macro_auroc']:.4f} AUPRC={test_m['macro_auprc']:.4f} "
                  f"F1={test_m['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
