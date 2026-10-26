"""Train RF / XGBoost / SVM on DWT features and report test metrics.

Usage: python classical.py [--fractions 5 10 ...] [--seeds 0 1 2]
Appends one row per (model, fraction, seed) to results/classical.csv.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

GRIDS = {
    "rf": {"n_estimators": [300], "max_depth": [None, 20], "min_samples_leaf": [1, 3]},
    "xgb": {"n_estimators": [200, 400], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]},
    "svm": {"C": [0.1, 1, 10], "gamma": ["scale", 0.001]},
}


def fit_predict(name, params, Xtr, Ytr, Xte_list, seed):
    """One-vs-rest, class-balanced. Returns a score array (n, 5) for each X in Xte_list."""
    outs = [np.zeros((len(X), len(CLASSES))) for X in Xte_list]
    for k in range(len(CLASSES)):
        y = Ytr[:, k]
        if name == "rf":
            m = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=seed, **params)
            score = lambda X: m.predict_proba(X)[:, 1]
        elif name == "xgb":
            spw = (len(y) - y.sum()) / max(y.sum(), 1)
            m = XGBClassifier(scale_pos_weight=spw, n_jobs=-1, random_state=seed,
                              eval_metric="logloss", **params)
            score = lambda X: m.predict_proba(X)[:, 1]
        else:  # hinge-loss SVM; decision_function is a monotone score for AUROC/AUPRC
            m = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", **params))
            score = lambda X: m.decision_function(X)
        m.fit(Xtr, y)
        for out, X in zip(outs, Xte_list):
            out[:, k] = score(X)
    return outs


def macro_auroc(Y, S):
    return np.mean([roc_auc_score(Y[:, k], S[:, k]) for k in range(len(CLASSES))])


def best_thresholds(Y, S):
    ths = []
    for k in range(len(CLASSES)):
        cands = np.quantile(S[:, k], np.linspace(0.01, 0.99, 99))
        f1s = [f1_score(Y[:, k], S[:, k] >= t, zero_division=0) for t in cands]
        ths.append(cands[int(np.argmax(f1s))])
    return np.array(ths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", nargs="+", type=int, default=[5, 10, 25, 50, 100])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "svm"])
    ap.add_argument("--splits-dir", default=str(REPO / "splits"))
    ap.add_argument("--out", default=str(REPO / "results" / "classical.csv"))
    args = ap.parse_args()

    z = np.load(REPO / "features" / "features.npz")
    index = {e: i for i, e in enumerate(z["ecg_id"])}

    def load(name):
        df = pd.read_csv(Path(args.splits_dir) / f"{name}.csv")
        return z["X"][[index[e] for e in df.ecg_id]], df[CLASSES].to_numpy()

    Xva, Yva = load("val")
    Xte, Yte = load("test")
    rows = []
    for frac, seed, name in itertools.product(args.fractions, args.seeds, args.models):
        Xtr, Ytr = load(f"train_{frac}")
        grid = GRIDS[name]
        best = None
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid, combo))
            Sva, = fit_predict(name, params, Xtr, Ytr, [Xva], seed)
            auc = macro_auroc(Yva, Sva)
            if best is None or auc > best[0]:
                best = (auc, params)
        val_auc, params = best
        # settings tuned on val; refit once more on train only (val is never trained on)
        Sva, Ste = fit_predict(name, params, Xtr, Ytr, [Xva, Xte], seed)
        th = best_thresholds(Yva, Sva)
        row = dict(
            model=name, fraction=frac, seed=seed, n_train=len(Xtr), val_auroc=val_auc,
            test_auroc=macro_auroc(Yte, Ste),
            test_auprc=np.mean([average_precision_score(Yte[:, k], Ste[:, k]) for k in range(5)]),
            test_f1=np.mean([f1_score(Yte[:, k], Ste[:, k] >= th[k], zero_division=0) for k in range(5)]),
            params=str(params),
        )
        print(row, flush=True)
        rows.append(row)
        out = Path(args.out)
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
