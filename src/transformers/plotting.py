"""Save experiment figures without requiring a notebook or display server."""

import pandas as pd


def save_figures(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percents = list(result.training_summary.index)
    figure, axes = plt.subplots(2, len(percents), figsize=(5 * len(percents), 7), squeeze=False)
    for column, percent in enumerate(percents):
        history = pd.read_csv(result.run_dir / f"train_{percent}" / "history.csv")
        axes[0, column].plot(history.epoch, history.train_loss, label="Train")
        axes[0, column].plot(history.epoch, history.val_loss, label="Validation")
        axes[0, column].set(title=f"{percent}% training subset", xlabel="Epoch", ylabel="BCE loss")
        axes[1, column].plot(history.epoch, history.val_macro_AUROC, label="Validation AUROC")
        axes[1, column].plot(history.epoch, history.val_macro_AUPRC, label="Validation AUPRC (AP)")
        axes[1, column].set(xlabel="Epoch", ylabel="Macro score", ylim=(0.4, 1))
        for row in range(2):
            axes[row, column].axvline(result.training_summary.loc[percent, "best_epoch"], color="gray", linestyle="--")
            axes[row, column].legend()
            axes[row, column].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(result.run_dir / "training_curves.png", dpi=150)
    plt.close(figure)
    if result.test_summary is None:
        return
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes[:2], ("AUROC", "AUPRC")):
        ax.plot(percents, result.test_summary[f"macro_{metric}"], marker="o", label="Test")
        ax.plot(percents, result.training_summary[f"val_macro_{metric}"], marker="s", label="Validation")
        ax.set(ylabel=f"Macro {metric}")
    axes[2].plot(percents, result.test_summary["macro_F1_at_0.5"], marker="o", label="Test F1 at 0.5")
    axes[2].plot(percents, result.test_summary.macro_F1_val_threshold, marker="s", label="Test F1, validation thresholds")
    axes[2].set(ylabel="Macro F1")
    for ax in axes:
        ax.set(xlabel="Training patient subset (%)", xticks=percents)
        ax.legend()
        ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(result.run_dir / "test_learning_curves.png", dpi=150)
    plt.close(figure)
