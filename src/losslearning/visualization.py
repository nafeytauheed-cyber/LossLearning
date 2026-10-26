from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def run_figures(model, split, output, title):
    X = split["X_train"]
    mask = split["noise_mask"]
    output = Path(output)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    axes[0, 0].plot(
        [r["epoch"] for r in model.history_],
        [r["train_ce"] for r in model.history_],
        label="Train CE (observed)",
    )
    axes[0, 0].plot(
        [r["epoch"] for r in model.history_],
        [r["validation_ce"] for r in model.history_],
        label="Validation CE (clean)",
    )
    axes[0, 0].set(title="Learning curves", xlabel="Epoch")
    axes[0, 0].legend()
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 130),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 130),
    )
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    axes[0, 1].contourf(xx, yy, pred, alpha=0.25, cmap="coolwarm")
    axes[0, 1].scatter(*X.T, c=split["y_train"], s=8, cmap="coolwarm")
    axes[0, 1].scatter(*X[mask].T, s=24, facecolors="none", edgecolors="black", linewidths=0.6)
    axes[0, 1].set(title="Observed labels; rings mark corruption")
    sc = axes[0, 2].scatter(*X.T, c=model.q_, vmin=0, vmax=1, s=10, cmap="viridis")
    axes[0, 2].set(title="Final robustness q")
    fig.colorbar(sc, ax=axes[0, 2])
    for label, select in [("Clean", ~mask), ("Corrupted", mask)]:
        if select.any():
            axes[1, 0].hist(
                model.contradiction_scores_[select],
                bins=np.linspace(0, 1, 25),
                alpha=0.55,
                density=True,
                label=label,
            )
            values = np.stack([r["q"] for r in model.trainer_.sample_history])[:, select].mean(1)
            axes[1, 1].plot(
                [r["epoch"] for r in model.trainer_.sample_history], values, label=label
            )
    axes[1, 0].set(title="Contradiction history", xlabel="Score")
    axes[1, 0].legend()
    axes[1, 1].set(title="Robustness history", xlabel="Epoch", ylabel="Mean q")
    axes[1, 1].legend()
    with torch.no_grad():
        h, _ = model.model_.features(torch.from_numpy(model._prepare(X)).to(model.device_))
    h = h.cpu().numpy()
    pca = PCA(2).fit(h)
    points = pca.transform(h)
    axes[1, 2].scatter(*points.T, c=split["clean_train"], s=8, cmap="coolwarm", alpha=0.5)
    if model.head == "potential":
        centers = pca.transform(model.model_.head.centers.detach().cpu().numpy())
        axes[1, 2].scatter(*centers.T, c="black", marker="x", s=50, label="Prototypes")
        axes[1, 2].legend()
    axes[1, 2].set(title="Embedding PCA (not input-space centers)")
    fig.suptitle(title, fontsize=15)
    fig.savefig(output / "diagnostics.png", dpi=150)
    plt.close(fig)


def summary_figures(frame, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = list(frame.groupby("dataset"))
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(8, 3.5 * len(groups)), squeeze=False, layout="constrained"
    )
    for ax, (dataset, data) in zip(axes[:, 0], groups):
        for method, rows in data.groupby("method"):
            g = rows.groupby("noise_rate")["balanced_accuracy"].agg(["mean", "std"])
            ax.errorbar(
                g.index, g["mean"], yerr=g["std"].fillna(0), marker="o", label=method, capsize=3
            )
        ax.set(
            title=dataset,
            xlabel="Training label flip probability",
            ylabel="Balanced accuracy",
            ylim=(0.3, 1.02),
        )
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, ncol=3, loc="lower left")
    fig.savefig(output / "benchmark.png", dpi=150)
    plt.close(fig)
