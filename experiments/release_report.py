"""Audit and package completed runs; generate tables from measured outputs."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from losslearning.visualization import plt


def markdown(df):
    rows = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    return "\n".join(
        rows
        + ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False, name=None)]
    )


def table(df):
    rows = []
    for (dataset, noise, method), g in df.groupby(["dataset", "noise_rate", "method"]):
        rows.append(
            dict(
                Dataset=dataset,
                Noise=f"{noise:.0%}",
                Method=method,
                Balanced_accuracy=f"{100 * g.balanced_accuracy.mean():.2f} ± {100 * g.balanced_accuracy.std():.2f}",
            )
        )
    return markdown(pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser()
    for key in ["synthetic", "ablation", "class-dependent", "images", "quick"]:
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    release = root / "results/release"
    release.mkdir(parents=True, exist_ok=True)
    sources = {k: Path(v) for k, v in vars(args).items()}
    frames = {}
    expected = dict(synthetic=504, ablation=84, class_dependent=36, images=4, quick=2)
    for name, source in sources.items():
        df = pd.read_csv(source / "summary.csv")
        assert len(df) == expected[name]
        frames[name] = df
        shutil.copytree(
            source,
            release / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.pt", "*.pth"),
        )
    shutil.copytree(root / "results/influence", release / "influence", dirs_exist_ok=True)
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    for method in ["ce", "full"]:
        source = next(sources["quick"].glob("*-" + method))
        shutil.copyfile(source / "diagnostics.png", assets / f"{method}_diagnostics.png")
    shutil.copyfile(root / "results/influence/influence_error.png", assets / "influence_error.png")
    df = frames["synthetic"]
    grid = (
        df.groupby(["dataset", "noise_rate", "method"]).balanced_accuracy.mean().unstack("method")
    )
    delta = ((grid["full"] - grid["ce"]) * 100).unstack("noise_rate")
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    bound = max(1, float(abs(delta.to_numpy()).max()))
    plot = ax.imshow(delta, vmin=-bound, vmax=bound, cmap="RdBu")
    ax.set_xticks(range(len(delta.columns)), [f"{v:.0%}" for v in delta.columns])
    ax.set_yticks(range(len(delta.index)), delta.index)
    for i in range(len(delta)):
        for j in range(len(delta.columns)):
            ax.text(
                j,
                i,
                f"{delta.iloc[i, j]:+.2f}",
                ha="center",
                va="center",
                color="white" if abs(delta.iloc[i, j]) > 0.7 * bound else "black",
            )
    ax.set(
        xlabel="Training label flip probability",
        title="Full model minus CE: balanced-accuracy points",
    )
    fig.colorbar(plot, ax=ax, label="Percentage points (3-seed mean)")
    fig.savefig(assets / "comparison.png", dpi=170)
    plt.close(fig)
    rare = df[(df.dataset == "rare") & (df.noise_rate == 0.2)]
    recall = np.stack([ast.literal_eval(x) for x in rare.recall])
    rare = rare.assign(majority_recall=recall[:, 0], minority_recall=recall[:, 1])
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    rare.groupby("method")[["majority_recall", "minority_recall"]].mean().plot.bar(
        ax=ax, color=["#137b80", "#d47b42"]
    )
    ax.set(ylim=(0, 1.05), ylabel="Recall", title="Rare-cluster task: 20% noise, 3-seed means")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(assets / "class_recall.png", dpi=160)
    plt.close(fig)
    paired = df.pivot(
        index=["dataset", "noise_rate", "seed"], columns="method", values="balanced_accuracy"
    )
    difference = paired["full"] - paired["ce"]
    runtime = df.pivot(
        index=["dataset", "noise_rate", "seed"], columns="method", values="runtime_seconds"
    )
    stats = dict(
        full_vs_ce_wins=int((difference > 0).sum()),
        ties=int((difference == 0).sum()),
        losses=int((difference < 0).sum()),
        mean_balanced_accuracy_difference_pp=float(difference.mean() * 100),
        median_full_to_ce_runtime=float((runtime["full"] / runtime["ce"]).median()),
    )
    (release / "paired_summary.json").write_text(json.dumps(stats, indent=2))
    sections = [
        "# Measured result tables",
        "Balanced accuracy: percentage mean ± sample standard deviation over seeds 11,22,33. Single-seed images use raw proportions. No test-driven selection.",
        "## Synthetic: 20% symmetric noise",
        table(df[(df.noise_rate == 0.2) & df.method.isin(["ce", "gce", "adaptive", "full"])]),
        "## Ablations",
        table(frames["ablation"]),
        "## Class-dependent noise",
        table(frames["class_dependent"]),
        "## Reduced Fashion-MNIST",
        markdown(
            frames["images"][
                ["method", "accuracy", "balanced_accuracy", "mean_q", "fallback_fraction"]
            ].round(4)
        ),
        "## Paired descriptive statistics",
        "```json\n" + json.dumps(stats, indent=2) + "\n```",
    ]
    (root / "docs/result_tables.md").write_text("\n\n".join(sections) + "\n")
    checked = 0
    for kind in ["synthetic", "ablation", "class_dependent", "quick"]:
        for folder in (release / kind).iterdir():
            if not folder.is_dir():
                continue
            d = np.load(folder / "diagnostics.npz")
            m = json.loads((folder / "metrics.json").read_text())
            for x, y in [
                ("train_ids", "test_ids"),
                ("train_ids", "val_ids"),
                ("val_ids", "test_ids"),
            ]:
                assert not set(d[x]) & set(d[y])
            assert np.array_equal(d["noise_mask"], d["y_train"] != d["clean_train"])
            assert np.isclose(accuracy_score(d["y_test"], d["test_predictions"]), m["accuracy"])
            assert np.isclose(
                balanced_accuracy_score(d["y_test"], d["test_predictions"]), m["balanced_accuracy"]
            )
            assert np.isclose(d["q"].mean(), m["mean_q"])
            checked += 1
    audit = dict(
        synthetic_runs_checked=checked,
        expected_runs=expected,
        status="passed",
        paired_statistics=stats,
    )
    (release / "audit.json").write_text(json.dumps(audit, indent=2))
    versions = {
        n: importlib.metadata.version(n)
        for n in [
            "torch",
            "torchvision",
            "numpy",
            "scikit-learn",
            "pandas",
            "matplotlib",
            "pytest",
            "ruff",
        ]
    }
    (release / "environment.txt").write_text(
        "\n".join(f"{k}=={v}" for k, v in versions.items()) + "\n"
    )
    for source, name in [
        (root / "results_tests.log", "tests.txt"),
        (root / "results_quickstart.log", "quickstart.txt"),
    ]:
        shutil.copyfile(source, release / name)
    hashes = {
        str(p.relative_to(release)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in release.rglob("*")
        if p.is_file() and p.name != "sha256.json"
    }
    (release / "sha256.json").write_text(json.dumps(hashes, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
