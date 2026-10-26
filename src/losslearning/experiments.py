"""CLI experiments with paired splits and final-epoch evaluation."""

import argparse
import json
import platform
import subprocess
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from .data import split_geometry
from .estimator import LossLearning
from .visualization import run_figures, summary_figures

METHODS = {
    "ce": dict(adaptive_loss=False),
    "focal": dict(adaptive_loss=False, trainer_options={"loss": "focal"}),
    "gce": dict(adaptive_loss=False, trainer_options={"loss": "gce"}),
    "potential_ce": dict(adaptive_loss=False, head="potential"),
    "adaptive": dict(adaptive_loss=True),
    "full": dict(adaptive_loss=True, head="potential"),
    "ce_matched": dict(adaptive_loss=False, hidden_dim=80),
}
ABLATIONS = {
    "full": dict(head="potential"),
    "no_adaptation": dict(head="potential", adaptive_loss=False),
    "no_potential": dict(head="linear"),
    "no_gradient": dict(head="potential", trainer_options={"gradient": False}),
    "confidence_only": dict(head="potential", trainer_options={"confidence_only": True}),
    "no_history": dict(head="potential", trainer_options={"rho": 0.0}),
    "no_entropy": dict(head="potential", trainer_options={"entropy": False}),
    "no_class_safeguard": dict(head="potential", trainer_options={"class_aware": False}),
    "class_quantile": dict(head="potential", trainer_options={"threshold_mode": "quantile"}),
    "no_stability": dict(head="potential", trainer_options={"stability_filter": False}),
    "fixed_gce": dict(head="potential", adaptive_loss=False, trainer_options={"loss": "gce"}),
    "cached": dict(head="potential", trainer_options={"reference_mode": "cached"}),
    "neighbors_4": dict(head="potential", trainer_options={"neighbors": 4}),
    "neighbors_24": dict(head="potential", trainer_options={"neighbors": 24}),
}


def classification_metrics(y, pred):
    r = recall_score(y, pred, average=None, zero_division=0)
    return dict(
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        recall=r.tolist(),
        worst_class_recall=float(r.min()),
        confusion_matrix=confusion_matrix(y, pred).tolist(),
    )


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def new_output(config, kind):
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    p = Path(config.get("output", "results/runs")) / (kind + "-" + tag)
    p.mkdir(parents=True)
    (p / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return p


def run_benchmark(config, ablate=False, quick=False):
    torch.set_num_threads(config.get("threads", 1))
    warm = torch.nn.Linear(2, 2)
    torch.optim.AdamW(warm.parameters())
    if config.get("dataset") in ("fashion_mnist", "cifar10"):
        from .images import run_image_benchmark

        return run_image_benchmark(config)
    output = new_output(config, "ablation" if ablate else "benchmark")
    rows = []
    for dataset in config.get("datasets", ["moons"]):
        for noise in config.get("noise_rates", [0.2]):
            for seed in config.get("seeds", [42]):
                split = split_geometry(
                    dataset,
                    config.get("n_samples", 1000),
                    seed,
                    noise,
                    config.get("noise_mode", "symmetric"),
                )
                for method in config.get("methods", list(ABLATIONS if ablate else METHODS)):
                    folder = output / f"{dataset}-noise{noise}-seed{seed}-{method}"
                    folder.mkdir()
                    override = dict((ABLATIONS if ablate else METHODS)[method])
                    options = dict(config.get("trainer_options", {}))
                    options.update(override.pop("trainer_options", {}))
                    params = dict(config.get("model", {}))
                    params.update(override)
                    model = LossLearning(**params, random_state=seed, trainer_options=options)
                    start = time.perf_counter()
                    model.fit(split["X_train"], split["y_train"], split["X_val"], split["y_val"])
                    runtime = time.perf_counter() - start
                    pred = model.predict(split["X_test"])
                    metrics = classification_metrics(split["y_test"], pred)
                    probs = model.predict_proba(split["X_train"])
                    confidence = 1 - probs[np.arange(len(probs)), split["y_train"]]
                    mask = split["noise_mask"]
                    clean = ~mask
                    hard = clean & (confidence >= np.quantile(confidence[clean], 0.8))

                    def auc(score):
                        return (
                            float(roc_auc_score(mask, score)) if len(np.unique(mask)) == 2 else None
                        )

                    row = dict(
                        dataset=dataset,
                        noise_rate=noise,
                        seed=seed,
                        method=method,
                        noise_mode=config.get("noise_mode", "symmetric"),
                        realized_noise=float(mask.mean()),
                        runtime_seconds=runtime,
                        parameters=sum(p.numel() for p in model.model_.parameters()),
                        noise_auc=auc(model.contradiction_scores_),
                        confidence_noise_auc=auc(confidence),
                        mean_q=float(model.q_.mean()),
                        clean_q=float(model.q_[clean].mean()),
                        hard_clean_q=float(model.q_[hard].mean()),
                        corrupted_q=float(model.q_[mask].mean()) if mask.any() else None,
                        fallback_fraction=model.history_[-1]["fallback_fraction"],
                        **metrics,
                    )
                    meta = dict(
                        **row,
                        estimator=model.get_params(),
                        trainer=asdict(model.trainer_.config),
                        torch_version=torch.__version__,
                        python=platform.python_version(),
                        device=model.device_,
                        optimizer="AdamW",
                        scheduler=None,
                        git_commit=git_commit(),
                        split_sizes={k: len(split[k]) for k in ["y_train", "y_val", "y_test"]},
                        peak_memory_bytes=None,
                    )
                    (folder / "metrics.json").write_text(json.dumps(meta, indent=2))
                    (folder / "history.json").write_text(json.dumps(model.history_, indent=2))
                    np.savez_compressed(
                        folder / "diagnostics.npz",
                        **split,
                        q=model.q_,
                        contradiction=model.contradiction_scores_,
                        test_predictions=pred,
                        score_history=np.stack([r["score"] for r in model.trainer_.sample_history]),
                        q_history=np.stack([r["q"] for r in model.trainer_.sample_history]),
                        history_epochs=np.array(
                            [r["epoch"] for r in model.trainer_.sample_history]
                        ),
                    )
                    if config.get("save_checkpoints", False) or quick:
                        model.save(folder / "model.pt")
                        model.trainer_.save_checkpoint(folder / "training.pt")
                    if config.get("figures", False) and (
                        quick or (seed == config["seeds"][0] and method in ("full", "ce"))
                    ):
                        run_figures(
                            model,
                            split,
                            folder,
                            f"{dataset} | {method} | noise={noise} | seed={seed}",
                        )
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
                    print(
                        f"{dataset} noise={noise} seed={seed} {method} acc={metrics['accuracy']:.4f} bal={metrics['balanced_accuracy']:.4f} q={row['mean_q']:.3f} {runtime:.2f}s",
                        flush=True,
                    )
    frame = pd.DataFrame(rows)
    frame.groupby(["dataset", "noise_rate", "method"])[
        [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "worst_class_recall",
            "runtime_seconds",
            "mean_q",
        ]
    ].agg(["mean", "std"]).to_csv(output / "aggregate.csv")
    summary_figures(frame, output)
    print(f"RESULTS: {output}", flush=True)
    return output


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return yaml.safe_load(Path(parser.parse_args().config).read_text())


def train_main():
    run_benchmark(parse(), quick=True)


def benchmark_main():
    run_benchmark(parse())


def ablate_main():
    run_benchmark(parse(), ablate=True)


def figures_main():
    from types import SimpleNamespace

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    path = Path(parser.parse_args().results)
    summary_figures(pd.read_csv(path / "summary.csv"), path)
    for folder in path.iterdir():
        if folder.is_dir() and (folder / "model.pt").exists():
            model = LossLearning.load(folder / "model.pt")
            data = np.load(folder / "diagnostics.npz")
            model.trainer_ = SimpleNamespace(
                sample_history=[
                    dict(q=q, epoch=int(e))
                    for q, e in zip(data["q_history"], data["history_epochs"])
                ]
            )
            run_figures(model, data, folder, folder.name)


if __name__ == "__main__":
    benchmark_main()
