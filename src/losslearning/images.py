"""Torchvision image pilot: transforms are streamed through indexed DataLoaders."""

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .data import corrupt_labels
from .models import LossLearningNet
from .training import LossLearningTrainer, TrainingConfig


class IndexedImages(Dataset):
    def __init__(self, source, indices, labels):
        self.source, self.indices, self.labels = source, np.asarray(indices), np.asarray(labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, _ = self.source[int(self.indices[i])]
        return x, int(self.labels[i]), i, i


def subset(indices, labels, limit, seed):
    if limit >= len(indices):
        return indices
    return train_test_split(indices, train_size=limit, stratify=labels[indices], random_state=seed)[
        0
    ]


def run_image_benchmark(config):
    from torchvision import datasets, transforms

    from .experiments import METHODS, classification_metrics, git_commit, new_output

    cls = datasets.FashionMNIST if config["dataset"] == "fashion_mnist" else datasets.CIFAR10
    transform = transforms.ToTensor()
    download = config.get("download", False)
    train_source = cls(root="data", train=True, transform=transform, download=download)
    test_source = cls(root="data", train=False, transform=transform, download=download)
    train_labels = np.asarray(train_source.targets)
    test_labels = np.asarray(test_source.targets)
    output = new_output(config, "image-pilot")
    rows = []
    for seed in config["seeds"]:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_labels)), test_size=0.1, stratify=train_labels, random_state=seed
        )
        train_idx = subset(train_idx, train_labels, config.get("train_limit", len(train_idx)), seed)
        val_idx = subset(val_idx, train_labels, config.get("validation_limit", len(val_idx)), seed)
        test_idx = subset(
            np.arange(len(test_labels)),
            test_labels,
            config.get("test_limit", len(test_labels)),
            seed,
        )
        for noise in config["noise_rates"]:
            observed, mask = corrupt_labels(train_labels[train_idx], noise, seed + 1000)
            for method in config["methods"]:
                folder = output / f"{method}-noise{noise}-seed{seed}"
                folder.mkdir()
                torch.manual_seed(seed)
                np.random.seed(seed)
                params = dict(config["model"])
                params.update(METHODS[method])
                options = dict(config.get("trainer_options", {}))
                options.update(params.pop("trainer_options", {}))
                device = params.get("device", "auto")
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                model = LossLearningNet(
                    tuple(train_source[0][0].shape),
                    10,
                    encoder="cnn",
                    embedding_dim=params.get("embedding_dim", 32),
                    n_prototypes=params.get("n_prototypes", 16),
                    head=params.get("head", "linear"),
                )
                cfg = TrainingConfig(
                    epochs=params.get("epochs", 10),
                    seed=seed,
                    adaptive=params.get("adaptive_loss", True),
                    **options,
                )
                trainer = LossLearningTrainer(model, observed, cfg, device)
                batch = params.get("batch_size", 128)
                data = IndexedImages(train_source, train_idx, observed)
                loader = DataLoader(
                    data,
                    batch_size=batch,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                )
                ref = DataLoader(data, batch_size=batch)
                val = DataLoader(
                    IndexedImages(train_source, val_idx, train_labels[val_idx]), batch_size=batch
                )
                test = DataLoader(
                    IndexedImages(test_source, test_idx, test_labels[test_idx]), batch_size=batch
                )
                start = time.perf_counter()
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                trainer.fit(loader, ref, val)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                pred = []
                with torch.no_grad():
                    for x, _, _, _ in test:
                        pred.extend(model(x.to(device)).argmax(1).cpu().tolist())
                row = dict(
                    dataset=config["dataset"],
                    method=method,
                    noise_rate=noise,
                    seed=seed,
                    realized_noise=float(mask.mean()),
                    runtime_seconds=elapsed,
                    parameters=sum(p.numel() for p in model.parameters()),
                    train_size=len(data),
                    validation_size=len(val.dataset),
                    test_size=len(test.dataset),
                    mean_q=float(trainer.controller.q.mean()),
                    fallback_fraction=trainer.history[-1]["fallback_fraction"],
                    **classification_metrics(test_labels[test_idx], pred),
                )
                meta = dict(
                    **row,
                    config=asdict(cfg),
                    architecture=str(model),
                    device=device,
                    torch_version=torch.__version__,
                    git_commit=git_commit(),
                    optimizer="AdamW",
                    scheduler=None,
                    peak_memory_bytes=torch.cuda.max_memory_allocated()
                    if device == "cuda"
                    else None,
                )
                (folder / "metrics.json").write_text(json.dumps(meta, indent=2))
                (folder / "history.json").write_text(json.dumps(trainer.history, indent=2))
                np.savez_compressed(
                    folder / "split.npz",
                    train_ids=train_idx,
                    val_ids=val_idx,
                    test_ids=test_idx,
                    observed_labels=observed,
                    noise_mask=mask,
                    predictions=pred,
                    q=trainer.controller.q.numpy(),
                    score=trainer.controller.scores.numpy(),
                )
                if config.get("save_checkpoints", False):
                    trainer.save_checkpoint(folder / "training.pt")
                rows.append(row)
                pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
                print(
                    f"{config['dataset']} {method} noise={noise} seed={seed} accuracy={row['accuracy']:.4f}",
                    flush=True,
                )
    print(f"RESULTS: {output}", flush=True)
    return output
