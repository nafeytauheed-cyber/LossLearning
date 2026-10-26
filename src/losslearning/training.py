"""Indexed DataLoader training and complete resumable state."""

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .adaptation import AdaptiveLossController, EMATeacher, InfluenceEstimator, ReferenceBank
from .losses import LossLearningLoss, focal_loss


@dataclass
class TrainingConfig:
    epochs: int = 40
    lr: float = 0.003
    weight_decay: float = 0.0001
    warmup: int = 8
    ramp_epochs: int = 8
    adaptive: bool = True
    loss: str = "ce"
    fixed_q: float = 0.7
    focal_gamma: float = 2.0
    q_max: float = 0.8
    tau: float = 0.1
    delta: float = 0.15
    width: float = 0.4
    rho: float = 0.8
    class_aware: bool = True
    threshold_mode: str = "count"
    teacher_beta: float = 0.98
    confidence: float = 0.7
    stability: float = 0.6
    stability_filter: bool = True
    bank_size: int = 256
    refresh_epochs: int = 2
    neighbors: int = 12
    min_neighbors: int = 3
    temperature: float = 0.15
    reference_mode: str = "exact"
    entropy: bool = True
    gradient: bool = True
    confidence_only: bool = False
    consistency: float = 0.0
    regularization: float = 0.0
    clip_grad: float = 5.0
    seed: int = 42
    history_snapshots: int = 40

    def validate(self):
        for key in (
            "epochs",
            "ramp_epochs",
            "bank_size",
            "refresh_epochs",
            "neighbors",
            "min_neighbors",
            "history_snapshots",
        ):
            v = getattr(self, key)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (
            not isinstance(self.warmup, int)
            or self.warmup < 0
            or self.min_neighbors > self.neighbors
        ):
            raise ValueError("invalid warmup/neighborhood configuration")
        for key in ("q_max", "fixed_q", "tau", "delta", "confidence", "stability"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"invalid {key}")
        for key in ("rho", "teacher_beta"):
            if not 0 <= getattr(self, key) < 1:
                raise ValueError(f"invalid {key}")
        for key in ("lr", "width", "temperature", "clip_grad"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"invalid {key}")
        for key in ("weight_decay", "regularization", "consistency", "focal_gamma"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"invalid {key}")
        if self.loss not in ("ce", "gce", "focal") or self.reference_mode not in (
            "exact",
            "cached",
        ):
            raise ValueError("invalid loss/reference mode")
        if self.threshold_mode not in ("count", "quantile"):
            raise ValueError("invalid threshold mode")
        if self.adaptive and self.loss != "ce":
            raise ValueError("disable adaptation for fixed loss baselines")


class LossLearningTrainer:
    """Batches: (x, observed_label, dense_training_id, duplicate_group_id).

    Reference loaders must contain training data only, with stable transforms.
    Validation is used only for logging. No clean labels enter training.
    """

    def __init__(self, model, labels, config=None, device="cpu"):
        self.config = config or TrainingConfig()
        self.config.validate()
        c = self.config
        self.device = torch.device(device)
        self.model = model.to(self.device)
        labels = torch.as_tensor(labels, dtype=torch.long)
        classes = torch.unique(labels)
        if len(classes) < 2 or not torch.equal(classes, torch.arange(len(classes))):
            raise ValueError("observed labels must cover consecutive classes 0..K-1")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        self.teacher = EMATeacher(model, c.teacher_beta)
        self.bank = ReferenceBank(
            len(labels), len(classes), c.bank_size, c.confidence, c.stability, c.stability_filter
        )
        self.controller = AdaptiveLossController(
            labels, c.q_max, c.tau, c.delta, c.width, c.rho, c.class_aware, c.threshold_mode
        )
        self.influence = InfluenceEstimator(
            c.neighbors, c.temperature, c.min_neighbors, c.reference_mode, c.entropy, c.gradient
        )
        self.loss_fn = LossLearningLoss()
        self.history = []
        self.sample_history = []
        self.epoch = 0
        self.step = 0

    def fit(self, train_loader, reference_loader, validation_loader=None, checkpoint_dir=None):
        c = self.config
        start = time.perf_counter()
        for epoch in range(self.epoch, c.epochs):
            torch.manual_seed(c.seed + epoch)
            if train_loader.generator is not None:
                train_loader.generator.manual_seed(c.seed + epoch)
            if c.adaptive and epoch % c.refresh_epochs == 0:
                self.bank.refresh(
                    reference_loader, self.model, self.teacher, self.device, self.step
                )
                self.controller.update_thresholds()
            totals = dict(loss=0.0, ce=0.0, q=0.0, available=0.0, correct=0.0)
            seen = 0
            self.model.train()
            for x, y, ids, groups in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                if ids.min() < 0 or ids.max() >= len(self.controller.labels):
                    raise ValueError("ID out of range")
                if not torch.equal(y.cpu(), self.controller.labels[ids]):
                    raise ValueError("labels do not match stable IDs")
                q = torch.zeros(len(x), device=self.device)
                available = torch.zeros_like(q)
                if c.adaptive and epoch >= c.warmup:
                    if c.confidence_only:
                        with torch.no_grad():
                            p = self.teacher.model(x).softmax(1)
                            score = 1 - p.gather(1, y[:, None]).squeeze(1)
                            available = torch.ones_like(score)
                    else:
                        score, available, _ = self.influence(
                            self.model, x, y, ids, groups, self.bank
                        )
                    q = self.controller.update(
                        ids, score, available, min(1.0, (epoch - c.warmup + 1) / c.ramp_epochs)
                    )
                logits = self.model(x)
                if not c.adaptive and c.loss == "focal":
                    loss = focal_loss(logits, y, c.focal_gamma)
                else:
                    if not c.adaptive and c.loss == "gce":
                        q.fill_(c.fixed_q)
                    loss = self.loss_fn(logits, y, q)
                self.controller.q[ids] = q.detach().cpu()
                if c.consistency:
                    with torch.no_grad():
                        target = self.teacher.model(x).softmax(1)
                    loss = loss + c.consistency * F.kl_div(
                        logits.log_softmax(1), target, reduction="batchmean"
                    )
                if c.regularization:
                    loss = loss + c.regularization * sum(
                        p.square().mean() for p in self.model.parameters()
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.clip_grad)
                self.optimizer.step()
                if c.adaptive or c.consistency:
                    self.teacher.update(self.model)
                self.step += 1
                n = len(x)
                seen += n
                totals["loss"] += float(loss.detach()) * n
                totals["ce"] += float(F.cross_entropy(logits.detach(), y)) * n
                totals["q"] += float(q.sum())
                totals["available"] += float(available.sum())
                totals["correct"] += float((logits.argmax(1) == y).sum())
            record = dict(
                epoch=epoch + 1,
                objective=totals["loss"] / seen,
                train_ce=totals["ce"] / seen,
                mean_q=totals["q"] / seen,
                fallback_fraction=1 - totals["available"] / seen,
                train_accuracy_observed=totals["correct"] / seen,
                bank_size=len(self.bank),
                coverage=self.bank.coverage.tolist(),
                elapsed_seconds=time.perf_counter() - start,
            )
            if validation_loader is not None:
                record.update(self.evaluate(validation_loader, "validation"))
            self.history.append(record)
            self.sample_history.append(
                dict(
                    epoch=epoch + 1,
                    score=self.controller.scores.numpy().copy(),
                    q=self.controller.q.numpy().copy(),
                )
            )
            self.sample_history = self.sample_history[-c.history_snapshots :]
            self.epoch = epoch + 1
            if checkpoint_dir:
                self.save_checkpoint(Path(checkpoint_dir) / "latest.pt")
        self.model.eval()
        return self

    @torch.no_grad()
    def evaluate(self, loader, prefix="test"):
        self.model.eval()
        correct, total, ce = 0, 0, 0.0
        for x, y, *_ in loader:
            x, y = x.to(self.device), y.to(self.device)
            z = self.model(x)
            correct += int((z.argmax(1) == y).sum())
            total += len(y)
            ce += float(F.cross_entropy(z, y, reduction="sum"))
        return {f"{prefix}_accuracy": correct / total, f"{prefix}_ce": ce / total}

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                model=self.model.state_dict(),
                teacher=self.teacher.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                controller=self.controller.__dict__,
                bank=self.bank.__dict__,
                epoch=self.epoch,
                step=self.step,
                history=self.history,
                sample_history=self.sample_history,
                config=asdict(self.config),
            ),
            path,
        )

    def load_checkpoint(self, path):
        """Load only trusted checkpoints; requires the same indexed dataset."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        old = dict(state["config"])
        new = asdict(self.config)
        old.pop("epochs")
        new.pop("epochs")
        if old != new:
            raise ValueError("resume configuration differs except epochs")
        self.model.load_state_dict(state["model"])
        self.teacher.model.load_state_dict(state["teacher"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.controller.__dict__.update(state["controller"])
        self.bank.__dict__.update(state["bank"])
        for obj in (self.controller, self.bank):
            for key, value in obj.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(obj, key, value.cpu())
        if self.bank.data is not None:
            self.bank.data = {k: v.cpu() for k, v in self.bank.data.items()}
        self.epoch, self.step = state["epoch"], state["step"]
        self.history, self.sample_history = state["history"], state["sample_history"]
        return self
