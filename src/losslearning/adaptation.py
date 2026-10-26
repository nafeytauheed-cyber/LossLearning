"""EMA teacher, training-only reference bank, gradient evidence, stable-ID history."""

import math
from copy import deepcopy

import torch
from torch.nn import functional as F


class EMATeacher:
    def __init__(self, student, beta=0.98):
        self.model = deepcopy(student).eval().requires_grad_(False)
        self.beta = beta

    @torch.no_grad()
    def update(self, student):
        for a, b in zip(self.model.parameters(), student.parameters()):
            a.lerp_(b, 1 - self.beta)
        for a, b in zip(self.model.buffers(), student.buffers()):
            a.copy_(b)


class AdaptiveLossController:
    def __init__(
        self,
        labels,
        q_max=0.8,
        tau=0.1,
        delta=0.15,
        width=0.4,
        rho=0.8,
        class_aware=True,
        threshold_mode="count",
    ):
        self.labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        counts = torch.bincount(self.labels).float()
        self.thresholds = (
            tau + (delta if class_aware else 0) * (1 - (counts / counts.max()).sqrt())
        ).clamp(0, 1)
        self.base_thresholds = self.thresholds.clone()
        self.scores = torch.zeros(len(labels))
        self.q = torch.zeros(len(labels))
        self.seen = torch.zeros(len(labels), dtype=torch.bool)
        self.q_max, self.width, self.rho = q_max, width, rho
        self.threshold_mode = threshold_mode

    def reset(self):
        self.scores.zero_()
        self.q.zero_()
        self.seen.zero_()
        self.thresholds.copy_(self.base_thresholds)

    def update_thresholds(self):
        if self.threshold_mode == "quantile":
            for c in range(len(self.thresholds)):
                valid = (self.labels == c) & self.seen
                if valid.sum() >= 10:
                    self.thresholds[c] = max(
                        self.base_thresholds[c], torch.quantile(self.scores[valid], 0.8)
                    )

    def update(self, ids, contradiction, available, ramp):
        ids = ids.cpu()
        score = contradiction.detach().cpu().clamp(0, 1)
        mask = available.detach().cpu().float()
        valid = mask > 0
        selected = ids[valid]
        self.scores[selected] = self.rho * self.scores[selected] + (1 - self.rho) * score[valid]
        self.seen[selected] = True
        q = (
            self.q_max
            * ramp
            * mask
            * ((self.scores[ids] - self.thresholds[self.labels[ids]]) / self.width).clamp(0, 1)
        )
        self.q[ids] = q
        return q.to(contradiction.device)


class ReferenceBank:
    """Equal class quotas never override confidence/stability requirements."""

    def __init__(
        self,
        n_samples,
        n_classes,
        max_size=256,
        confidence=0.7,
        stability=0.6,
        stability_filter=True,
    ):
        if max_size < n_classes:
            raise ValueError("bank_size must cover number of classes")
        self.n_classes, self.max_size = n_classes, max_size
        self.confidence, self.stability, self.stability_filter = (
            confidence,
            stability,
            stability_filter,
        )
        self.last = torch.full((n_samples,), -1, dtype=torch.long)
        self.agreement = torch.zeros(n_samples)
        self.observations = torch.zeros(n_samples, dtype=torch.long)
        self.data = None
        self.last_refresh = -1
        self.coverage = torch.zeros(n_classes, dtype=torch.long)

    def __len__(self):
        return 0 if self.data is None else len(self.data["ids"])

    @torch.no_grad()
    def refresh(self, loader, student, teacher, device, step):
        pools = [None] * self.n_classes
        capacity = self.max_size // self.n_classes
        for x, _, ids, groups in loader:
            x = x.to(device)
            confidence, label = teacher.model(x).softmax(1).max(1)
            confidence, label = confidence.cpu(), label.cpu()
            self.agreement[ids] = (
                0.5 * self.agreement[ids] + 0.5 * (self.last[ids] == label).float()
            )
            self.observations[ids] += 1
            self.last[ids] = label
            eligible = confidence >= self.confidence
            if self.stability_filter:
                eligible &= (self.observations[ids] >= 2) & (self.agreement[ids] >= self.stability)
            h, psi = student.features(x)
            batch = dict(
                x=x.cpu(),
                h=h.cpu(),
                psi=psi.cpu(),
                ids=ids.cpu(),
                groups=groups.cpu(),
                labels=label,
                confidence=confidence,
                stability=self.agreement[ids].clone(),
            )
            for c in range(self.n_classes):
                candidate = {k: v[eligible & (label == c)] for k, v in batch.items()}
                if pools[c] is not None:
                    candidate = {k: torch.cat((pools[c][k], v)) for k, v in candidate.items()}
                keep, seen = [], set()
                for j in candidate["confidence"].argsort(descending=True).tolist():
                    group = int(candidate["groups"][j])
                    if group not in seen:
                        keep.append(j)
                        seen.add(group)
                    if len(keep) >= capacity:
                        break
                idx = torch.tensor(keep, dtype=torch.long)
                pools[c] = {k: v[idx] for k, v in candidate.items()}
        nonempty = [p for p in pools if p is not None and len(p["ids"])]
        self.data = (
            {k: torch.cat([p[k] for p in nonempty]) for k in nonempty[0]} if nonempty else None
        )
        self.coverage = torch.tensor([len(p["ids"]) if p is not None else 0 for p in pools])
        self.last_refresh = step


def gradient_similarity(residual, psi, ref_residual, ref_psi, cosine=True):
    if cosine:
        residual = F.normalize(residual, dim=-1, eps=1e-8)
        ref_residual = F.normalize(ref_residual, dim=-1, eps=1e-8)
        psi = F.normalize(psi, dim=-1, eps=1e-8)
        ref_psi = F.normalize(ref_psi, dim=-1, eps=1e-8)
    return (residual @ ref_residual.T) * (psi @ ref_psi.T)


class InfluenceEstimator:
    def __init__(
        self,
        neighbors=12,
        temperature=0.15,
        min_neighbors=3,
        mode="exact",
        entropy=True,
        gradient=True,
        chunk_size=128,
    ):
        self.neighbors, self.temperature, self.min_neighbors = neighbors, temperature, min_neighbors
        self.mode, self.entropy, self.gradient, self.chunk_size = (
            mode,
            entropy,
            gradient,
            chunk_size,
        )

    @torch.no_grad()
    def __call__(self, model, x, y, ids, groups, bank):
        device, b = x.device, len(x)
        zero = torch.zeros(b, device=device)
        if not len(bank):
            return zero, zero, {"weights": torch.empty(b, 0, device=device)}
        h, psi = model.features(x)
        residual = model.head.linear(psi).softmax(1) - F.one_hot(y, bank.n_classes)
        k = min(self.neighbors, len(bank))
        best = torch.full((b, k), -torch.inf, device=device)
        best_indices = torch.zeros((b, k), dtype=torch.long, device=device)
        psis = []
        for start in range(0, len(bank), self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            if self.mode == "exact":
                rh, rp = model.features(bank.data["x"][sl].to(device))
            else:
                rh, rp = bank.data["h"][sl].to(device), bank.data["psi"][sl].to(device)
            psis.append(rp)
            sim = h @ rh.T
            invalid = (ids.to(device)[:, None] == bank.data["ids"][sl].to(device)[None]) | (
                groups.to(device)[:, None] == bank.data["groups"][sl].to(device)[None]
            )
            sim.masked_fill_(invalid, -torch.inf)
            indices = torch.arange(start, start + len(rh), device=device).expand(b, -1)
            all_indices = torch.cat((best_indices, indices), 1)
            best, pos = torch.cat((best, sim), 1).topk(k, dim=1)
            best_indices = all_indices.gather(1, pos)
        rp = torch.cat(psis)[best_indices]
        labels = bank.data["labels"].to(device)[best_indices]
        valid = torch.isfinite(best)
        weights = (
            torch.softmax(
                torch.where(valid, best / self.temperature, torch.full_like(best, -1e9)), 1
            )
            * valid
        )
        weights /= weights.sum(1, keepdim=True).clamp_min(1e-8)
        rr = model.head.linear(rp).softmax(-1) - F.one_hot(labels, bank.n_classes)
        cosine = (
            F.normalize(residual, dim=-1, eps=1e-8)[:, None] * F.normalize(rr, dim=-1, eps=1e-8)
        ).sum(-1) * (
            F.normalize(psi, dim=-1, eps=1e-8)[:, None] * F.normalize(rp, dim=-1, eps=1e-8)
        ).sum(-1)
        u = (weights * cosine).sum(1)
        pi = (weights[..., None] * F.one_hot(labels, bank.n_classes)).sum(1)
        entropy = -(pi * pi.clamp_min(1e-12).log()).sum(1) / math.log(bank.n_classes)
        other = pi.clone()
        other.scatter_(1, y[:, None], 0)
        conflict = (-u).clamp(0, 1) if self.gradient else torch.ones_like(u)
        score = conflict * ((1 - entropy).clamp(0, 1) if self.entropy else 1) * other.max(1).values
        available = (
            (valid.sum(1) >= self.min_neighbors)
            & (bank.coverage.to(device)[y] >= self.min_neighbors)
        ).float()
        return (
            score.clamp(0, 1) * available,
            available,
            {"weights": weights, "u": u, "entropy": entropy},
        )
