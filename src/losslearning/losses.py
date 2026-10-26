"""Stable generalized CE; q is a detached controller output."""

import torch
from torch import nn
from torch.nn import functional as F


class LossLearningLoss(nn.Module):
    def forward(self, logits, targets, q=0.0, reduction="mean"):
        logp = F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        q = torch.as_tensor(q, dtype=logp.dtype, device=logp.device).detach()
        if not torch.isfinite(q).all() or (q < 0).any() or (q > 1).any():
            raise ValueError("q must be finite in [0,1]")
        safe = torch.where(q == 0, torch.ones_like(q), q)
        loss = torch.where(q == 0, -logp, -torch.expm1(q * logp) / safe)
        if reduction == "none":
            return loss
        if reduction != "mean":
            raise ValueError("reduction must be none or mean")
        return loss.mean()


def focal_loss(logits, targets, gamma=2.0):
    logp = F.log_softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
    return (-(1 - logp.exp()).pow(gamma) * logp).mean()
