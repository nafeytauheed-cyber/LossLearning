"""Autograd agreement, finite-step approximation, and stale reference diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .adaptation import InfluenceEstimator, gradient_similarity


def validate_influence(output):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    psi = torch.cat(
        (torch.randn(48, 12, dtype=torch.double), torch.ones(48, 1, dtype=torch.double)), 1
    )
    weight = (torch.randn(3, 13, dtype=torch.double) * 0.25).requires_grad_()
    y = torch.randint(3, (48,))
    z = psi @ weight.T
    r = z.softmax(1) - F.one_hot(y, 3)
    gradients = torch.stack(
        [
            torch.autograd.grad(
                F.cross_entropy(z[i : i + 1], y[i : i + 1]), weight, retain_graph=True
            )[0]
            for i in range(48)
        ]
    )
    dot = gradient_similarity(r, psi, r, psi, False).detach()
    error = float((dot - gradients.flatten(1) @ gradients.flatten(1).T).abs().max())
    original = F.cross_entropy(z, y, reduction="none").detach()
    rows = []
    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]:
        measured, predicted = [], []
        for i in range(12):
            actual = (
                F.cross_entropy(psi @ (weight.detach() - lr * gradients[i]).T, y, reduction="none")
                - original
            )
            measured.extend(actual.tolist())
            predicted.extend((-lr * dot[i]).tolist())
        a, p = np.array(measured), np.array(predicted)
        rows.append(
            dict(
                lr=lr,
                mae=float(abs(a - p).mean()),
                relative_l2=float(np.linalg.norm(a - p) / np.linalg.norm(a)),
                correlation=float(np.corrcoef(a, p)[0, 1]),
                sign_agreement=float(np.mean(np.sign(a) == np.sign(p))),
            )
        )
    report = dict(max_factorization_error=error, n_pairs=576, dtype="float64", results=rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "influence.json").write_text(json.dumps(report, indent=2))
    from .visualization import plt

    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.loglog([r["lr"] for r in rows], [r["relative_l2"] for r in rows], "o-", color="#146b76")
    ax.set(
        xlabel="Hypothetical head-only SGD step",
        ylabel="Relative L2 prediction error",
        title="Taylor accuracy degrades with step size",
    )
    ax.grid(alpha=0.2)
    fig.savefig(output / "influence_error.png", dpi=160)
    plt.close(fig)
    return report


def stale_diagnostic(output):
    from .data import split_geometry
    from .estimator import LossLearning

    rows = []
    for refresh in [1, 2, 8]:
        split = split_geometry("moons", 600, 42, 0.2)
        m = LossLearning(
            head="potential",
            epochs=30,
            trainer_options={"refresh_epochs": refresh, "confidence": 0.6},
        ).fit(split["X_train"], split["y_train"])
        x = torch.from_numpy(m._prepare(split["X_train"]))
        y = torch.tensor(split["y_train"])
        ids = torch.arange(len(x))
        _, g = np.unique(x.numpy(), axis=0, return_inverse=True)
        g = torch.tensor(g)
        bank = m.trainer_.bank
        exact, mask, _ = InfluenceEstimator(mode="exact")(m.model_, x, y, ids, g, bank)
        cached, _, _ = InfluenceEstimator(mode="cached")(m.model_, x, y, ids, g, bank)
        rows.append(
            dict(
                refresh_epochs=refresh,
                bank_size=len(bank),
                steps_since_refresh=m.trainer_.step - bank.last_refresh,
                availability=float(mask.mean()),
                score_mae=float((exact - cached).abs().mean()),
                score_max_error=float((exact - cached).abs().max()),
            )
        )
    Path(output, "stale_reference.json").write_text(json.dumps(rows, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/influence")
    args = parser.parse_args()
    print(json.dumps(validate_influence(args.output), indent=2))
    print(json.dumps(stale_diagnostic(args.output), indent=2))


if __name__ == "__main__":
    main()
