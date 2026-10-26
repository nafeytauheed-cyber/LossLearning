import pytest
import torch
from torch.nn import functional as F

from losslearning import LossLearningLoss, LossLearningNet, PotentialFieldHead
from losslearning.adaptation import (
    AdaptiveLossController,
    EMATeacher,
    InfluenceEstimator,
    ReferenceBank,
    gradient_similarity,
)


@pytest.mark.parametrize("q", [0.0, 1e-12, 0.2, 0.7, 1.0])
def test_loss_gradient(q):
    torch.manual_seed(3)
    z = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    y = torch.tensor([0, 1, 2, 0, 1])
    loss = LossLearningLoss()
    assert torch.autograd.gradcheck(lambda a: loss(a, y, q), (z,))
    g = torch.autograd.grad(loss(z, y, q), z)[0]
    p = z.softmax(1)
    torch.testing.assert_close(g, p.gather(1, y[:, None]).pow(q) * (p - F.one_hot(y, 3)) / len(y))
    if q < 1e-8:
        torch.testing.assert_close(loss(z, y, q), F.cross_entropy(z, y))


def test_extreme_logits_detach():
    z = torch.tensor([[1e4, -1e4], [-1e4, 1e4], [0.0, 0.0]], requires_grad=True)
    q = torch.tensor([0.0, 0.7, 1e-8], requires_grad=True)
    loss = LossLearningLoss()(z, torch.tensor([1, 0, 1]), q)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and q.grad is None
    with pytest.raises(ValueError):
        LossLearningLoss()(z, torch.tensor([1, 0, 1]), -0.2)


def test_potential_parameters():
    head = PotentialFieldHead(4, 3, 5)
    x = torch.randn(7, 4, requires_grad=True)
    assert head(x).shape == (7, 3)
    head(x).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    with torch.no_grad():
        head.raw_scale.fill_(-100)
        head.raw_power.fill_(100)
    assert (head.scales > 0).all() and (head.powers > 0).all() and torch.isfinite(head(x)).all()


@pytest.mark.parametrize("head", ["linear", "potential"])
def test_factorization(head):
    torch.manual_seed(7)
    m = LossLearningNet((2,), 3, head=head).double()
    x = torch.randn(6, 2, dtype=torch.double)
    y = torch.arange(6) % 3
    _, psi = m.features(x)
    z = m(x)
    g = torch.stack(
        [
            torch.autograd.grad(
                F.cross_entropy(z[i : i + 1], y[i : i + 1]), m.head.linear.weight, retain_graph=True
            )[0].flatten()
            for i in range(6)
        ]
    )
    r = z.softmax(1) - F.one_hot(y, 3)
    torch.testing.assert_close(gradient_similarity(r, psi, r, psi, False), g @ g.T)
    norm = F.normalize(g, dim=1)
    torch.testing.assert_close(gradient_similarity(r, psi, r, psi), norm @ norm.T)


def test_zero_gradient():
    assert not gradient_similarity(
        torch.zeros(2, 3), torch.ones(2, 4), torch.zeros(2, 3), torch.ones(2, 4)
    ).any()


def test_weighted_ce_equivalence():
    z = torch.randn(6, 3, dtype=torch.double, requires_grad=True)
    y = torch.arange(6) % 3
    q = torch.linspace(0, 1, 6, dtype=torch.double)
    a = torch.autograd.grad(LossLearningLoss()(z, y, q), z)[0]
    weight = z.softmax(1).gather(1, y[:, None]).squeeze(1).pow(q).detach()
    b = torch.autograd.grad((weight * F.cross_entropy(z, y, reduction="none")).mean(), z)[0]
    torch.testing.assert_close(a, b)


def test_controller_ids():
    c = AdaptiveLossController([0, 0, 0, 1], tau=0, delta=0.2, rho=0.5)
    assert c.thresholds[1] > c.thresholds[0]
    q = c.update(torch.tensor([3, 1]), torch.tensor([0.8, 0.4]), torch.ones(2), 1.0)
    torch.testing.assert_close(c.scores, torch.tensor([0.0, 0.2, 0.0, 0.4]))
    assert q.min() >= 0 and q.max() <= 0.8
    q = c.update(torch.tensor([3]), torch.tensor([1.0]), torch.zeros(1), 1.0)
    assert q.item() == 0 and c.scores[3] == 0.4
    c.reset()
    assert not c.scores.any()


def test_empty_bank():
    m = LossLearningNet((2,), 2)
    b = ReferenceBank(10, 2)
    score, avail, _ = InfluenceEstimator()(
        m, torch.randn(3, 2), torch.tensor([0, 1, 0]), torch.arange(3), torch.arange(3), b
    )
    assert not score.any() and not avail.any()


@pytest.mark.parametrize("mode", ["exact", "cached"])
def test_reference_exclusion(mode):
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(5)
    m = LossLearningNet((2,), 2)
    t = EMATeacher(m)
    x = torch.randn(20, 2)
    y = torch.arange(20) % 2
    ids = torch.arange(20)
    bank = ReferenceBank(20, 2, 16, confidence=0, stability_filter=False)
    bank.refresh(DataLoader(TensorDataset(x, y, ids, ids), batch_size=5), m, t, "cpu", 0)
    assert len(bank) <= 16
    score, avail, info = InfluenceEstimator(neighbors=4, min_neighbors=1, mode=mode)(
        m, x, y, ids, ids, bank
    )
    assert ((score >= 0) & (score <= 1)).all() and torch.isfinite(score).all()
    torch.testing.assert_close(info["weights"].sum(1), torch.ones(20))
    bank.data = {k: v[:1] for k, v in bank.data.items()}
    i = bank.data["ids"]
    score, avail, info = InfluenceEstimator(min_neighbors=1)(m, x[i], y[i], i, i, bank)
    assert not avail.any() and not info["weights"].any()


def test_teacher_update():
    m = LossLearningNet((2,), 2)
    t = EMATeacher(m, 0.5)
    old = [p.clone() for p in t.model.parameters()]
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1)
    t.update(m)
    for p, o in zip(t.model.parameters(), old):
        torch.testing.assert_close(p, o + 0.5)


def test_duplicate_group_cannot_corroborate_itself():
    m = LossLearningNet((2,), 2)
    bank = ReferenceBank(3, 2)
    x = torch.tensor([[0.2, 0.3], [0.2, 0.3]])
    h, psi = m.features(x)
    bank.data = dict(
        x=x,
        h=h.detach(),
        psi=psi.detach(),
        ids=torch.tensor([1, 2]),
        groups=torch.tensor([7, 7]),
        labels=torch.tensor([0, 1]),
    )
    bank.coverage = torch.tensor([1, 1])
    score, available, info = InfluenceEstimator(min_neighbors=1)(
        m, x[:1], torch.tensor([0]), torch.tensor([0]), torch.tensor([7]), bank
    )
    assert not score.any() and not available.any() and not info["weights"].any()


def test_stability_filter_and_missing_class_gate():
    from torch.utils.data import DataLoader, TensorDataset

    m = LossLearningNet((2,), 2)
    with torch.no_grad():
        m.head.linear.weight.zero_()
        m.head.linear.weight[0, -1] = 8
    t = EMATeacher(m)
    x = torch.randn(10, 2)
    y = torch.arange(10) % 2
    ids = torch.arange(10)
    loader = DataLoader(TensorDataset(x, y, ids, ids), batch_size=5)
    bank = ReferenceBank(10, 2, confidence=0.5, stability=0.6)
    bank.refresh(loader, m, t, "cpu", 0)
    assert len(bank) == 0
    bank.refresh(loader, m, t, "cpu", 1)
    assert len(bank) == 0
    bank.refresh(loader, m, t, "cpu", 2)
    assert len(bank) == 10
    score, available, _ = InfluenceEstimator(min_neighbors=1)(m, x, y, ids, ids, bank)
    assert not available[y == 1].any() and not score[y == 1].any()
