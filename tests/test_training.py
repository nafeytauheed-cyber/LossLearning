import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.datasets import make_moons
from sklearn.exceptions import NotFittedError
from torch.utils.data import DataLoader, TensorDataset

from losslearning import LossLearning, LossLearningNet
from losslearning.data import corrupt_labels, split_geometry
from losslearning.training import LossLearningTrainer, TrainingConfig


def opts():
    return dict(
        epochs=4,
        batch_size=32,
        hidden_dim=12,
        embedding_dim=8,
        trainer_options=dict(
            warmup=1,
            refresh_epochs=1,
            confidence=0.5,
            stability_filter=False,
            bank_size=32,
            min_neighbors=1,
        ),
    )


def test_train_save_reproduce(tmp_path):
    X, y = make_moons(100, noise=0.15, random_state=4)
    y = y.astype(str)
    m = LossLearning(**opts()).fit(X, y, X_val=X[:10], y_val=y[:10])
    p = m.predict_proba(X)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    assert len(m.history_) == 4
    assert m.history_[-1]["bank_size"] > 0 and "validation_ce" in m.history_[-1]
    m.save(tmp_path / "m.pt")
    restored = LossLearning.load(tmp_path / "m.pt")
    np.testing.assert_allclose(restored.predict_proba(X), p, rtol=0, atol=0)
    twin = clone(m).fit(X, y)
    np.testing.assert_allclose(twin.predict_proba(X), p, rtol=0, atol=0)
    assert 0 <= m.score(X, y) <= 1


def test_invalid_inputs():
    X, y = make_moons(20, random_state=1)
    with pytest.raises(NotFittedError):
        LossLearning().predict(X)
    for params in [
        dict(epochs=0),
        dict(batch_size=0),
        dict(head="bad"),
        dict(trainer_options={"width": 0}),
    ]:
        with pytest.raises(ValueError):
            LossLearning(**params).fit(X, y)
    with pytest.raises(ValueError):
        LossLearning().fit(X, np.zeros(20))
    with pytest.raises(ValueError):
        LossLearning().fit(X[:3], y)
    with pytest.raises(ValueError):
        LossLearning().fit(X * np.nan, y)


def test_cnn_frozen_cpu():
    X = np.random.default_rng(1).random((20, 1, 8, 8), dtype=np.float32)
    y = np.arange(20) % 2
    m = LossLearning(encoder="cnn", epochs=1, freeze_encoder=True, adaptive_loss=False).fit(X, y)
    assert m.predict_proba(X).shape == (20, 2) and all(
        p.grad is None for p in m.model_.encoder.parameters()
    )


def test_fixed_q_logging_and_history_bound():
    X, y = make_moons(40, random_state=2)
    m = LossLearning(
        epochs=4, adaptive_loss=False, trainer_options={"loss": "gce", "history_snapshots": 2}
    ).fit(X, y)
    np.testing.assert_allclose(m.q_, 0.7)
    assert len(m.trainer_.sample_history) == 2


def test_resume(tmp_path):
    X, y = make_moons(60, random_state=3)
    labels = torch.tensor(y)
    ids = torch.arange(60)
    data = TensorDataset(torch.tensor(X, dtype=torch.float32), labels, ids, ids)

    def create(epochs):
        torch.manual_seed(42)
        return LossLearningTrainer(
            LossLearningNet((2,), 2),
            labels,
            TrainingConfig(
                epochs=epochs,
                warmup=1,
                refresh_epochs=1,
                confidence=0.5,
                stability_filter=False,
                bank_size=32,
            ),
        )

    def fit(t):
        t.fit(
            DataLoader(data, batch_size=20, shuffle=True, generator=torch.Generator()),
            DataLoader(data, batch_size=20),
        )

    full = create(4)
    fit(full)
    part = create(2)
    fit(part)
    part.save_checkpoint(tmp_path / "state.pt")
    resume = create(4).load_checkpoint(tmp_path / "state.pt")
    fit(resume)
    for a, b in zip(full.model.parameters(), resume.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(full.controller.scores, resume.controller.scores)


def test_split_no_leakage():
    a = split_geometry("moons", 200, 3, 0.4)
    b = split_geometry("moons", 200, 3, 0)
    for x, y in [("train_ids", "test_ids"), ("train_ids", "val_ids"), ("val_ids", "test_ids")]:
        assert not set(a[x]) & set(a[y])
    np.testing.assert_array_equal(a["y_test"], b["y_test"])
    np.testing.assert_array_equal(a["noise_mask"], a["y_train"] != a["clean_train"])
    with pytest.raises(ValueError):
        corrupt_labels(np.array([0, 1]), -1, 1)


def test_tiny_complete_benchmark(tmp_path):
    from losslearning.experiments import run_benchmark

    output = run_benchmark(
        dict(
            output=str(tmp_path),
            n_samples=80,
            seeds=[1],
            datasets=["moons"],
            noise_rates=[0.2],
            methods=["ce", "full"],
            model={"epochs": 2},
            trainer_options={
                "warmup": 1,
                "confidence": 0.5,
                "stability_filter": False,
                "bank_size": 32,
            },
        )
    )
    assert (output / "summary.csv").exists() and (output / "benchmark.png").exists()
