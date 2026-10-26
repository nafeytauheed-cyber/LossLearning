import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split


def make_geometry(name="moons", n_samples=1000, seed=42):
    if name == "moons":
        return make_moons(n_samples=n_samples, noise=0.22, random_state=seed)
    if name == "circles":
        return make_circles(n_samples=n_samples, noise=0.1, factor=0.45, random_state=seed)
    if name == "blobs":
        return make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=seed)
    if name == "overlap":
        return make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=0.7,
            flip_y=0,
            random_state=seed,
        )
    if name in ("disconnected", "rare"):
        counts = (
            [n_samples // 4] * 3 + [n_samples - 3 * (n_samples // 4)]
            if name == "disconnected"
            else [
                int(n_samples * 0.45),
                int(n_samples * 0.45),
                int(n_samples * 0.08),
                n_samples - 2 * int(n_samples * 0.45) - int(n_samples * 0.08),
            ]
        )
        X, clusters = make_blobs(
            n_samples=counts,
            centers=[[-2, -1], [2, 1], [-2, 1], [0, 0]],
            cluster_std=[0.5, 0.5, 0.3, 0.18],
            random_state=seed,
        )
        return X, np.array([0, 0, 1, 1])[clusters]
    raise ValueError(f"unknown geometry {name}")


def corrupt_labels(y, rate, seed, mode="symmetric"):
    if not 0 <= rate <= 1 or mode not in ("symmetric", "class_dependent"):
        raise ValueError("invalid corruption")
    y = np.asarray(y)
    out = y.copy()
    k = int(y.max()) + 1
    if k < 2:
        raise ValueError("at least two classes required")
    rng = np.random.default_rng(seed)
    rates = np.full(len(y), rate, dtype=float)
    if mode == "class_dependent":
        rates *= (y + 1) / k
    mask = rng.random(len(y)) < rates
    out[mask] = (
        out[mask] + (rng.integers(1, k, size=mask.sum()) if mode == "symmetric" else 1)
    ) % k
    return out, mask


def split_geometry(name, n_samples, seed, noise_rate, noise_mode="symmetric"):
    X, y = make_geometry(name, n_samples, seed)
    idx = np.arange(len(y))
    train, test = train_test_split(idx, test_size=0.25, stratify=y, random_state=seed)
    train, val = train_test_split(train, test_size=0.2, stratify=y[train], random_state=seed + 1)
    observed, mask = corrupt_labels(y[train], noise_rate, seed + 1000, noise_mode)
    return dict(
        X_train=X[train],
        y_train=observed,
        X_val=X[val],
        y_val=y[val],
        X_test=X[test],
        y_test=y[test],
        clean_train=y[train],
        noise_mask=mask,
        train_ids=train,
        val_ids=val,
        test_ids=test,
    )
