"""Array API; larger datasets use LossLearningTrainer with DataLoaders."""

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

from .models import LossLearningNet
from .training import LossLearningTrainer, TrainingConfig


class LossLearning(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        encoder="mlp",
        embedding_dim=32,
        hidden_dim=64,
        n_prototypes=16,
        head="linear",
        adaptive_loss=True,
        epochs=40,
        batch_size=128,
        learning_rate=0.003,
        random_state=42,
        device="auto",
        standardize=True,
        freeze_encoder=False,
        trainer_options=None,
    ):
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_prototypes = n_prototypes
        self.head = head
        self.adaptive_loss = adaptive_loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device
        self.standardize = standardize
        self.freeze_encoder = freeze_encoder
        self.trainer_options = trainer_options

    def _validate_X(self, X):
        X = check_array(X, dtype=np.float32, allow_nd=self.encoder == "cnn")
        if X.ndim != (4 if self.encoder == "cnn" else 2):
            raise ValueError("invalid input dimensions")
        return X

    def fit(self, X, y, X_val=None, y_val=None):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have one label per sample")
        check_classification_targets(y)
        for name in ("batch_size", "embedding_dim", "hidden_dim", "n_prototypes"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"invalid {name}")
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) < 2:
            raise ValueError("at least two classes required")
        labels = self.label_encoder_.transform(y)
        self.n_features_in_ = X.shape[1]
        self.input_shape_ = tuple(X.shape[1:])
        self.scaler_ = (
            StandardScaler().fit(X) if self.encoder == "mlp" and self.standardize else None
        )
        X = self.scaler_.transform(X).astype(np.float32) if self.scaler_ else X
        _, groups = np.unique(X.reshape(len(X), -1), axis=0, return_inverse=True)
        seed = self.random_state
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device_ = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == "auto"
            else self.device
        )
        options = dict(self.trainer_options or {})
        if {"epochs", "lr", "seed", "adaptive"} & options.keys():
            raise ValueError("reserved trainer option: use estimator argument")
        cfg = TrainingConfig(
            epochs=self.epochs,
            lr=self.learning_rate,
            seed=seed,
            adaptive=self.adaptive_loss,
            **options,
        )
        cfg.validate()
        self.network_config_ = dict(
            input_shape=self.input_shape_,
            n_classes=len(self.classes_),
            encoder=self.encoder,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_prototypes=self.n_prototypes,
            head=self.head,
            freeze_encoder=self.freeze_encoder,
        )
        self.model_ = LossLearningNet(**self.network_config_)
        self.trainer_ = LossLearningTrainer(self.model_, labels, cfg, self.device_)
        dataset = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X)),
            torch.tensor(labels),
            torch.arange(len(X)),
            torch.tensor(groups),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        ref = DataLoader(dataset, batch_size=self.batch_size)
        val = None
        if (X_val is None) != (y_val is None):
            raise ValueError("supply both validation inputs and labels")
        if X_val is not None:
            xv = self._prepare(X_val)
            yv = self.label_encoder_.transform(y_val)
            val = DataLoader(
                TensorDataset(torch.from_numpy(xv), torch.tensor(yv)), batch_size=self.batch_size
            )
        self.trainer_.fit(loader, ref, val)
        self.history_ = self.trainer_.history
        self.contradiction_scores_ = self.trainer_.controller.scores.numpy().copy()
        self.q_ = self.trainer_.controller.q.numpy().copy()
        return self

    def _prepare(self, X):
        X = self._validate_X(X)
        if tuple(X.shape[1:]) != self.input_shape_:
            raise ValueError("input shape differs from training")
        if self.scaler_:
            X = self.scaler_.transform(X)
        return np.ascontiguousarray(X, dtype=np.float32)

    @torch.no_grad()
    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = self._prepare(X)
        self.model_.eval()
        out = []
        for start in range(0, len(X), self.batch_size):
            batch = torch.from_numpy(X[start : start + self.batch_size]).to(self.device_)
            out.append(self.model_(batch).softmax(1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[p.argmax(1)]

    def save(self, path):
        check_is_fitted(self, "model_")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                params=self.get_params(),
                network_config=self.network_config_,
                model=self.model_.state_dict(),
                classes=self.classes_,
                scaler=self.scaler_,
                history=self.history_,
                scores=self.contradiction_scores_,
                q=self.q_,
            ),
            path,
        )

    @classmethod
    def load(cls, path, device="cpu"):
        """Restore inference from a trusted file only (pickle is unsafe for unknown files)."""
        s = torch.load(path, map_location=device, weights_only=False)
        params = dict(s["params"])
        params["device"] = device
        obj = cls(**params)
        obj.network_config_ = s["network_config"]
        obj.model_ = LossLearningNet(**obj.network_config_).to(device)
        obj.model_.load_state_dict(s["model"])
        obj.model_.eval()
        obj.classes_ = s["classes"]
        obj.label_encoder_ = LabelEncoder().fit(obj.classes_)
        obj.input_shape_ = tuple(obj.network_config_["input_shape"])
        obj.n_features_in_ = obj.input_shape_[0]
        obj.device_, obj.scaler_, obj.history_ = device, s["scaler"], s["history"]
        obj.contradiction_scores_, obj.q_ = s["scores"], s["q"]
        return obj
