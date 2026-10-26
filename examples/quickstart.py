import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from losslearning import LossLearning

torch.set_num_threads(1)
X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = LossLearning(encoder="mlp", adaptive_loss=True, random_state=42)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))
model.save("checkpoints/losslearning.pt")
restored = LossLearning.load("checkpoints/losslearning.pt")
assert (restored.predict(X_test) == model.predict(X_test)).all()
