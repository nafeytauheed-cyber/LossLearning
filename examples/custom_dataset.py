from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from losslearning import LossLearning

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = LossLearning(epochs=40).fit(X_train, y_train)
print("Iris accuracy:", model.score(X_test, y_test))
