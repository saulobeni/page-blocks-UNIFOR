import numpy as np
from typing import Tuple


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    oh = np.zeros((y.size, n_classes))
    for i, val in enumerate(y):
        oh[i, int(val)] = 1.0
    return oh


class MLP:
    def __init__(self, n_features: int, n_hidden: int, n_classes: int, lr: float = 0.01, epochs: int = 100):
        self.W1 = np.random.randn(n_features, n_hidden) * np.sqrt(2.0 / (n_features + n_hidden))
        self.b1 = np.zeros((n_hidden,))
        self.W2 = np.random.randn(n_hidden, n_classes) * np.sqrt(2.0 / (n_hidden + n_classes))
        self.b2 = np.zeros((n_classes,))
        self.lr = lr
        self.epochs = epochs

    def _relu(self, x):
        return np.maximum(0, x)

    def _drelu(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        Y = _one_hot(y, self.b2.shape[0] if hasattr(self, 'b2') else np.unique(y).size)
        n_classes = self.W2.shape[1]
        for ep in range(self.epochs):
            Z1 = X.dot(self.W1) + self.b1
            A1 = self._relu(Z1)
            Z2 = A1.dot(self.W2) + self.b2
            A2 = self._softmax(Z2)
            dZ2 = (A2 - _one_hot(y, n_classes)) / n_samples
            dW2 = A1.T.dot(dZ2)
            db2 = dZ2.sum(axis=0)
            dA1 = dZ2.dot(self.W2.T)
            dZ1 = dA1 * self._drelu(Z1)
            dW1 = X.T.dot(dZ1)
            db1 = dZ1.sum(axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z1 = X.dot(self.W1) + self.b1
        A1 = self._relu(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = self._softmax(Z2)
        return np.argmax(A2, axis=1)
