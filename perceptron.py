
import numpy as np


class Perceptron:
    def __init__(self, n_features: int, n_classes: int, lr: float = 0.1, epochs: int = 50):
        self.W = np.zeros((n_classes, n_features + 1), dtype=float)
        self.lr = lr
        self.epochs = epochs
        self.n_classes = n_classes

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        for ep in range(self.epochs):
            for i in range(Xb.shape[0]):
                xi = Xb[i]
                true = int(y[i])
                scores = self.W.dot(xi)
                pred = int(np.argmax(scores))
                if pred != true:
                    self.W[true] += self.lr * xi
                    self.W[pred] -= self.lr * xi

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        scores = self.W.dot(Xb.T)
        preds = np.argmax(scores, axis=0)
        return preds

