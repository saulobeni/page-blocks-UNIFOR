import numpy as np
from typing import Callable


class KNN:
    def __init__(self, k: int = 5, metric: str = 'euclidean'):
        self.k = k
        if metric not in ('euclidean', 'manhattan'):
            raise ValueError('metric must be euclidean or manhattan')
        self.metric = metric
        self.X = None
        self.y = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X.copy()
        self.y = y.copy()

    def _dist(self, a: np.ndarray, b: np.ndarray):
        if self.metric == 'euclidean':
            return np.sqrt(((a - b) ** 2).sum(axis=1))
        else:
            return np.abs(a - b).sum(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for i in range(X.shape[0]):
            d = self._dist(self.X, X[i, :])
            idx = np.argsort(d)[: self.k]
            labs = self.y[idx]
            vals, counts = np.unique(labs, return_counts=True)
            choice = vals[np.argmax(counts)]
            preds.append(choice)
        return np.array(preds)