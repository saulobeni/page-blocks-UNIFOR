import math
import numpy as np
from typing import Tuple


def zscore_normalize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    Xtr = (X_train - mean) / std
    Xte = (X_test - mean) / std
    return Xtr, Xte


class LabelEncoder:
    def __init__(self):
        self.classes_ = []
        self.mapping = {}

    def fit(self, y):
        uniq = []
        for v in y:
            if v not in uniq:
                uniq.append(v)
        self.classes_ = uniq
        self.mapping = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, y):
        return [self.mapping[v] for v in y]

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y_idx):
        return [self.classes_[i] for i in y_idx]