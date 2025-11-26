import random
from collections import defaultdict
from typing import List, Tuple


def stratified_k_fold(y: List[int], k: int, seed: int = 42):
    """Yield (train_idx, test_idx) for stratified k-fold."""
    random.seed(seed)
    label_to_idx = defaultdict(list)
    for i, lab in enumerate(y):
        label_to_idx[lab].append(i)
    folds = [[] for _ in range(k)]
    for lab, idxs in label_to_idx.items():
        idxs_copy = idxs[:]
        random.shuffle(idxs_copy)
        for i, idx in enumerate(idxs_copy):
            folds[i % k].append(idx)
    for i in range(k):
        test_idx = folds[i]
        train_idx = []
        for j in range(k):
            if j != i:
                train_idx.extend(folds[j])
        yield train_idx, test_idx