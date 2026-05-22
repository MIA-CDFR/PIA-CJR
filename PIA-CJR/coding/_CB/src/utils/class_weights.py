# ============================================================
# FILE: src/utils/class_weights.py
# ============================================================

from collections import Counter

import torch


def compute_class_weights(labels):
    """
    Compute balanced class weights automatically.

    Parameters
    ----------
    labels : list[int]

    Returns
    -------
    torch.Tensor
    """

    counter = Counter(labels)

    total_samples = sum(counter.values())

    num_classes = len(counter)

    weights = []

    for class_idx in sorted(counter.keys()):

        class_count = counter[class_idx]

        weight = total_samples / (num_classes * class_count)

        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)
