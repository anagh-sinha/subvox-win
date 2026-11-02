"""Leave-one-out accuracy and label-permutation significance test."""
import numpy as np
from .dtw import dtw_distance


def _dist_matrix(xs, band):
    n = len(xs)
    d = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = dtw_distance(xs[i], xs[j], band)
    return d


def _loo(d, labels):
    pred = labels[np.argmin(d, axis=1)]
    return pred


def evaluate(samples, permutations: int = 500, band: float = 0.25, seed: int = 0):
    """samples: list[(word, features)]. Returns dict with accuracy, p_value, confusion."""
    labels = np.array([w for w, _ in samples])
    allx = np.vstack([x for _, x in samples])
    scale = allx.std(0) + 1e-6
    d = _dist_matrix([x / scale for _, x in samples], band)
    pred = _loo(d, labels)
    acc = float((pred == labels).mean())
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(permutations):
        perm = rng.permutation(labels)
        if (_loo(d, perm) == perm).mean() >= acc:
            ge += 1
    words = sorted(set(labels))
    conf = {t: {p: int(((labels == t) & (pred == p)).sum()) for p in words} for t in words}
    return {"accuracy": acc, "chance": 1 / len(words), "p_value": (ge + 1) / (permutations + 1),
            "confusion": conf, "n": len(samples)}
