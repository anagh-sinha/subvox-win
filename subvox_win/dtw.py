"""Dynamic time warping with a Sakoe-Chiba band."""
import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.25) -> float:
    """Length-normalised DTW distance between (T1,D) and (T2,D) sequences."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    w = max(int(band * max(n, m)), abs(n - m)) + 1
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        for j in range(lo, hi + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))
