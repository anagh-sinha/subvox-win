"""1-nearest-neighbour DTW template classifier with reject option."""
from dataclasses import dataclass
import numpy as np
from .dtw import dtw_distance


@dataclass
class Result:
    word: str | None
    distance: float
    margin: float


class TemplateClassifier:
    def __init__(self, max_distance: float = float("inf"), min_margin: float = 0.0, band: float = 0.25):
        self.max_distance, self.min_margin, self.band = max_distance, min_margin, band
        self.templates: list[tuple[str, np.ndarray]] = []
        self.scale: np.ndarray | None = None

    def fit(self, samples: list[tuple[str, np.ndarray]]):
        allx = np.vstack([x for _, x in samples])
        self.scale = allx.std(0) + 1e-6
        self.templates = [(w, x / self.scale) for w, x in samples]
        return self

    def distances(self, x: np.ndarray) -> dict[str, float]:
        xs = x / self.scale
        best: dict[str, float] = {}
        for w, t in self.templates:
            d = dtw_distance(xs, t, self.band)
            if d < best.get(w, float("inf")):
                best[w] = d
        return best

    def predict(self, x: np.ndarray) -> Result:
        d = sorted(self.distances(x).items(), key=lambda kv: kv[1])
        if not d:
            return Result(None, float("inf"), 0.0)
        margin = d[1][1] - d[0][1] if len(d) > 1 else float("inf")
        ok = d[0][1] <= self.max_distance and margin >= self.min_margin
        return Result(d[0][0] if ok else None, d[0][1], margin)
