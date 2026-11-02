"""Persist recorded word templates as JSON."""
import json
from pathlib import Path
import numpy as np

DEFAULT = Path("data") / "dataset.json"


def load(path: Path = DEFAULT) -> list[tuple[str, np.ndarray]]:
    if not Path(path).exists():
        return []
    return [(r["word"], np.array(r["features"])) for r in json.loads(Path(path).read_text())]


def save(samples, path: Path = DEFAULT):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([{"word": w, "features": x.round(5).tolist()} for w, x in samples]))
