"""Tiny helper used by the notebooks to collect all key numbers in one JSON file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _to_plain(obj):
    """Convert numpy types (arrays, floats, ints) to plain python so json can store them."""
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_plain(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and obj != obj:   # NaN is not valid JSON -> store null
        return None
    return obj


def update_results(path: str | Path, section: str, values: dict) -> None:
    """Add (or replace) one section of the results JSON, keeping the other sections."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[section] = _to_plain(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
