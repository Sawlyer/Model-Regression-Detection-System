"""Load and validate the versioned golden dataset."""
from __future__ import annotations

import json
from pathlib import Path

from .models import GoldenDataset


def load_dataset(path: Path) -> GoldenDataset:
    """Load the golden dataset from JSON and validate it.

    Raises ValueError if the dataset is empty or contains duplicate case IDs.
    """
    with open(path, encoding="utf-8") as f:
        ds = GoldenDataset.model_validate(json.load(f))
    if not ds.cases:
        raise ValueError("golden dataset has no cases")
    ids = [c.id for c in ds.cases]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate case ids: {dupes}")
    return ds
