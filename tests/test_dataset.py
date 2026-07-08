import json
from collections import Counter
from pathlib import Path

import pytest

from regression_detector.dataset import load_dataset
from regression_detector.models import Category

REAL_DATASET = Path(__file__).parent.parent / "data" / "golden_dataset.json"


def test_load_real_golden_dataset():
    ds = load_dataset(REAL_DATASET)
    assert len(ds.cases) >= 50
    cats = Counter(c.expected_category for c in ds.cases)
    for cat in Category:
        assert cats[cat] >= 10, f"category {cat.value} under-represented"
    diffs = Counter(c.expected_difficulty for c in ds.cases)
    assert diffs["hard"] >= 8, "need deliberate edge cases"
    assert all(c.notes for c in ds.cases if c.expected_difficulty == "hard")


def test_duplicate_ids_rejected(tmp_path: Path):
    case = {
        "id": "tc-001", "input": "x", "expected_category": "general",
        "ideal_summary": "s", "expected_difficulty": "easy", "notes": "",
    }
    bad = {"version": "1.0.0", "cases": [case, case]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(p)
