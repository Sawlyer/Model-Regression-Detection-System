from datetime import datetime, timezone

import pytest

from regression_detector.drift import detect_drift
from regression_detector.models import EvalRun

NOW = datetime.now(timezone.utc)


def _run(rate: float) -> EvalRun:
    return EvalRun(run_id="r", prompt_version="v", model="m", timestamp=NOW,
                   results=[], pass_rate=rate, per_category_accuracy={},
                   avg_judge_score=None, avg_latency_ms=0, total_tokens=0)


def test_not_enough_runs_no_drift():
    drifting, avg = detect_drift([_run(0.5)] * 3, window=7, threshold=0.85)
    assert drifting is False and avg is None


def test_healthy_average_no_drift():
    drifting, avg = detect_drift([_run(0.9)] * 7, window=7, threshold=0.85)
    assert drifting is False and avg == pytest.approx(0.9)


def test_slow_drift_detected():
    # each run individually fine-ish, average below threshold
    runs = [_run(r) for r in [0.86, 0.85, 0.84, 0.83, 0.84, 0.82, 0.83]]
    drifting, avg = detect_drift(runs, window=7, threshold=0.85)
    assert drifting is True
    assert avg < 0.85


def test_only_window_newest_runs_counted():
    runs = [_run(0.9)] * 7 + [_run(0.1)] * 5  # old bad runs ignored (list is newest-first)
    drifting, avg = detect_drift(runs, window=7, threshold=0.85)
    assert drifting is False and avg == pytest.approx(0.9)
