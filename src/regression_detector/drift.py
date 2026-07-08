"""Slow-drift detection: rolling average of pass rates across recent runs."""
from __future__ import annotations

from .models import EvalRun


def detect_drift(
    runs: list[EvalRun], window: int, threshold: float
) -> tuple[bool, float | None]:
    """Detect slow decay via a rolling average of pass rate over recent runs.

    `runs` must be newest-first (as returned by RunStore.get_last_n_runs).
    Returns ``(drifting, moving_average)``. If there are fewer than `window`
    runs there isn't enough history, so it returns ``(False, None)``; otherwise
    it averages the newest `window` runs and flags drift when that average
    falls below `threshold`.
    """
    if len(runs) < window:
        return False, None
    recent = runs[:window]
    avg = sum(r.pass_rate for r in recent) / window
    return avg < threshold, avg
