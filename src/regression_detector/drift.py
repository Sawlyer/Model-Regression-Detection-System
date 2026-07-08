"""Slow-drift detection: rolling average of pass rates across recent runs."""
from __future__ import annotations

from .models import EvalRun


def detect_drift(
    runs: list[EvalRun], window: int, threshold: float
) -> tuple[bool, float | None]:
    """`runs` must be newest-first (as returned by RunStore.get_last_n_runs)."""
    if len(runs) < window:
        return False, None
    recent = runs[:window]
    avg = sum(r.pass_rate for r in recent) / window
    return avg < threshold, avg
