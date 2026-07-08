"""Diff two eval runs: deltas, regressions, improvements, alert status."""
from __future__ import annotations

from .config import Settings
from .models import CaseFlip, DiffReport, EvalRun, GoldenDataset, RunStatus


def compare(
    baseline: EvalRun,
    current: EvalRun,
    dataset: GoldenDataset,
    settings: Settings,
) -> DiffReport:
    cases_by_id = {c.id: c for c in dataset.cases}
    base_by_id = {r.case_id: r for r in baseline.results}
    curr_by_id = {r.case_id: r for r in current.results}

    regressions: list[CaseFlip] = []
    improvements: list[CaseFlip] = []
    for case_id in base_by_id.keys() & curr_by_id.keys():
        b, c = base_by_id[case_id], curr_by_id[case_id]
        if b.passed == c.passed:
            continue
        case = cases_by_id.get(case_id)
        flip = CaseFlip(
            case_id=case_id,
            input=case.input if case else "",
            expected_category=case.expected_category if case else "general",
            baseline_output=b.output,
            current_output=c.output,
        )
        (regressions if b.passed else improvements).append(flip)

    regressions.sort(key=lambda f: f.case_id)
    improvements.sort(key=lambda f: f.case_id)

    delta = current.pass_rate - baseline.pass_rate
    if delta <= -settings.critical_threshold:
        status = RunStatus.CRITICAL
    elif delta <= -settings.warning_threshold:
        status = RunStatus.WARNING
    else:
        status = RunStatus.OK

    all_cats = baseline.per_category_accuracy.keys() | current.per_category_accuracy.keys()
    per_category_delta = {
        cat: current.per_category_accuracy.get(cat, 0.0)
        - baseline.per_category_accuracy.get(cat, 0.0)
        for cat in sorted(all_cats)
    }

    return DiffReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        pass_rate_delta=delta,
        per_category_delta=per_category_delta,
        regressions=regressions,
        improvements=improvements,
        status=status,
    )
