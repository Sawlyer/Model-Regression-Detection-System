from datetime import datetime, timezone

import pytest

from regression_detector.config import Settings
from regression_detector.diff import compare
from regression_detector.models import (
    CaseResult,
    ClassifierOutput,
    EvalRun,
    GoldenDataset,
    RunStatus,
    TestCase,
)

SETTINGS = Settings()
NOW = datetime.now(timezone.utc)

DS = GoldenDataset(version="1", cases=[
    TestCase(id=f"c{i}", input=f"email {i}", expected_category="billing",
             ideal_summary="s", expected_difficulty="easy")
    for i in range(10)
])

DS30 = GoldenDataset(version="1", cases=[
    TestCase(id=f"c{i}", input="e", expected_category="billing",
             ideal_summary="s", expected_difficulty="easy")
    for i in range(30)
])


def _run(run_id: str, passes: list[bool]) -> EvalRun:
    results = [
        CaseResult(case_id=f"c{i}", output=ClassifierOutput(category="billing", summary="s"),
                   category_match=p, passed=p, latency_ms=1, tokens_in=1, tokens_out=1)
        for i, p in enumerate(passes)
    ]
    rate = sum(passes) / len(passes)
    return EvalRun(run_id=run_id, prompt_version="v", model="m", timestamp=NOW,
                   results=results, pass_rate=rate,
                   per_category_accuracy={"billing": rate},
                   avg_judge_score=4.0, avg_latency_ms=1.0, total_tokens=20)


def _run30(run_id: str, n_pass: int, total: int = 30) -> EvalRun:
    return _run(run_id, [True] * n_pass + [False] * (total - n_pass))


def test_no_change_is_ok():
    d = compare(_run("a", [True] * 10), _run("b", [True] * 10), DS, SETTINGS)
    assert d.status is RunStatus.OK
    assert d.regressions == [] and d.improvements == []


def test_small_drop_is_warning():
    base = _run30("a", 30)
    curr = _run30("b", 29)  # -3.33% -> warning
    d = compare(base, curr, DS30, SETTINGS)
    assert d.status is RunStatus.WARNING
    assert len(d.regressions) == 1


def test_big_drop_is_critical_with_flip_details():
    d = compare(_run("a", [True] * 10), _run("b", [True] * 8 + [False] * 2), DS, SETTINGS)
    assert d.status is RunStatus.CRITICAL  # -20%
    assert {f.case_id for f in d.regressions} == {"c8", "c9"}
    assert d.regressions[0].baseline_output is not None
    assert d.pass_rate_delta == pytest.approx(-0.2)


def test_improvements_tracked_and_gain_is_ok():
    d = compare(_run("a", [False] * 2 + [True] * 8), _run("b", [True] * 10), DS, SETTINGS)
    assert d.status is RunStatus.OK
    assert len(d.improvements) == 2
