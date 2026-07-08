from datetime import datetime, timezone

from regression_detector.models import (
    CaseResult,
    ClassifierOutput,
    GoldenDataset,
    PromptConfig,
    TestCase,
)
from regression_detector.scoring import aggregate

CFG = PromptConfig(version="v1", timestamp="2026-07-08T10:00:00Z", model="m", system_prompt="s")


def _case(i: str, cat: str) -> TestCase:
    return TestCase(id=i, input="x", expected_category=cat,
                    ideal_summary="s", expected_difficulty="easy")


def _result(i: str, passed: bool, judge: int | None = 4) -> CaseResult:
    return CaseResult(case_id=i, output=ClassifierOutput(category="billing", summary="s"),
                      category_match=passed, judge_score=judge,
                      latency_ms=100.0, tokens_in=10, tokens_out=5, passed=passed)


def test_aggregate_computes_rates():
    ds = GoldenDataset(version="1", cases=[
        _case("a", "billing"), _case("b", "billing"), _case("c", "technical"),
    ])
    results = [_result("a", True), _result("b", False), _result("c", True)]
    run = aggregate("run-1", CFG, datetime.now(timezone.utc), results, ds)
    assert run.pass_rate == 2 / 3
    assert run.per_category_accuracy["billing"] == 0.5
    assert run.per_category_accuracy["technical"] == 1.0
    assert run.avg_judge_score == 4.0
    assert run.total_tokens == 45
    assert run.avg_latency_ms == 100.0


def test_aggregate_handles_no_judge_scores():
    ds = GoldenDataset(version="1", cases=[_case("a", "general")])
    run = aggregate("run-1", CFG, datetime.now(timezone.utc), [_result("a", True, judge=None)], ds)
    assert run.avg_judge_score is None
