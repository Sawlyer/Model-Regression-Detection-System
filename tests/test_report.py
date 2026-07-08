from datetime import datetime, timezone
from pathlib import Path

from regression_detector.models import (
    CaseFlip,
    CaseResult,
    ClassifierOutput,
    DiffReport,
    EvalRun,
    RunStatus,
)
from regression_detector.report import generate_report

NOW = datetime.now(timezone.utc)


def _run(run_id: str, rate: float) -> EvalRun:
    return EvalRun(run_id=run_id, prompt_version="v1", model="test-model", timestamp=NOW,
                   results=[CaseResult(case_id="c1", passed=True, latency_ms=100,
                                       tokens_in=10, tokens_out=5, category_match=True)],
                   pass_rate=rate, per_category_accuracy={"billing": rate},
                   avg_judge_score=4.2, avg_latency_ms=100.0, total_tokens=15)


def _diff() -> DiffReport:
    return DiffReport(
        baseline_run_id="a", current_run_id="b", pass_rate_delta=-0.1,
        per_category_delta={"billing": -0.1},
        regressions=[CaseFlip(
            case_id="c1", input="I want a refund", expected_category="billing",
            baseline_output=ClassifierOutput(category="billing", summary="old ok"),
            current_output=ClassifierOutput(category="general", summary="new bad"),
        )],
        improvements=[], status=RunStatus.CRITICAL,
    )


def test_report_contains_key_sections(tmp_path: Path):
    out = generate_report(_run("b", 0.8), _diff(), [_run("b", 0.8), _run("a", 0.9)],
                          (False, None), tmp_path / "r.html")
    html = out.read_text()
    assert "test-model" in html
    assert "critical" in html.lower()
    assert "I want a refund" in html          # regression input shown
    # side-by-side categories: was 'billing' (correct), now 'general' (wrong)
    assert "billing" in html and "general" in html
    assert "<svg" in html                      # trend chart


def test_first_run_report_without_diff(tmp_path: Path):
    out = generate_report(_run("a", 0.9), None, [_run("a", 0.9)], (False, None),
                          tmp_path / "r.html")
    html = out.read_text()
    assert "baseline" in html.lower()


def test_report_shows_drift_warning(tmp_path: Path):
    out = generate_report(_run("b", 0.8), None, [_run("b", 0.8)], (True, 0.82),
                          tmp_path / "r.html")
    assert "drift" in out.read_text().lower()
