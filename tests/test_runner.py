from regression_detector.config import Settings
from regression_detector.models import GoldenDataset, PromptConfig, TestCase
from regression_detector.provider import MockProvider
from regression_detector.runner import run_eval

CFG = PromptConfig(version="v1", timestamp="2026-07-08T10:00:00Z", model="m",
                   system_prompt="classify")
SETTINGS = Settings(max_concurrency=3)


def _ds(cases) -> GoldenDataset:
    return GoldenDataset(version="1", cases=cases)


async def test_run_eval_full_pass_on_keyword_cases():
    ds = _ds([
        TestCase(id="a", input="I want a refund now", expected_category="billing",
                 ideal_summary="s", expected_difficulty="easy"),
        TestCase(id="b", input="the app crashes with an error", expected_category="technical",
                 ideal_summary="s", expected_difficulty="easy"),
    ])
    run = await run_eval(MockProvider(), CFG, ds, SETTINGS, run_id="r1")
    assert run.run_id == "r1"
    assert run.pass_rate == 1.0
    assert all(r.judge_score == 4 for r in run.results)
    assert all(r.latency_ms >= 0 for r in run.results)


async def test_run_eval_provider_failure_becomes_error_case(monkeypatch):
    import regression_detector.runner as runner_mod

    async def no_sleep(_):  # skip real backoff sleeps
        pass

    monkeypatch.setattr(runner_mod.asyncio, "sleep", no_sleep)

    ds = _ds([
        TestCase(id="a", input="BOOM refund", expected_category="billing",
                 ideal_summary="s", expected_difficulty="easy"),
        TestCase(id="b", input="password reset fails at login", expected_category="account",
                 ideal_summary="s", expected_difficulty="easy"),
    ])
    run = await run_eval(MockProvider(fail_case_markers={"BOOM"}), CFG, ds, SETTINGS, run_id="r2")
    failed = next(r for r in run.results if r.case_id == "a")
    ok = next(r for r in run.results if r.case_id == "b")
    assert failed.error and not failed.passed and failed.output is None
    assert ok.passed
    assert run.pass_rate == 0.5


async def test_run_eval_degraded_prompt_fails_non_general():
    bad_cfg = CFG.model_copy(update={
        "system_prompt": "Classify emails. When unsure, prefer general.",
        "version": "v2",
    })
    ds = _ds([
        TestCase(id="a", input="refund my payment", expected_category="billing",
                 ideal_summary="s", expected_difficulty="easy"),
        TestCase(id="b", input="hello, love your product", expected_category="general",
                 ideal_summary="s", expected_difficulty="easy"),
    ])
    run = await run_eval(MockProvider(), bad_cfg, ds, SETTINGS, run_id="r3")
    assert run.pass_rate == 0.5  # billing case regressed to general
