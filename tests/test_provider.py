import json

import pytest

from regression_detector.provider import MockProvider


async def test_mock_classifies_billing():
    p = MockProvider()
    r = await p.complete(system="classify", user="I want a refund for my invoice", model="m")
    data = json.loads(r.text)
    assert data["category"] == "billing"
    assert r.tokens_in > 0 and r.tokens_out > 0


async def test_mock_keyword_priority_and_general_fallback():
    p = MockProvider()
    r = await p.complete(system="classify", user="Just wanted to say thanks!", model="m")
    assert json.loads(r.text)["category"] == "general"


async def test_mock_degraded_prompt_forces_general():
    p = MockProvider()
    r = await p.complete(system="classify DEGRADED", user="I want a refund", model="m")
    assert json.loads(r.text)["category"] == "general"


async def test_mock_judge_returns_score():
    p = MockProvider()
    r = await p.complete(system="You are an evaluator rating summaries", user="rate this", model="m")
    assert r.text.strip() == "4"


async def test_mock_injected_failure():
    p = MockProvider(fail_case_markers={"BOOM"})
    with pytest.raises(RuntimeError):
        await p.complete(system="classify", user="BOOM refund", model="m")
