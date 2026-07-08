from regression_detector.judge import JUDGE_SYSTEM, judge_summary
from regression_detector.provider import LLMResponse, MockProvider


class ScriptedProvider:
    def __init__(self, text: str):
        self._text = text

    async def complete(self, system, user, model):
        return LLMResponse(text=self._text, tokens_in=8, tokens_out=1)


async def test_judge_system_prompt_contains_evaluator_marker():
    assert "evaluator" in JUDGE_SYSTEM.lower()


async def test_judge_parses_score():
    score, tin, tout = await judge_summary(ScriptedProvider("5"), "m", "email", "ideal", "actual")
    assert score == 5


async def test_judge_extracts_digit_from_verbose_reply():
    score, _, _ = await judge_summary(ScriptedProvider("I rate this 3 out of 5"), "m", "e", "i", "a")
    assert score == 3


async def test_judge_returns_none_on_unparseable():
    score, _, _ = await judge_summary(ScriptedProvider("excellent"), "m", "e", "i", "a")
    assert score is None


async def test_judge_works_with_mock_provider():
    score, _, _ = await judge_summary(MockProvider(), "m", "e", "i", "a")
    assert score == 4
