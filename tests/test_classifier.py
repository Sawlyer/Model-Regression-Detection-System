import pytest

from regression_detector.classifier import ClassificationError, classify_email
from regression_detector.models import Category, PromptConfig
from regression_detector.provider import LLMResponse

CFG = PromptConfig(
    version="v1", timestamp="2026-07-08T10:00:00Z",
    model="m", system_prompt="You are a support email classifier.",
)


class ScriptedProvider:
    """Returns queued responses in order."""

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.calls: list[str] = []

    async def complete(self, system, user, model):
        self.calls.append(user)
        return LLMResponse(text=self._texts.pop(0), tokens_in=10, tokens_out=5)


async def test_classify_parses_clean_json():
    p = ScriptedProvider(['{"category": "billing", "summary": "Refund request."}'])
    out, tin, tout = await classify_email(p, CFG, "I want a refund")
    assert out.category is Category.BILLING
    assert tin == 10 and tout == 5


async def test_classify_strips_code_fences():
    p = ScriptedProvider(['```json\n{"category": "technical", "summary": "App crash."}\n```'])
    out, _, _ = await classify_email(p, CFG, "app crashes")
    assert out.category is Category.TECHNICAL


async def test_classify_repairs_once_then_succeeds():
    p = ScriptedProvider(["not json at all", '{"category": "account", "summary": "Login issue."}'])
    out, tin, tout = await classify_email(p, CFG, "cannot login")
    assert out.category is Category.ACCOUNT
    assert len(p.calls) == 2
    assert tin == 20 and tout == 10  # both attempts counted


async def test_classify_raises_after_failed_repair():
    p = ScriptedProvider(["garbage", "still garbage"])
    with pytest.raises(ClassificationError):
        await classify_email(p, CFG, "hello")
