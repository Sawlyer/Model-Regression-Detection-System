"""LLM providers: OpenRouter (real) and Mock (deterministic, offline)."""
from __future__ import annotations

import json
import re
from typing import Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMResponse(BaseModel):
    text: str
    tokens_in: int
    tokens_out: int


class LLMProvider(Protocol):
    async def complete(self, system: str, user: str, model: str) -> LLMResponse: ...


class OpenRouterProvider:
    def __init__(self, api_key: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    async def complete(self, system: str, user: str, model: str) -> LLMResponse:
        resp = await self._client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
        )


# Keyword routing, first match wins.
_KEYWORDS: list[tuple[str, str]] = [
    ("billing", r"refund|invoice|charge|bill|payment|subscription"),
    ("technical", r"error|crash|bug|broken|500|load|website|app"),
    ("account", r"password|login|locked|account|delete|email address"),
]


class MockProvider:
    """Deterministic offline provider for tests and keyless CI runs."""

    def __init__(self, fail_case_markers: set[str] | None = None) -> None:
        self._fail_markers = fail_case_markers or set()

    async def complete(self, system: str, user: str, model: str) -> LLMResponse:
        for marker in self._fail_markers:
            if marker in user:
                raise RuntimeError("mock provider failure")

        if "evaluator" in system.lower():
            return LLMResponse(text="4", tokens_in=len(user) // 4 or 1, tokens_out=1)

        # A prompt that tells the classifier to fall back to "general" when
        # unsure is a genuinely lazy instruction: the mock mirrors what a real
        # model does with it — over-predict "general" and lose the signal.
        # This is how the shipped degraded prompt (prompts/v2.yaml) regresses.
        if "prefer general" in system.lower():
            category = "general"
        else:
            # Only match keywords inside the email body, not the JSON-format
            # instructions (which mention every category name).
            email = user
            if "Email:" in email:
                email = email.split("Email:", 1)[1]
            email = email.split("Respond with", 1)[0]
            category = "general"
            lowered = email.lower()
            for cat, pattern in _KEYWORDS:
                if re.search(pattern, lowered):
                    category = cat
                    break

        payload = {
            "category": category,
            "summary": f"Customer writes about a {category} issue.",
        }
        return LLMResponse(
            text=json.dumps(payload),
            tokens_in=len(user) // 4 or 1,
            tokens_out=20,
        )
