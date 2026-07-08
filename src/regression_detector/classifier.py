"""The LLM feature under test: email -> {category, summary}."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .models import ClassifierOutput, PromptConfig
from .provider import LLMProvider


class ClassificationError(Exception):
    """Raised when the model cannot produce valid structured output."""


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_REPAIR_SUFFIX = (
    "\n\nYour previous reply was not valid JSON. Reply with ONLY a JSON object: "
    '{"category": "<billing|technical|account|general>", "summary": "<one sentence>"}'
)


def build_user_message(email: str) -> str:
    return (
        f"Email:\n{email}\n\n"
        'Respond with ONLY a JSON object: {"category": '
        '"<billing|technical|account|general>", "summary": "<one sentence>"}'
    )


def _build_system(cfg: PromptConfig) -> str:
    parts = [cfg.system_prompt]
    for ex in cfg.few_shot:
        parts.append(
            f"Example:\nEmail: {ex.email}\n"
            f'Output: {{"category": "{ex.category.value}", "summary": "{ex.summary}"}}'
        )
    return "\n\n".join(parts)


def _parse(text: str) -> ClassifierOutput:
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return ClassifierOutput.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ClassificationError(f"invalid classifier output: {text!r}") from exc


async def classify_email(
    provider: LLMProvider, cfg: PromptConfig, email: str
) -> tuple[ClassifierOutput, int, int]:
    """Classify one email. One repair attempt on malformed output."""
    system = _build_system(cfg)
    user = build_user_message(email)

    resp = await provider.complete(system=system, user=user, model=cfg.model)
    tokens_in, tokens_out = resp.tokens_in, resp.tokens_out
    try:
        return _parse(resp.text), tokens_in, tokens_out
    except ClassificationError:
        pass  # one repair attempt below

    resp = await provider.complete(system=system, user=user + _REPAIR_SUFFIX, model=cfg.model)
    tokens_in += resp.tokens_in
    tokens_out += resp.tokens_out
    return _parse(resp.text), tokens_in, tokens_out
