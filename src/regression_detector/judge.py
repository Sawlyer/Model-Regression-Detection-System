"""LLM-as-judge: rate summary relevance 1-5 against the ideal summary."""
from __future__ import annotations

import re

from .provider import LLMProvider

JUDGE_SYSTEM = (
    "You are an evaluator. Rate how well a candidate summary captures the key "
    "point of a customer email, compared to an ideal reference summary. "
    "Reply with ONLY a single integer from 1 (irrelevant) to 5 (equivalent to the ideal)."
)

_SCORE_RE = re.compile(r"[1-5]")


async def judge_summary(
    provider: LLMProvider,
    model: str,
    email: str,
    ideal_summary: str,
    actual_summary: str,
) -> tuple[int | None, int, int]:
    """Ask the model to rate a candidate summary 1–5 against the ideal one.

    Returns ``(score, tokens_in, tokens_out)`` where score is the first 1–5
    digit found in the reply, or None if none is present.
    """
    user = (
        f"Email:\n{email}\n\n"
        f"Ideal summary: {ideal_summary}\n"
        f"Candidate summary: {actual_summary}\n\n"
        "Score (1-5):"
    )
    resp = await provider.complete(system=JUDGE_SYSTEM, user=user, model=model)
    match = _SCORE_RE.search(resp.text)
    score = int(match.group()) if match else None
    return score, resp.tokens_in, resp.tokens_out
