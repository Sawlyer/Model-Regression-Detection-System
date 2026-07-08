"""Async batched eval runner: every golden case through the classifier + judge."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from .classifier import classify_email
from .config import Settings
from .judge import judge_summary
from .models import CaseResult, EvalRun, GoldenDataset, PromptConfig, TestCase
from .provider import LLMProvider
from .scoring import aggregate

_MAX_ATTEMPTS = 3  # 1 try + 2 retries
_BACKOFF_SECONDS = [1.0, 2.0]


async def _run_case(
    provider: LLMProvider, cfg: PromptConfig, case: TestCase, sem: asyncio.Semaphore
) -> CaseResult:
    async with sem:
        start = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                output, tin, tout = await classify_email(provider, cfg, case.input)
                latency_ms = (time.perf_counter() - start) * 1000
                score, jin, jout = await judge_summary(
                    provider, cfg.model, case.input, case.ideal_summary, output.summary
                )
                match = output.category is case.expected_category
                return CaseResult(
                    case_id=case.id,
                    output=output,
                    category_match=match,
                    judge_score=score,
                    latency_ms=latency_ms,
                    tokens_in=tin + jin,
                    tokens_out=tout + jout,
                    passed=match,
                )
            except Exception as exc:  # noqa: BLE001 — a single case must never crash the run
                last_error = exc
                if attempt < _MAX_ATTEMPTS - 1:
                    await asyncio.sleep(_BACKOFF_SECONDS[attempt])
        latency_ms = (time.perf_counter() - start) * 1000
        return CaseResult(
            case_id=case.id,
            error=f"{type(last_error).__name__}: {last_error}",
            latency_ms=latency_ms,
            passed=False,
        )


async def run_eval(
    provider: LLMProvider,
    cfg: PromptConfig,
    dataset: GoldenDataset,
    settings: Settings,
    run_id: str | None = None,
) -> EvalRun:
    now = datetime.now(timezone.utc)
    run_id = run_id or f"run-{now:%Y%m%d-%H%M%S}"
    sem = asyncio.Semaphore(settings.max_concurrency)
    results = await asyncio.gather(
        *(_run_case(provider, cfg, case, sem) for case in dataset.cases)
    )
    return aggregate(run_id, cfg, now, list(results), dataset)
