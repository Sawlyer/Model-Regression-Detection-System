"""Aggregate per-case results into an EvalRun with multi-dimensional metrics."""
from __future__ import annotations

from datetime import datetime

from .models import CaseResult, EvalRun, GoldenDataset, PromptConfig


def aggregate(
    run_id: str,
    cfg: PromptConfig,
    timestamp: datetime,
    results: list[CaseResult],
    dataset: GoldenDataset,
) -> EvalRun:
    total = len(results)
    passed = sum(r.passed for r in results)
    expected_by_id = {c.id: c.expected_category for c in dataset.cases}

    per_cat_total: dict[str, int] = {}
    per_cat_pass: dict[str, int] = {}
    for r in results:
        cat = expected_by_id[r.case_id].value
        per_cat_total[cat] = per_cat_total.get(cat, 0) + 1
        per_cat_pass[cat] = per_cat_pass.get(cat, 0) + int(r.passed)

    judge_scores = [r.judge_score for r in results if r.judge_score is not None]

    return EvalRun(
        run_id=run_id,
        prompt_version=cfg.version,
        model=cfg.model,
        timestamp=timestamp,
        results=results,
        pass_rate=passed / total if total else 0.0,
        per_category_accuracy={
            cat: per_cat_pass[cat] / n for cat, n in per_cat_total.items()
        },
        avg_judge_score=sum(judge_scores) / len(judge_scores) if judge_scores else None,
        avg_latency_ms=sum(r.latency_ms for r in results) / total if total else 0.0,
        total_tokens=sum(r.tokens_in + r.tokens_out for r in results),
    )
