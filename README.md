# Model Regression Detection System

CI/CD pipeline for LLM behavior. Teams ship prompt and model changes blind; this system runs our LLM feature (a customer-support email classifier) against a human-verified golden dataset on every change, diffs the results against the previous run, and alerts before a regression reaches users. Prompts are versioned YAML files — they are the "code" this pipeline runs CI against.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add your OPENROUTER_API_KEY
regression-eval run prompts/v1.yaml
```

No `OPENROUTER_API_KEY` in the environment? The CLI falls back to a deterministic offline `MockProvider` — useful for demos, tests, and CI without secrets. Force it anytime with `--mock`.

Run the test suite (fully offline, no network calls):

```bash
pytest
```

## How it works

1. `runner` sends every golden case through the classifier (async, bounded by `MAX_CONCURRENCY`, 2 retries with backoff per case).
2. `scoring` computes per-case dimensions: exact category match, summary relevance via LLM-as-judge (1–5), latency, token usage.
3. `storage` persists the full run in SQLite (`eval_runs.db`).
4. `diff` compares against the previous run: pass-rate delta, per-category deltas, and the exact cases that flipped pass→fail (regressions) or fail→pass (improvements).
5. `report` renders a standalone HTML report (scorecard, side-by-side regressed outputs, pass-rate trend chart).
6. `alert` posts to Slack (or prints to console when no webhook is configured). Exit code `2` on critical — CI uses this to block merges.

Slow drift is tracked separately: a rolling average of pass rate over the last `DRIFT_WINDOW` runs fires a warning when it sinks below `DRIFT_THRESHOLD`, even if no single run tripped the per-run gates.

## Adding golden dataset cases

Edit `data/golden_dataset.json`. Each case:

```json
{
  "id": "tc-061",
  "input": "the raw customer email",
  "expected_category": "billing | technical | account | general",
  "ideal_summary": "one sentence, human-written",
  "expected_difficulty": "easy | medium | hard",
  "notes": "required for hard cases: why this case matters"
}
```

Rules we hold ourselves to:

- Cases are **human-written and human-verified**. Never generate them with an LLM — eval quality is bounded by data quality.
- New cases come primarily from **production failures**: when the classifier gets something wrong in the wild, that email (anonymized) becomes a test case.
- Keep IDs stable; bump the dataset `version` when the eval bar meaningfully changes.
- Deliberately include hard cases: ambiguous two-category emails, very short ones, typos, mixed languages, sarcasm.

## Adding a prompt version

```bash
cp prompts/v1.yaml prompts/v2.yaml
# edit: bump version, timestamp, change system_prompt / few_shot / model
```

Open a PR — any change under `prompts/` triggers the eval workflow.

## Thresholds

All configurable via env vars (see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `WARNING_THRESHOLD` | `0.03` | pass-rate drop > 3% → warning |
| `CRITICAL_THRESHOLD` | `0.08` | pass-rate drop > 8% → critical, blocks merge |
| `DRIFT_WINDOW` | `7` | runs in the rolling average |
| `DRIFT_THRESHOLD` | `0.85` | rolling average below this → slow-drift warning |
| `MAX_CONCURRENCY` | `5` | parallel LLM calls (lower it if the free tier rate-limits) |

## CLI reference

```bash
regression-eval run prompts/v2.yaml            # eval + diff + report + alert
regression-eval run prompts/v2.yaml --mock     # offline deterministic provider
regression-eval list-runs                      # run history
regression-eval compare run-A run-B            # diff two stored runs
regression-eval report run-A --out out.html    # regenerate a report
```

`run` writes `reports/<run_id>.html` and optionally a machine-readable `--summary-json` (used by CI).

## CI behavior

`.github/workflows/eval.yml` runs on PRs touching `prompts/`, the dataset, or `src/`:

1. unit tests
2. baseline eval on the oldest prompt, then eval of the newest prompt
3. posts a summary comment on the PR
4. uploads the HTML report as an artifact
5. fails the job (blocking merge) if the eval exited with code `2` (critical regression)

Add `OPENROUTER_API_KEY` as a repo secret for real-model evals; without it the workflow runs on the mock provider.

## Docker

```bash
docker build -t regression-detector .
docker run --rm -e OPENROUTER_API_KEY=sk-or-... regression-detector run prompts/v1.yaml
```

## Architecture decisions

- **Hand-written golden dataset** — the eval bar must be ground truth, not model output; LLM-generated test data inherits the model's blind spots.
- **Per-case regression tracking, not just aggregate scores** — "accuracy dropped 3%" is not actionable; "these 2 emails flipped from billing to general" is.
- **Slow drift separate from per-run diffs** — five consecutive -2% runs never trip an 8% gate but add up to -10%; the rolling average catches what per-run checks structurally cannot.
- **SQLite + JSON payloads** — zero infrastructure, portable, diffable; the full `EvalRun` is stored as JSON so the schema never blocks adding metrics.
- **MockProvider as a first-class provider** — deterministic keyword-based classification makes tests hermetic and lets CI run without secrets; a `DEGRADED` marker in a prompt simulates a regression end-to-end.
- **OpenRouter via the OpenAI SDK** — one `base_url` override gives access to many models (free tiers included); swapping models is a YAML edit, which is itself a change this pipeline evals.
