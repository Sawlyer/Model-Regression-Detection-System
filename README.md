# Model Regression Detection System

**CI/CD for LLM behavior.** Every time you change a prompt or swap a model, this system re-tests your LLM feature against a fixed set of human-verified examples, compares the new results to the last run, and alerts you *before* a quality drop ships to users.

Think of it as unit tests for a thing that has no fixed output. You can't `assert output == expected` on an LLM — so instead you measure quality across many cases, track it over time, and fail the build when it drops.

---

## The problem it solves

LLM features are invisible to normal CI. You tweak a prompt, tests still pass (there are none for prompt quality), you merge, and three days later support is drowning because the classifier started dumping everything into the wrong bucket. Nobody saw it happen.

This project makes prompt quality a **measurable, versioned, gated** thing:

- **Measurable** — every change is scored against a golden dataset.
- **Versioned** — prompts live as YAML files; changing one is a diff you can review.
- **Gated** — a big enough quality drop fails CI and blocks the merge.

---

## The feature under test

A **customer-support email classifier**: it reads an email and returns a category (`billing`, `technical`, `account`, `general`) plus a one-sentence summary. It's deliberately simple — the interesting part is the machinery *around* it that catches when it gets worse.

---

## How detection actually works

Five moving parts. Here's the whole loop.

### 1. Golden dataset — the ground truth
`data/golden_dataset.json` holds 60 hand-written emails, each labelled by a human with the correct category and an ideal summary. This is the answer key. Quality is measured entirely against it, so the dataset *is* the standard — LLM-generated test data would just bake in the model's own blind spots.

### 2. Run — score one prompt
`regression-eval run prompts/v1.yaml` sends all 60 emails through the classifier and scores each one on four dimensions:

| Dimension | How |
|---|---|
| Category correct? | exact match vs the label (pass/fail) |
| Summary good? | a second LLM call ("LLM-as-judge") rates relevance 1–5 |
| Latency | milliseconds per call |
| Cost | tokens used |

A case **passes** when its category matches. The run's **pass rate** = passed / total. Everything is saved to SQLite (`eval_runs.db`).

### 3. Diff — compare to the previous run
This is the core. The new run is compared against the last stored run:

- overall pass-rate change (e.g. 93% → 85%)
- per-category change (maybe only `billing` broke)
- **regressions** — the exact emails that passed before and fail now
- **improvements** — the ones that flipped the good way

"Accuracy dropped 8%" is not actionable. "These 5 billing emails now get classified as general — here's the old output next to the new output" is. The diff gives you the second thing.

### 4. Thresholds — decide if it matters
If 1 case out of 80 flips, that's noise. Configurable gates turn the delta into a verdict:

- pass-rate drop **> 3%** → `WARNING`
- pass-rate drop **> 8%** → `CRITICAL` → the command exits with code **2**, which fails CI and blocks the merge.

### 5. Alert + report
- A standalone **HTML report** (`reports/<run-id>.html`): scorecard, the regressed cases side-by-side (old output vs new), and a pass-rate trend chart.
- A **Slack message** with the headline numbers and a link. No webhook configured? It prints the same message to the console instead.

### Bonus: slow-drift detection
Per-run gates miss *gradual* decay — five runs each dropping 2% never trip an 8% gate but add up to −10%. So the system also tracks a rolling average of the pass rate over the last 7 runs and warns if that average sinks below a threshold, independent of any single run.

---

## Quick start

```bash
cd Model-Regression-Detection-System
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env      # then paste your OpenRouter key into it
```

Run your first eval:

```bash
regression-eval run prompts/v1.yaml
```

**No API key?** The system falls back to a built-in offline `MockProvider` (deterministic, free, no network) — great for trying the flow and for CI without secrets. Force it anytime with `--mock`:

```bash
regression-eval run prompts/v1.yaml --mock
```

Run the tests (fully offline, no API calls):

```bash
pytest          # 49 tests
```

---

## See it catch a regression (2-minute demo)

The repo ships two prompts: `prompts/v1.yaml` (good) and `prompts/v2.yaml` (**deliberately degraded** — vague instructions, no examples, a lazy "prefer general" fallback). Run them in order:

```bash
regression-eval run prompts/v1.yaml --mock     # baseline  -> 93% pass, exit 0
regression-eval run prompts/v2.yaml --mock     # degraded  -> 25% pass, CRITICAL, exit 2
regression-eval run prompts/v1.yaml --mock     # recovered -> 93% pass, exit 0
```

Then open the middle report to see the regressions laid out:

```bash
xdg-open reports/*.html     # macOS: open reports/*.html
```

> **Important:** the `CRITICAL` / `exit 2` on the v2 run is the system **working**, not breaking. It means "a big quality drop was detected — do not merge this." That exit code is exactly what fails the build in CI. A green run (exit 0) means no significant regression.

See the run history any time:

```bash
regression-eval list-runs
```

---

## Commands

```bash
regression-eval run <prompt.yaml> [options]   # eval + diff + report + alert
regression-eval list-runs                     # run history, newest first
regression-eval compare <run-A> <run-B>       # diff two stored runs
regression-eval report <run-id> --out x.html  # regenerate a report
```

`run` options:

| Option | Meaning |
|---|---|
| `--mock` | use the offline deterministic provider (no API, no cost) |
| `--dataset PATH` | use a different golden dataset (default `data/golden_dataset.json`) |
| `--db PATH` | SQLite location (default `eval_runs.db`) |
| `--report-dir DIR` | where HTML reports go (default `reports/`) |
| `--summary-json PATH` | write a machine-readable summary (used by CI) |

Free-tier models rate-limit hard; if you hit 429s, lower concurrency:

```bash
MAX_CONCURRENCY=1 regression-eval run prompts/v1.yaml
```

---

## Configuration (env vars, see `.env.example`)

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | your OpenRouter key; omit to run on the mock provider |
| `SLACK_WEBHOOK_URL` | — | Slack incoming webhook; omit to print alerts to console |
| `WARNING_THRESHOLD` | `0.03` | pass-rate drop above this → WARNING |
| `CRITICAL_THRESHOLD` | `0.08` | pass-rate drop above this → CRITICAL, blocks merge |
| `DRIFT_WINDOW` | `7` | number of runs in the rolling average |
| `DRIFT_THRESHOLD` | `0.85` | rolling average below this → slow-drift warning |
| `MAX_CONCURRENCY` | `5` | parallel LLM calls |

The LLM provider is **OpenRouter** (used through the OpenAI SDK with a `base_url` override), which exposes many models including free tiers. The model is set per-prompt in the YAML `model:` field — so switching models is itself a change this pipeline evaluates.

---

## Adding to the golden dataset

Edit `data/golden_dataset.json`:

```json
{
  "id": "tc-061",
  "input": "the raw customer email",
  "expected_category": "billing | technical | account | general",
  "ideal_summary": "one sentence, written by a human",
  "expected_difficulty": "easy | medium | hard",
  "notes": "for hard cases: why this case matters"
}
```

Rules: cases are **human-verified, never LLM-generated**; new cases come mostly from **real production failures** (an email the classifier got wrong becomes a permanent test); keep IDs stable; bump the dataset `version` when the bar changes. The set deliberately includes hard cases — ambiguous emails, very short ones, typos, mixed languages, sarcasm — because that's where regressions hide.

## Adding a prompt version

```bash
cp prompts/v1.yaml prompts/v3.yaml    # then edit: bump version + timestamp, change the prompt
regression-eval run prompts/v3.yaml   # diffs against the previous run automatically
```

Opening a PR that touches `prompts/` triggers the CI eval.

---

## CI/CD

`.github/workflows/eval.yml` runs on PRs that touch `prompts/`, the dataset, or `src/`:

1. runs the unit tests
2. evals the oldest prompt (baseline) then the newest prompt
3. posts a summary comment on the PR (status, pass rate, regression count)
4. uploads the HTML report as a build artifact
5. **fails the job — blocking the merge — if the eval exited `2` (critical regression)**

Add `OPENROUTER_API_KEY` (and optionally `SLACK_WEBHOOK_URL`) as repository secrets for real-model evals. Without the secret, CI runs on the mock provider so the pipeline still works end-to-end.

## Docker

```bash
docker build -t regression-detector .
docker run --rm -e OPENROUTER_API_KEY=sk-or-... regression-detector run prompts/v1.yaml
docker run --rm regression-detector run prompts/v1.yaml --mock    # no key needed
```

---

## Design decisions (and why)

- **Hand-written golden dataset** — evaluation quality is bounded by data quality; LLM-generated cases inherit the model's blind spots.
- **Per-case regression tracking, not just an aggregate score** — a number isn't actionable; the specific emails that flipped, with old-vs-new output, are.
- **Slow drift tracked separately from per-run diffs** — gradual decay never trips a single-run gate but the rolling average catches it.
- **SQLite + JSON payloads** — zero infrastructure, portable, git-friendly; the full run is stored as JSON so adding a new metric never requires a schema migration.
- **MockProvider as a first-class provider** — makes tests hermetic (no network) and lets CI run without secrets; the shipped `v2` prompt regresses under it deterministically, so the whole detect-and-block loop is demonstrable offline.

## Project layout

```
src/regression_detector/
  config.py       settings from env / .env
  models.py       typed domain models (Pydantic)
  provider.py     OpenRouter client + offline MockProvider
  classifier.py   the feature under test (email -> category + summary)
  judge.py        LLM-as-judge summary scoring
  dataset.py      load + validate the golden dataset
  runner.py       async batched eval runner (retries, concurrency limit)
  scoring.py      aggregate per-case results into a run
  diff.py         compare two runs -> deltas, regressions, status
  drift.py        rolling-average slow-drift detection
  storage.py      SQLite persistence
  report.py       standalone HTML report + trend chart
  alert.py        Slack webhook / console fallback
  cli.py          the `regression-eval` command
prompts/          versioned prompt YAML (the "code" under test)
data/             the golden dataset
tests/            49 offline tests
```
