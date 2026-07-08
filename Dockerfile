FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
COPY prompts/ prompts/
COPY data/ data/

RUN pip install --no-cache-dir .

# Config via env vars:
#   OPENROUTER_API_KEY  - LLM provider key (omit to run with the mock provider)
#   SLACK_WEBHOOK_URL   - optional Slack alerting
#   WARNING_THRESHOLD / CRITICAL_THRESHOLD / DRIFT_THRESHOLD - eval gates
ENTRYPOINT ["regression-eval"]
CMD ["run", "prompts/v1.yaml"]
