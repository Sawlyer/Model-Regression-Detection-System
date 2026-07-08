"""Slack webhook alerting with console fallback."""
from __future__ import annotations

import httpx

from .models import DiffReport, EvalRun, RunStatus

_EMOJI = {RunStatus.OK: "✅", RunStatus.WARNING: "⚠️", RunStatus.CRITICAL: "🚨"}


def build_alert_text(diff: DiffReport, current: EvalRun, report_path: str) -> str:
    """Render the plain-text alert used for the console and inside the Slack payload."""
    baseline_rate = current.pass_rate - diff.pass_rate_delta
    return (
        f"[{diff.status.value.upper()}] Eval run {current.run_id} "
        f"(prompt {current.prompt_version}, model {current.model})\n"
        f"{len(diff.regressions)} regression(s), {len(diff.improvements)} improvement(s). "
        f"Pass rate: {baseline_rate:.0%} -> {current.pass_rate:.0%} "
        f"({diff.pass_rate_delta:+.1%})\n"
        f"Full report: {report_path}"
    )


def build_slack_payload(diff: DiffReport, current: EvalRun, report_path: str) -> dict:
    """Build the Slack Block Kit payload: status header, headline numbers, report link."""
    emoji = _EMOJI[diff.status]
    baseline_rate = current.pass_rate - diff.pass_rate_delta
    return {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Model eval: {diff.status.value.upper()}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Prompt:* {current.prompt_version}"},
                    {"type": "mrkdwn", "text": f"*Model:* {current.model}"},
                    {"type": "mrkdwn",
                     "text": f"*Pass rate:* {baseline_rate:.0%} → {current.pass_rate:.0%}"},
                    {"type": "mrkdwn",
                     "text": f"*Regressions:* {len(diff.regressions)}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Full report: `{report_path}`"},
            },
        ]
    }


def send_alert(payload: dict, text: str, webhook_url: str | None) -> bool:
    """POST to Slack when configured; otherwise print to console. Never raises."""
    if not webhook_url:
        print(text)
        return False
    try:
        resp = httpx.post(webhook_url, json=payload, timeout=10)
        return 200 <= resp.status_code < 300
    except httpx.HTTPError:
        print(text)
        return False
