from datetime import datetime, timezone

from regression_detector.alert import build_alert_text, build_slack_payload, send_alert
from regression_detector.models import DiffReport, EvalRun, RunStatus

NOW = datetime.now(timezone.utc)

RUN = EvalRun(run_id="b", prompt_version="v2", model="m", timestamp=NOW, results=[],
              pass_rate=0.89, per_category_accuracy={}, avg_judge_score=4.0,
              avg_latency_ms=100, total_tokens=10)

DIFF = DiffReport(baseline_run_id="a", current_run_id="b", pass_rate_delta=-0.05,
                  per_category_delta={}, regressions=[], improvements=[],
                  status=RunStatus.WARNING)


def test_alert_text_has_headline_numbers():
    text = build_alert_text(DIFF, RUN, "reports/r.html")
    assert "WARNING" in text
    assert "89" in text  # current pass rate percent
    assert "reports/r.html" in text


def test_slack_payload_structure():
    payload = build_slack_payload(DIFF, RUN, "reports/r.html")
    assert "blocks" in payload
    header = payload["blocks"][0]
    assert header["type"] == "header"
    assert "⚠️" in header["text"]["text"]


def test_send_alert_console_fallback(capsys):
    ok = send_alert({"blocks": []}, "ALERT TEXT", webhook_url=None)
    assert ok is False
    assert "ALERT TEXT" in capsys.readouterr().out


def test_send_alert_posts_to_webhook(monkeypatch):
    calls = {}

    def fake_post(url, json, timeout):
        calls["url"] = url

        class R:
            status_code = 200

        return R()

    import regression_detector.alert as alert_mod
    monkeypatch.setattr(alert_mod.httpx, "post", fake_post)
    ok = send_alert({"blocks": []}, "t", webhook_url="https://hooks.slack.com/x")
    assert ok is True and calls["url"].startswith("https://hooks.slack.com")
