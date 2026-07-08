from regression_detector.config import Settings


def test_settings_defaults(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("WARNING_THRESHOLD", raising=False)
    s = Settings.load(env_file=None)
    assert s.warning_threshold == 0.03
    assert s.critical_threshold == 0.08
    assert s.drift_window == 7
    assert s.max_concurrency == 5


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("WARNING_THRESHOLD", "0.05")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    s = Settings.load(env_file=None)
    assert s.warning_threshold == 0.05
    assert s.openrouter_api_key == "sk-test"
