"""Runtime settings loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    openrouter_api_key: str | None = None
    slack_webhook_url: str | None = None
    warning_threshold: float = 0.03
    critical_threshold: float = 0.08
    drift_window: int = 7
    drift_threshold: float = 0.85
    max_concurrency: int = 5
    db_path: Path = field(default_factory=lambda: Path("eval_runs.db"))

    @classmethod
    def load(cls, env_file: str | None = ".env") -> "Settings":
        if env_file:
            load_dotenv(env_file)
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY") or None,
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL") or None,
            warning_threshold=float(os.getenv("WARNING_THRESHOLD", "0.03")),
            critical_threshold=float(os.getenv("CRITICAL_THRESHOLD", "0.08")),
            drift_window=int(os.getenv("DRIFT_WINDOW", "7")),
            drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "0.85")),
            max_concurrency=int(os.getenv("MAX_CONCURRENCY", "5")),
            db_path=Path(os.getenv("DB_PATH", "eval_runs.db")),
        )
