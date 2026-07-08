"""SQLite persistence for eval runs (full run stored as JSON payload)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import EvalRun

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    pass_rate REAL NOT NULL,
    payload TEXT NOT NULL
);
"""


class RunStore:
    """SQLite-backed store of eval runs (each run kept as a full JSON payload)."""

    def __init__(self, db_path: Path) -> None:
        """Open (creating if needed) the SQLite database and ensure the schema exists."""
        self._db_path = db_path
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        """Open a connection to the database file."""
        return sqlite3.connect(self._db_path)

    def save_run(self, run: EvalRun) -> None:
        """Persist a run, replacing any existing row with the same run_id."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.prompt_version,
                    run.model,
                    run.timestamp.isoformat(),
                    run.pass_rate,
                    run.model_dump_json(),
                ),
            )

    def get_run(self, run_id: str) -> EvalRun | None:
        """Return the run with this id, or None if it doesn't exist."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return EvalRun.model_validate_json(row[0]) if row else None

    def get_latest_run(self) -> EvalRun | None:
        """Return the most recent run (by timestamp), or None if the store is empty."""
        runs = self.get_last_n_runs(1)
        return runs[0] if runs else None

    def get_last_n_runs(self, n: int) -> list[EvalRun]:
        """Return up to the `n` most recent runs, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM runs ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [EvalRun.model_validate_json(r[0]) for r in rows]

    def list_runs(self) -> list[tuple[str, str, str, float]]:
        """Return a lightweight (run_id, prompt_version, timestamp, pass_rate) row per run."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT run_id, prompt_version, timestamp, pass_rate "
                "FROM runs ORDER BY timestamp DESC"
            ).fetchall()
