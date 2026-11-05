"""Durable, transactional storage for workflow runs."""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))
DB_PATH = RUNS_DIR / "workflow.sqlite3"
_lock = threading.RLock()


def _connect() -> sqlite3.Connection:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS runs ("
        "run_id TEXT PRIMARY KEY, status TEXT NOT NULL, created_at TEXT, "
        "updated_at TEXT, payload TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS action_claims ("
        "action_key TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
        "status TEXT NOT NULL, result TEXT)"
    )
    return conn


@contextmanager
def _database():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def _serialize(run: Run) -> str:
    return json.dumps(asdict(run), ensure_ascii=False, separators=(",", ":"))


def _deserialize(payload: str) -> Run:
    data = json.loads(payload)
    data["steps"] = [Step(**step) for step in data.get("steps", [])]
    return Run(**data)


def save(run: Run) -> None:
    """Atomically store a complete run snapshot."""
    with _lock, _database() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, status, created_at, updated_at, payload) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
            "updated_at=excluded.updated_at, payload=excluded.payload",
            (run.run_id, run.status, run.created_at, run.updated_at, _serialize(run)),
        )


def load(run_id: str) -> Run | None:
    """Load one run, or return None if it does not exist."""
    with _lock, _database() as conn:
        row = conn.execute(
            "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _deserialize(row["payload"]) if row else None


def list_runs(status: str | None = None) -> list[Run]:
    """Return saved runs newest first, optionally filtered by status."""
    with _lock, _database() as conn:
        if status:
            rows = conn.execute(
                "SELECT payload FROM runs WHERE status = ? "
                "ORDER BY updated_at DESC, run_id DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT payload FROM runs ORDER BY updated_at DESC, run_id DESC"
            ).fetchall()
    return [_deserialize(row["payload"]) for row in rows]


def claim_decision(run_id: str, approve: bool, comment: str) -> tuple[Run | None, bool]:
    """Atomically consume the single operator decision for a waiting run."""
    with _lock, _database() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None, False
            run = _deserialize(row["payload"])
            if run.status != "awaiting_approval":
                conn.execute("COMMIT")
                return run, False

            run.status = "executing" if approve else "rejected"
            run.decision = {"approve": approve, "comment": comment}
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ?, payload = ? "
                "WHERE run_id = ? AND status = 'awaiting_approval'",
                (run.status, run.updated_at, _serialize(run), run_id),
            )
            conn.execute("COMMIT")
            return run, True
        except Exception:
            conn.execute("ROLLBACK")
            raise


def reserve_action(action_key: str, run_id: str) -> str | None:
    """Reserve a side effect once; return the run already holding this action."""
    with _lock, _database() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT run_id, status FROM action_claims WHERE action_key = ?",
                (action_key,),
            ).fetchone()
            if row and row["status"] in {"awaiting_approval", "executing", "uncertain", "done"}:
                conn.execute("COMMIT")
                return row["run_id"]
            conn.execute(
                "INSERT INTO action_claims (action_key, run_id, status, result) "
                "VALUES (?, ?, 'awaiting_approval', NULL) "
                "ON CONFLICT(action_key) DO UPDATE SET run_id=excluded.run_id, "
                "status='awaiting_approval', result=NULL",
                (action_key, run_id),
            )
            conn.execute("COMMIT")
            return None
        except Exception:
            conn.execute("ROLLBACK")
            raise


def finish_action(action_key: str, status: str, result: dict | None = None) -> None:
    """Persist the final outcome of a reserved side effect."""
    with _lock, _database() as conn:
        conn.execute(
            "UPDATE action_claims SET status = ?, result = ? WHERE action_key = ?",
            (status, json.dumps(result, ensure_ascii=False) if result is not None else None,
             action_key),
        )


def action_claim(action_key: str) -> dict | None:
    """Return the durable reservation and any recorded result."""
    with _lock, _database() as conn:
        row = conn.execute(
            "SELECT run_id, status, result FROM action_claims WHERE action_key = ?",
            (action_key,),
        ).fetchone()
    if not row:
        return None
    return {
        "run_id": row["run_id"],
        "status": row["status"],
        "result": json.loads(row["result"]) if row["result"] else None,
    }


def recover_interrupted() -> None:
    """Close runs left in an automatic state by a process restart."""
    with _lock, _database() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute(
                "SELECT payload FROM runs WHERE status IN ('running', 'executing')"
            ).fetchall()
            for row in rows:
                run = _deserialize(row["payload"])
                run.status = "stopped"
                run.stopped = "process_interrupted"
                run.reply = (
                    "Обробку перервано через перезапуск застосунку. "
                    "Щоб уникнути повторної дії, зверніться до оператора."
                )
                run.updated_at = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    "UPDATE runs SET status = ?, updated_at = ?, payload = ? "
                    "WHERE run_id = ?",
                    (run.status, run.updated_at, _serialize(run), run.run_id),
                )
                action_key = run.data.get("action_key")
                if action_key:
                    conn.execute(
                        "UPDATE action_claims SET status = 'uncertain' "
                        "WHERE action_key = ? AND status IN ('awaiting_approval', 'executing')",
                        (action_key,),
                    )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
