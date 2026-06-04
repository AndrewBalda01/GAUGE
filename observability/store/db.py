"""SQLite persistence (async via aiosqlite)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from store.schema import Alert, LLMTrace

_DEFAULT_DB = Path(__file__).parent.parent / "traces.db"

_DDL = """
CREATE TABLE IF NOT EXISTS llm_traces (
    trace_id    TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    system      TEXT,
    operation   TEXT,
    app         TEXT,
    route       TEXT,
    run_id      TEXT,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    cost_usd    REAL,
    latency_ms  REAL,
    ttft_ms     REAL,
    status      TEXT,
    error       TEXT,
    prompt      TEXT,
    completion  TEXT,
    timestamp   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    message     TEXT NOT NULL,
    severity    TEXT,
    metric_value REAL,
    threshold   REAL,
    window_start TEXT,
    window_end   TEXT,
    app         TEXT,
    fired_at    TEXT NOT NULL,
    acknowledged INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_traces_ts    ON llm_traces(timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_app   ON llm_traces(app, timestamp);
CREATE INDEX IF NOT EXISTS idx_traces_model ON llm_traces(model, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_fired ON alerts(fired_at);
"""


async def init_db(db_path: Path = _DEFAULT_DB) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_DDL)
        await db.commit()


async def save_trace(trace: LLMTrace, db_path: Path = _DEFAULT_DB) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO llm_traces
               (trace_id, model, system, operation, app, route, run_id,
                tokens_in, tokens_out, cost_usd, latency_ms, ttft_ms,
                status, error, prompt, completion, timestamp)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                trace.trace_id, trace.model, trace.system, trace.operation,
                trace.app, trace.route, trace.run_id,
                trace.tokens_in, trace.tokens_out, trace.cost_usd,
                trace.latency_ms, trace.ttft_ms,
                trace.status, trace.error, trace.prompt, trace.completion,
                trace.timestamp.isoformat(),
            ),
        )
        await db.commit()


async def save_alert(alert: Alert, db_path: Path = _DEFAULT_DB) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO alerts
               (id, kind, message, severity, metric_value, threshold,
                window_start, window_end, app, fired_at, acknowledged)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                alert.id, alert.kind, alert.message, alert.severity,
                alert.metric_value, alert.threshold,
                alert.window_start.isoformat() if alert.window_start else None,
                alert.window_end.isoformat() if alert.window_end else None,
                alert.app, alert.fired_at.isoformat(), int(alert.acknowledged),
            ),
        )
        await db.commit()


async def load_trace(trace_id: str, db_path: Path = _DEFAULT_DB) -> LLMTrace | None:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM llm_traces WHERE trace_id = ?", (trace_id,)
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return _row_to_trace(dict(row))


async def query_traces(
    app: str | None = None,
    model: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    status: str | None = None,
    limit: int = 500,
    db_path: Path = _DEFAULT_DB,
) -> list[LLMTrace]:
    clauses, params = [], []
    if app:
        clauses.append("app = ?"); params.append(app)
    if model:
        clauses.append("model = ?"); params.append(model)
    if since:
        clauses.append("timestamp >= ?"); params.append(since.isoformat())
    if until:
        clauses.append("timestamp <= ?"); params.append(until.isoformat())
    if status:
        clauses.append("status = ?"); params.append(status)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM llm_traces {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_trace(dict(r)) for r in rows]


async def query_alerts(
    acknowledged: bool | None = None,
    limit: int = 50,
    db_path: Path = _DEFAULT_DB,
) -> list[Alert]:
    clauses, params = [], []
    if acknowledged is not None:
        clauses.append("acknowledged = ?"); params.append(int(acknowledged))
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"SELECT * FROM alerts {where} ORDER BY fired_at DESC LIMIT ?",
            params,
        ) as cur:
            rows = await cur.fetchall()
    return [_row_to_alert(dict(r)) for r in rows]


def _row_to_trace(row: dict) -> LLMTrace:
    row["timestamp"] = datetime.fromisoformat(row["timestamp"])
    return LLMTrace(**row)


def _row_to_alert(row: dict) -> Alert:
    row["fired_at"] = datetime.fromisoformat(row["fired_at"])
    if row.get("window_start"):
        row["window_start"] = datetime.fromisoformat(row["window_start"])
    if row.get("window_end"):
        row["window_end"] = datetime.fromisoformat(row["window_end"])
    row["acknowledged"] = bool(row["acknowledged"])
    return Alert(**row)
