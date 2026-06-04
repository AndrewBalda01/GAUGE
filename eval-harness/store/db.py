"""SQLite persistence for EvalRun objects (async via aiosqlite)."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from store.schema import EvalRun, RegressionReport

_DEFAULT_DB = Path(__file__).parent.parent / "runs.db"

_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id          TEXT PRIMARY KEY,
    dataset     TEXT NOT NULL,
    version     TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    status      TEXT NOT NULL,
    git_ref     TEXT,
    branch      TEXT,
    created_at  TEXT NOT NULL,
    completed_at TEXT,
    notes       TEXT,
    payload     TEXT NOT NULL   -- full JSON blob
);

CREATE TABLE IF NOT EXISTS regression_reports (
    id              TEXT PRIMARY KEY,
    baseline_run_id TEXT NOT NULL,
    candidate_run_id TEXT NOT NULL,
    regression      INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    payload         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_branch ON eval_runs(branch, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON eval_runs(dataset, version, created_at);
"""


async def init_db(db_path: Path = _DEFAULT_DB) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(_DDL)
        await db.commit()


async def save_run(run: EvalRun, db_path: Path = _DEFAULT_DB) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO eval_runs
                (id, dataset, version, sha256, model_id, status,
                 git_ref, branch, created_at, completed_at, notes, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.dataset_name,
                run.dataset_version,
                run.dataset_sha256,
                run.config.model_id,
                run.status.value,
                run.git_ref,
                run.branch,
                run.created_at.isoformat(),
                run.completed_at.isoformat() if run.completed_at else None,
                run.notes,
                run.model_dump_json(),
            ),
        )
        await db.commit()


async def load_run(run_id: str, db_path: Path = _DEFAULT_DB) -> EvalRun | None:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT payload FROM eval_runs WHERE id = ?", (run_id,)
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return EvalRun.model_validate_json(row[0])


async def latest_run_on_branch(
    branch: str,
    dataset: str,
    version: str,
    db_path: Path = _DEFAULT_DB,
) -> EvalRun | None:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            """
            SELECT payload FROM eval_runs
            WHERE branch = ? AND dataset = ? AND version = ? AND status = 'completed'
            ORDER BY created_at DESC LIMIT 1
            """,
            (branch, dataset, version),
        ) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return EvalRun.model_validate_json(row[0])


async def list_runs(
    dataset: str | None = None,
    branch: str | None = None,
    limit: int = 20,
    db_path: Path = _DEFAULT_DB,
) -> list[dict]:
    clauses, params = [], []
    if dataset:
        clauses.append("dataset = ?")
        params.append(dataset)
    if branch:
        clauses.append("branch = ?")
        params.append(branch)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            f"""
            SELECT id, dataset, version, model_id, status, branch, created_at, completed_at
            FROM eval_runs {where}
            ORDER BY created_at DESC LIMIT ?
            """,
            params,
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def save_regression(report: RegressionReport, db_path: Path = _DEFAULT_DB) -> None:
    import uuid

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO regression_reports
                (id, baseline_run_id, candidate_run_id, regression, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                report.baseline_run_id,
                report.candidate_run_id,
                int(report.regression_detected),
                report.created_at.isoformat(),
                report.model_dump_json(),
            ),
        )
        await db.commit()
