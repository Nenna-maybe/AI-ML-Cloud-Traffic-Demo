"""
database.py - SQLite data layer.

Rules followed here:
* Every query uses "?" placeholders (parameterized) - never string formatting
  with user input - which prevents SQL injection.
* Writes that touch more than one table run inside a single transaction.
* Images and videos are stored as files on disk (or S3); the database keeps
  only metadata and the file path.
* No personal information is stored.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS detection_sessions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at           TEXT    NOT NULL,
    source_type          TEXT    NOT NULL CHECK (source_type IN ('image', 'video', 'sample')),
    file_name            TEXT,
    stored_path          TEXT,
    annotated_path       TEXT,
    storage_backend      TEXT    NOT NULL DEFAULT 'local',
    location             TEXT,
    model_name           TEXT,
    confidence_threshold REAL,
    total_vehicles       INTEGER NOT NULL DEFAULT 0,
    frames_processed     INTEGER NOT NULL DEFAULT 1,
    avg_vehicles         REAL,
    peak_vehicles        INTEGER,
    congestion_score     REAL,
    congestion_category  TEXT,
    is_sample            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vehicle_counts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES detection_sessions(id) ON DELETE CASCADE,
    vehicle_class TEXT    NOT NULL,
    count         REAL    NOT NULL CHECK (count >= 0)
);

CREATE TABLE IF NOT EXISTS incidents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    incident_type TEXT NOT NULL,
    location      TEXT NOT NULL,
    description   TEXT,
    severity      TEXT NOT NULL,
    status        TEXT NOT NULL,
    is_sample     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fusion_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT    NOT NULL,
    session_id         INTEGER REFERENCES detection_sessions(id) ON DELETE SET NULL,
    incident_id        INTEGER REFERENCES incidents(id) ON DELETE SET NULL,
    inputs_json        TEXT    NOT NULL,
    priority_score     REAL    NOT NULL,
    priority_category  TEXT    NOT NULL,
    explanation        TEXT    NOT NULL,
    recommended_action TEXT    NOT NULL,
    is_sample          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS processing_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    level      TEXT NOT NULL,
    event      TEXT NOT NULL,
    details    TEXT
);

CREATE INDEX IF NOT EXISTS idx_counts_session ON vehicle_counts(session_id);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_fusion_created ON fusion_results(created_at);
"""


def now_iso() -> str:
    """Current local time as an ISO string, e.g. 2026-10-27T20:30:00."""
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def get_connection(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, roll back on error, always close."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str) -> None:
    """Create the database file and all tables if they do not exist."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


# --------------------------------------------------------------------------
# Detection sessions
# --------------------------------------------------------------------------
def insert_detection_session(
    db_path: Path | str,
    *,
    source_type: str,
    counts: dict[str, float],
    total_vehicles: int,
    congestion_score: Optional[float],
    congestion_category: Optional[str],
    file_name: Optional[str] = None,
    stored_path: Optional[str] = None,
    annotated_path: Optional[str] = None,
    storage_backend: str = "local",
    location: Optional[str] = None,
    model_name: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    frames_processed: int = 1,
    avg_vehicles: Optional[float] = None,
    peak_vehicles: Optional[int] = None,
    is_sample: bool = False,
    created_at: Optional[str] = None,
) -> int:
    """Save one detection session and its per-class counts (one transaction)."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO detection_sessions (
                created_at, source_type, file_name, stored_path, annotated_path,
                storage_backend, location, model_name, confidence_threshold,
                total_vehicles, frames_processed, avg_vehicles, peak_vehicles,
                congestion_score, congestion_category, is_sample
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at or now_iso(), source_type, file_name, stored_path, annotated_path,
                storage_backend, location, model_name, confidence_threshold,
                int(total_vehicles), int(frames_processed), avg_vehicles, peak_vehicles,
                congestion_score, congestion_category, int(is_sample),
            ),
        )
        session_id = int(cur.lastrowid)
        conn.executemany(
            "INSERT INTO vehicle_counts (session_id, vehicle_class, count) VALUES (?, ?, ?)",
            [(session_id, cls, float(n)) for cls, n in counts.items()],
        )
    return session_id


def get_detection_session(db_path: Path | str, session_id: int) -> Optional[dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM detection_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        session = dict(row)
        session["counts"] = {
            r["vehicle_class"]: r["count"]
            for r in conn.execute(
                "SELECT vehicle_class, count FROM vehicle_counts WHERE session_id = ?",
                (session_id,),
            )
        }
    return session


def list_detection_sessions(
    db_path: Path | str,
    *,
    include_sample: bool = True,
    source_type: Optional[str] = None,
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    """List sessions, newest first. Dates are 'YYYY-MM-DD' strings (inclusive)."""
    query = "SELECT * FROM detection_sessions WHERE 1 = 1"
    params: list[Any] = []
    if not include_sample:
        query += " AND is_sample = 0"
    if source_type:
        query += " AND source_type = ?"
        params.append(source_type)
    if category:
        query += " AND congestion_category = ?"
        params.append(category)
    if date_from:
        query += " AND date(created_at) >= date(?)"
        params.append(date_from)
    if date_to:
        query += " AND date(created_at) <= date(?)"
        params.append(date_to)
    query += " ORDER BY created_at DESC, id DESC"
    if limit:
        query += " LIMIT ?"
        params.append(int(limit))
    with get_connection(db_path) as conn:
        return _rows(conn.execute(query, params))


def list_vehicle_counts(db_path: Path | str, *, include_sample: bool = True) -> list[dict[str, Any]]:
    """All per-class counts joined with their session date."""
    query = """
        SELECT vc.session_id, vc.vehicle_class, vc.count, ds.created_at, ds.is_sample
        FROM vehicle_counts vc
        JOIN detection_sessions ds ON ds.id = vc.session_id
    """
    if not include_sample:
        query += " WHERE ds.is_sample = 0"
    with get_connection(db_path) as conn:
        return _rows(conn.execute(query))


# --------------------------------------------------------------------------
# Incidents
# --------------------------------------------------------------------------
def insert_incident(
    db_path: Path | str,
    *,
    incident_type: str,
    location: str,
    description: str,
    severity: str,
    status: str,
    is_sample: bool = False,
    created_at: Optional[str] = None,
) -> int:
    timestamp = created_at or now_iso()
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO incidents (created_at, updated_at, incident_type, location,
                                   description, severity, status, is_sample)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, timestamp, incident_type, location, description, severity, status, int(is_sample)),
        )
        return int(cur.lastrowid)


ALLOWED_INCIDENT_FIELDS = {"incident_type", "location", "description", "severity", "status"}


def update_incident(db_path: Path | str, incident_id: int, **fields: Any) -> bool:
    """Update the given fields of an incident. Returns False if it does not exist."""
    unknown = set(fields) - ALLOWED_INCIDENT_FIELDS
    if unknown:
        raise ValueError(f"Cannot update unknown field(s): {sorted(unknown)}")
    if not fields:
        return get_incident(db_path, incident_id) is not None
    # Column names come only from the fixed allow-list above, values use "?".
    assignments = ", ".join(f"{name} = ?" for name in fields)
    params = list(fields.values()) + [now_iso(), incident_id]
    with get_connection(db_path) as conn:
        cur = conn.execute(
            f"UPDATE incidents SET {assignments}, updated_at = ? WHERE id = ?", params
        )
        return cur.rowcount > 0


def get_incident(db_path: Path | str, incident_id: int) -> Optional[dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return dict(row) if row else None


def list_incidents(
    db_path: Path | str,
    *,
    include_sample: bool = True,
    incident_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM incidents WHERE 1 = 1"
    params: list[Any] = []
    if not include_sample:
        query += " AND is_sample = 0"
    for column, value in (("incident_type", incident_type), ("severity", severity), ("status", status)):
        if value:
            query += f" AND {column} = ?"
            params.append(value)
    if date_from:
        query += " AND date(created_at) >= date(?)"
        params.append(date_from)
    if date_to:
        query += " AND date(created_at) <= date(?)"
        params.append(date_to)
    query += " ORDER BY created_at DESC, id DESC"
    with get_connection(db_path) as conn:
        return _rows(conn.execute(query, params))


# --------------------------------------------------------------------------
# Fusion results
# --------------------------------------------------------------------------
def insert_fusion_result(
    db_path: Path | str,
    *,
    inputs: dict[str, Any],
    priority_score: float,
    priority_category: str,
    explanation: str,
    recommended_action: str,
    session_id: Optional[int] = None,
    incident_id: Optional[int] = None,
    is_sample: bool = False,
    created_at: Optional[str] = None,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO fusion_results (created_at, session_id, incident_id, inputs_json,
                priority_score, priority_category, explanation, recommended_action, is_sample)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at or now_iso(), session_id, incident_id, json.dumps(inputs, sort_keys=True),
                float(priority_score), priority_category, explanation, recommended_action, int(is_sample),
            ),
        )
        return int(cur.lastrowid)


def list_fusion_results(
    db_path: Path | str,
    *,
    include_sample: bool = True,
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict[str, Any]]:
    query = "SELECT * FROM fusion_results WHERE 1 = 1"
    params: list[Any] = []
    if not include_sample:
        query += " AND is_sample = 0"
    if category:
        query += " AND priority_category = ?"
        params.append(category)
    if date_from:
        query += " AND date(created_at) >= date(?)"
        params.append(date_from)
    if date_to:
        query += " AND date(created_at) <= date(?)"
        params.append(date_to)
    query += " ORDER BY created_at DESC, id DESC"
    with get_connection(db_path) as conn:
        return _rows(conn.execute(query, params))


def get_fusion_result(db_path: Path | str, fusion_id: int) -> Optional[dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM fusion_results WHERE id = ?", (fusion_id,)).fetchone()
        return dict(row) if row else None


# --------------------------------------------------------------------------
# Logs, statistics, sample data, backup
# --------------------------------------------------------------------------
def log_event(db_path: Path | str, level: str, event: str, details: str = "") -> None:
    """Write a processing log row. Never raises (logging must not break the app)."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO processing_logs (created_at, level, event, details) VALUES (?, ?, ?, ?)",
                (now_iso(), level.upper(), event, details[:2000]),
            )
    except sqlite3.Error as exc:  # pragma: no cover
        logger.warning("Could not write processing log: %s", exc)


def list_logs(db_path: Path | str, limit: int = 100) -> list[dict[str, Any]]:
    with get_connection(db_path) as conn:
        return _rows(
            conn.execute("SELECT * FROM processing_logs ORDER BY id DESC LIMIT ?", (int(limit),))
        )


def get_dashboard_stats(db_path: Path | str, *, include_sample: bool = True) -> dict[str, Any]:
    sample_filter = "" if include_sample else " WHERE is_sample = 0"
    sample_and = "" if include_sample else " AND is_sample = 0"
    with get_connection(db_path) as conn:
        sessions, vehicles = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(total_vehicles), 0) FROM detection_sessions{sample_filter}"
        ).fetchone()
        incidents = conn.execute(f"SELECT COUNT(*) FROM incidents{sample_filter}").fetchone()[0]
        open_incidents = conn.execute(
            f"SELECT COUNT(*) FROM incidents WHERE status != 'Resolved'{sample_and}"
        ).fetchone()[0]
        high_congestion = conn.execute(
            "SELECT COUNT(*) FROM detection_sessions "
            f"WHERE congestion_category IN ('High', 'Severe'){sample_and}"
        ).fetchone()[0]
        fusion = conn.execute(f"SELECT COUNT(*) FROM fusion_results{sample_filter}").fetchone()[0]
    return {
        "total_sessions": int(sessions),
        "total_vehicles": int(vehicles),
        "total_incidents": int(incidents),
        "open_incidents": int(open_incidents),
        "high_congestion": int(high_congestion),
        "fusion_results": int(fusion),
    }


def delete_sample_data(db_path: Path | str) -> int:
    """Delete every row marked as sample data. Returns number of rows removed."""
    with get_connection(db_path) as conn:
        removed = 0
        for table in ("fusion_results", "detection_sessions", "incidents"):
            removed += conn.execute(f"DELETE FROM {table} WHERE is_sample = 1").rowcount
    return removed


def backup_database(db_path: Path | str, backup_dir: Path | str) -> Path:
    """Create a consistent copy of the database using SQLite's backup API."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"traffic_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
    source = sqlite3.connect(str(db_path))
    destination = sqlite3.connect(str(target))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return target
