"""
analytics.py - Turns database rows into DataFrames, charts and CSV exports.

Every function handles empty data and returns None (for charts) or an
empty DataFrame instead of crashing.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import database as db
from fusion import FusionInput, compute_priority
from incident_manager import create_incident
from traffic import CATEGORIES as CONGESTION_ORDER
from traffic import estimate_congestion
from fusion import PRIORITY_CATEGORIES as PRIORITY_ORDER

COLORS = {
    "Low": "#2e9e5b", "Moderate": "#e0b100", "Medium": "#e0b100",
    "High": "#e8730c", "Severe": "#d62828", "Critical": "#d62828",
    "Open": "#d62828", "In Progress": "#e8730c", "Resolved": "#2e9e5b",
}


# --------------------------------------------------------------------------
# DataFrames
# --------------------------------------------------------------------------
def sessions_df(db_path, **filters) -> pd.DataFrame:
    df = pd.DataFrame(db.list_detection_sessions(db_path, **filters))
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def incidents_df(db_path, **filters) -> pd.DataFrame:
    df = pd.DataFrame(db.list_incidents(db_path, **filters))
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def fusion_df(db_path, **filters) -> pd.DataFrame:
    df = pd.DataFrame(db.list_fusion_results(db_path, **filters))
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def class_totals_df(db_path, include_sample: bool = True) -> pd.DataFrame:
    rows = db.list_vehicle_counts(db_path, include_sample=include_sample)
    if not rows:
        return pd.DataFrame(columns=["vehicle_class", "count"])
    df = pd.DataFrame(rows)
    return df.groupby("vehicle_class", as_index=False)["count"].sum().sort_values("count", ascending=False)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def label_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable 'data' column so sample rows are clearly marked."""
    if df.empty or "is_sample" not in df.columns:
        return df
    df = df.copy()
    df.insert(1, "data", df["is_sample"].map({1: "SAMPLE", 0: "real"}))
    return df


# --------------------------------------------------------------------------
# Charts (each returns None when there is nothing to plot)
# --------------------------------------------------------------------------
def _style(fig: go.Figure) -> go.Figure:
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=330, legend_title_text="")
    return fig


def category_chart(df: pd.DataFrame, column: str, order: list[str], title: str) -> Optional[go.Figure]:
    if df.empty or column not in df or df[column].dropna().empty:
        return None
    counts = df[column].value_counts().reindex(order, fill_value=0).reset_index()
    counts.columns = [column, "records"]
    fig = px.bar(counts, x=column, y="records", color=column, color_discrete_map=COLORS, title=title)
    fig.update_layout(showlegend=False)
    return _style(fig)


def congestion_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    return category_chart(df, "congestion_category", CONGESTION_ORDER, "Congestion categories")


def priority_distribution(df: pd.DataFrame) -> Optional[go.Figure]:
    return category_chart(df, "priority_category", PRIORITY_ORDER, "Priority categories")


def vehicle_class_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty:
        return None
    fig = px.bar(df, x="vehicle_class", y="count", color="vehicle_class", title="Vehicles detected by class")
    fig.update_layout(showlegend=False)
    return _style(fig)


def counts_bar(counts: dict[str, float], title: str) -> Optional[go.Figure]:
    if not counts:
        return None
    df = pd.DataFrame({"vehicle_class": list(counts), "count": list(counts.values())})
    return vehicle_class_chart(df).update_layout(title=title)


def incidents_by_type_severity(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty:
        return None
    grouped = df.groupby(["incident_type", "severity"]).size().reset_index(name="incidents")
    fig = px.bar(
        grouped, x="incident_type", y="incidents", color="severity",
        category_orders={"severity": PRIORITY_ORDER},
        color_discrete_map={"Low": "#2e9e5b", "Medium": "#e0b100", "High": "#e8730c", "Critical": "#d62828"},
        title="Incidents by type and severity",
    )
    return _style(fig)


def incident_status_chart(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty:
        return None
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["status", "incidents"]
    fig = px.pie(counts, names="status", values="incidents", color="status",
                 color_discrete_map=COLORS, hole=0.45, title="Open vs resolved incidents")
    return _style(fig)


def priority_histogram(df: pd.DataFrame) -> Optional[go.Figure]:
    if df.empty:
        return None
    fig = px.histogram(df, x="priority_score", nbins=10, range_x=[0, 100], title="Priority score distribution")
    fig.update_layout(bargap=0.05)
    return _style(fig)


def daily_trend(sessions: pd.DataFrame, incidents: pd.DataFrame) -> Optional[go.Figure]:
    """Daily trend; needs data on at least 2 different days to be meaningful."""
    frames = []
    if not sessions.empty:
        s = sessions.assign(day=sessions["created_at"].dt.date).groupby("day")["total_vehicles"].sum()
        frames.append(s.rename("vehicles detected"))
    if not incidents.empty:
        i = incidents.assign(day=incidents["created_at"].dt.date).groupby("day").size()
        frames.append(i.rename("incidents reported"))
    if not frames:
        return None
    df = pd.concat(frames, axis=1).fillna(0).sort_index()
    if len(df) < 2:
        return None
    df = df.reset_index().melt(id_vars="day", var_name="metric", value_name="value")
    fig = px.line(df, x="day", y="value", color="metric", markers=True, title="Daily trend")
    return _style(fig)


def congestion_gauge(score: float, category: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Estimated congestion: {category}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f3a5f"},
                "steps": [
                    {"range": [0, 25], "color": "#d8f0e0"},
                    {"range": [25, 50], "color": "#fbf0c4"},
                    {"range": [50, 75], "color": "#fde0c5"},
                    {"range": [75, 100], "color": "#f8d0d0"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# --------------------------------------------------------------------------
# Sample data (clearly flagged with is_sample = 1)
# --------------------------------------------------------------------------
def load_sample_data(db_path: Path | str, settings, seed: int = 7) -> dict[str, int]:
    """Insert demo records marked as SAMPLE. They are never mixed with real results."""
    rng = random.Random(seed)
    now = datetime.now().replace(microsecond=0)
    session_ids = []
    for day in range(6, -1, -1):
        for _ in range(2):
            counts = {
                "car": rng.randint(2, 25), "motorcycle": rng.randint(0, 12),
                "bus": rng.randint(0, 3), "truck": rng.randint(0, 4),
            }
            result = estimate_congestion(counts, settings.congestion)
            created = now - timedelta(days=day, hours=rng.randint(0, 10))
            session_ids.append(
                db.insert_detection_session(
                    db_path, source_type="sample", counts=counts,
                    total_vehicles=int(sum(counts.values())),
                    congestion_score=result.score, congestion_category=result.category,
                    file_name="[SAMPLE] generated record", location=f"Sample Road {day + 1}",
                    model_name="none (sample data)", is_sample=True,
                    created_at=created.isoformat(),
                )
            )
    incident_specs = [
        ("Accident report", "Sample Road 1", "High", "Open"),
        ("Road blockage", "Sample Road 2", "Medium", "In Progress"),
        ("Stalled vehicle", "Sample Road 3", "Low", "Resolved"),
        ("Roadwork", "Sample Road 4", "Medium", "Open"),
        ("Accident report", "Sample Road 5", "Critical", "Resolved"),
    ]
    incident_ids = []
    for i, (itype, loc, sev, status) in enumerate(incident_specs):
        incident_id = create_incident(
            db_path,
            {"incident_type": itype, "location": loc, "severity": sev, "status": status,
             "description": "[SAMPLE] demonstration record"},
            is_sample=True,
        )
        created = (now - timedelta(days=i, hours=2)).isoformat()
        with db.get_connection(db_path) as conn:
            conn.execute("UPDATE incidents SET created_at = ?, updated_at = ? WHERE id = ?",
                         (created, created, incident_id))
        incident_ids.append(incident_id)

    fusion_count = 0
    for session_id, incident_id in zip(session_ids[-5:], incident_ids):
        session = db.get_detection_session(db_path, session_id)
        incident = db.get_incident(db_path, incident_id)
        data = FusionInput(
            congestion_score=session["congestion_score"],
            congestion_category=session["congestion_category"],
            vehicle_count=session["total_vehicles"],
            incident_severity=incident["severity"], incident_status=incident["status"],
            incident_reported_at=incident["created_at"], now=now.isoformat(),
        )
        result = compute_priority(data, settings.fusion)
        db.insert_fusion_result(
            db_path, inputs=result.inputs, priority_score=result.priority_score,
            priority_category=result.priority_category, explanation=result.explanation,
            recommended_action=result.recommended_action, session_id=session_id,
            incident_id=incident_id, is_sample=True,
        )
        fusion_count += 1
    db.log_event(db_path, "INFO", "sample_data_loaded", f"sessions={len(session_ids)}")
    return {"sessions": len(session_ids), "incidents": len(incident_ids), "fusion": fusion_count}


def stored_inputs(row: dict) -> dict:
    return json.loads(row["inputs_json"]) if row.get("inputs_json") else {}
