import sqlite3

import pandas as pd
import pytest

import analytics
import database as db
from config import load_settings


def add_session(path, **kw):
    params = dict(source_type="image", counts={"car": 3, "bus": 1}, total_vehicles=4,
                  congestion_score=20.0, congestion_category="Low")
    params.update(kw)
    return db.insert_detection_session(path, **params)


def test_init_creates_tables(db_path):
    with sqlite3.connect(db_path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"detection_sessions", "vehicle_counts", "incidents", "fusion_results", "processing_logs"} <= names


def test_init_is_idempotent(db_path):
    add_session(db_path)
    db.init_db(db_path)  # running again must not delete data
    assert len(db.list_detection_sessions(db_path)) == 1


def test_session_and_counts_persist(db_path):
    sid = add_session(db_path, file_name="cam1.jpg", location="MG Road")
    loaded = db.get_detection_session(db_path, sid)
    assert loaded["file_name"] == "cam1.jpg"
    assert loaded["counts"] == {"car": 3.0, "bus": 1.0}
    assert db.get_detection_session(db_path, 9999) is None


def test_failed_insert_rolls_back(db_path):
    with pytest.raises(sqlite3.IntegrityError):
        add_session(db_path, counts={"car": -5})  # CHECK constraint fails on counts
    assert db.list_detection_sessions(db_path) == []  # session row rolled back too


def test_cascade_delete_counts(db_path):
    sid = add_session(db_path)
    with db.get_connection(db_path) as conn:
        conn.execute("DELETE FROM detection_sessions WHERE id = ?", (sid,))
    assert db.list_vehicle_counts(db_path) == []


def test_session_filters(db_path):
    add_session(db_path, congestion_category="High", created_at="2026-10-01T10:00:00")
    add_session(db_path, source_type="video", created_at="2026-10-05T10:00:00")
    add_session(db_path, source_type="sample", is_sample=True, created_at="2025-01-01T10:00:00")
    assert len(db.list_detection_sessions(db_path)) == 3
    assert len(db.list_detection_sessions(db_path, include_sample=False)) == 2
    assert len(db.list_detection_sessions(db_path, category="High")) == 1
    assert len(db.list_detection_sessions(db_path, source_type="video")) == 1
    assert len(db.list_detection_sessions(db_path, date_from="2026-10-02", date_to="2026-10-31")) == 1
    assert len(db.list_detection_sessions(db_path, limit=1)) == 1


def test_sql_injection_is_harmless(db_path):
    add_session(db_path)
    assert db.list_detection_sessions(db_path, category="x' OR '1'='1") == []
    assert len(db.list_detection_sessions(db_path)) == 1


def test_fusion_result_storage(db_path):
    sid = add_session(db_path)
    fid = db.insert_fusion_result(db_path, inputs={"congestion_score": 20}, priority_score=20,
                                  priority_category="Low", explanation="x", recommended_action="y",
                                  session_id=sid)
    row = db.get_fusion_result(db_path, fid)
    assert row["session_id"] == sid
    assert analytics.stored_inputs(row) == {"congestion_score": 20}


def test_update_incident_rejects_unknown_columns(db_path):
    iid = db.insert_incident(db_path, incident_type="Other", location="Road", description="",
                             severity="Low", status="Open")
    with pytest.raises(ValueError):
        db.update_incident(db_path, iid, created_at="2000-01-01")


def test_dashboard_stats(db_path):
    add_session(db_path, congestion_category="Severe", total_vehicles=10)
    add_session(db_path, is_sample=True, source_type="sample", congestion_category="High")
    db.insert_incident(db_path, incident_type="Other", location="Road", description="",
                       severity="Low", status="Resolved")
    stats = db.get_dashboard_stats(db_path, include_sample=False)
    assert stats["total_sessions"] == 1
    assert stats["total_vehicles"] == 10
    assert stats["high_congestion"] == 1
    assert stats["total_incidents"] == 1
    assert stats["open_incidents"] == 0
    assert db.get_dashboard_stats(db_path)["high_congestion"] == 2


def test_sample_data_load_and_delete(db_path):
    counts = analytics.load_sample_data(db_path, load_settings())
    assert counts["sessions"] > 0 and counts["incidents"] == 5 and counts["fusion"] == 5
    assert db.list_detection_sessions(db_path, include_sample=False) == []
    real = add_session(db_path)
    removed = db.delete_sample_data(db_path)
    assert removed == counts["sessions"] + counts["incidents"] + counts["fusion"]
    remaining = db.list_detection_sessions(db_path)
    assert [r["id"] for r in remaining] == [real]


def test_backup_creates_copy(db_path, tmp_path):
    add_session(db_path)
    backup = db.backup_database(db_path, tmp_path / "backups")
    assert backup.exists()
    assert len(db.list_detection_sessions(backup)) == 1


def test_logs(db_path):
    db.log_event(db_path, "info", "test_event", "details")
    logs = db.list_logs(db_path)
    assert logs[0]["level"] == "INFO" and logs[0]["event"] == "test_event"


def test_csv_export_roundtrip(db_path):
    add_session(db_path, file_name="a.jpg")
    df = analytics.label_sample(analytics.sessions_df(db_path))
    csv_bytes = analytics.to_csv_bytes(df)
    back = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    assert list(back["file_name"]) == ["a.jpg"]
    assert list(back["data"]) == ["real"]


def test_analytics_handles_empty_database(db_path):
    sessions = analytics.sessions_df(db_path)
    incidents = analytics.incidents_df(db_path)
    fusion = analytics.fusion_df(db_path)
    assert sessions.empty and incidents.empty and fusion.empty
    assert analytics.congestion_distribution(sessions) is None
    assert analytics.priority_distribution(fusion) is None
    assert analytics.vehicle_class_chart(analytics.class_totals_df(db_path)) is None
    assert analytics.incidents_by_type_severity(incidents) is None
    assert analytics.incident_status_chart(incidents) is None
    assert analytics.priority_histogram(fusion) is None
    assert analytics.daily_trend(sessions, incidents) is None
    assert analytics.to_csv_bytes(sessions) is not None


def test_analytics_charts_with_data(db_path):
    analytics.load_sample_data(db_path, load_settings())
    sessions = analytics.sessions_df(db_path)
    incidents = analytics.incidents_df(db_path)
    fusion = analytics.fusion_df(db_path)
    assert analytics.congestion_distribution(sessions) is not None
    assert analytics.priority_distribution(fusion) is not None
    assert analytics.daily_trend(sessions, incidents) is not None
    totals = analytics.class_totals_df(db_path)
    assert set(totals["vehicle_class"]) == {"car", "motorcycle", "bus", "truck"}
