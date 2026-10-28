import pytest

import database as db
import incident_manager as im

VALID = {"incident_type": "Accident report", "location": "  Anna   Salai  ", "severity": "High",
         "description": "Two cars blocking left lane"}


def test_create_incident_defaults_to_open(db_path):
    iid = im.create_incident(db_path, VALID)
    row = db.get_incident(db_path, iid)
    assert row["status"] == "Open"
    assert row["location"] == "Anna Salai"  # whitespace cleaned
    assert row["created_at"] == row["updated_at"]


def test_status_update(db_path):
    iid = im.create_incident(db_path, VALID)
    im.set_status(db_path, iid, "In Progress")
    assert db.get_incident(db_path, iid)["status"] == "In Progress"
    im.set_status(db_path, iid, "Resolved")
    assert db.get_incident(db_path, iid)["status"] == "Resolved"


def test_edit_multiple_fields(db_path):
    iid = im.create_incident(db_path, VALID)
    im.edit_incident(db_path, iid, {"severity": "Critical", "location": "OMR Road"})
    row = db.get_incident(db_path, iid)
    assert (row["severity"], row["location"], row["incident_type"]) == ("Critical", "OMR Road", "Accident report")


@pytest.mark.parametrize("field, value", [
    ("incident_type", "Alien landing"),
    ("severity", "Huge"),
    ("status", "Closed"),
    ("location", "ab"),
    ("location", "x" * 200),
    ("description", "y" * 1001),
])
def test_invalid_inputs_rejected(db_path, field, value):
    with pytest.raises(im.IncidentValidationError):
        im.create_incident(db_path, {**VALID, field: value})
    assert db.list_incidents(db_path) == []


def test_multiple_errors_reported_together():
    with pytest.raises(im.IncidentValidationError) as info:
        im.validate_incident({"incident_type": "?", "location": "", "severity": "?", "status": "?"})
    assert len(info.value.errors) == 4


def test_invalid_status_update_rejected(db_path):
    iid = im.create_incident(db_path, VALID)
    with pytest.raises(im.IncidentValidationError):
        im.set_status(db_path, iid, "Done")
    assert db.get_incident(db_path, iid)["status"] == "Open"


def test_edit_missing_incident(db_path):
    with pytest.raises(LookupError):
        im.set_status(db_path, 404, "Resolved")


def test_filters(db_path):
    im.create_incident(db_path, VALID)
    im.create_incident(db_path, {**VALID, "incident_type": "Roadwork", "severity": "Low", "status": "Resolved"})
    assert len(db.list_incidents(db_path, incident_type="Roadwork")) == 1
    assert len(db.list_incidents(db_path, severity="High")) == 1
    assert len(db.list_incidents(db_path, status="Resolved")) == 1
    assert len(db.list_incidents(db_path, date_from="2000-01-01")) == 2
    assert db.list_incidents(db_path, date_to="2000-01-01") == []


def test_authorization():
    assert im.is_authorized("", "")          # no PIN configured -> open demo mode
    assert im.is_authorized("1234", "1234")
    assert not im.is_authorized("0000", "1234")
    assert not im.is_authorized("", "1234")


def test_incident_actions_are_logged(db_path):
    iid = im.create_incident(db_path, VALID)
    im.set_status(db_path, iid, "Resolved")
    events = [log["event"] for log in db.list_logs(db_path)]
    assert "incident_created" in events and "incident_updated" in events
