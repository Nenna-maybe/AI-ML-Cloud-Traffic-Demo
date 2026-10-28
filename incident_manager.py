"""
incident_manager.py - Validation and business rules for manually reported incidents.

Incidents are reported by a person through the app. The YOLO model does NOT
detect accidents automatically; it only detects vehicles.
"""

from __future__ import annotations

import hmac
from pathlib import Path
from typing import Any, Optional

import database as db

INCIDENT_TYPES = ["Accident report", "Road blockage", "Stalled vehicle", "Roadwork", "Other"]
SEVERITIES = ["Low", "Medium", "High", "Critical"]
STATUSES = ["Open", "In Progress", "Resolved"]

MAX_LOCATION_LEN = 120
MAX_DESCRIPTION_LEN = 1000


class IncidentValidationError(ValueError):
    """Raised when incident data is invalid. `errors` lists every problem."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def clean_text(value: Optional[str]) -> str:
    return " ".join((value or "").split())


def validate_incident(data: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """Return cleaned data or raise IncidentValidationError.

    partial=True validates only the fields that are present (used for edits).
    """
    errors: list[str] = []
    cleaned: dict[str, Any] = {}

    def present(key: str) -> bool:
        return not partial or key in data

    if present("incident_type"):
        if data.get("incident_type") not in INCIDENT_TYPES:
            errors.append(f"Incident type must be one of: {', '.join(INCIDENT_TYPES)}.")
        else:
            cleaned["incident_type"] = data["incident_type"]

    if present("location"):
        location = clean_text(data.get("location"))
        if len(location) < 3:
            errors.append("Location / road name must be at least 3 characters.")
        elif len(location) > MAX_LOCATION_LEN:
            errors.append(f"Location must be at most {MAX_LOCATION_LEN} characters.")
        else:
            cleaned["location"] = location

    if present("description"):
        description = (data.get("description") or "").strip()
        if len(description) > MAX_DESCRIPTION_LEN:
            errors.append(f"Description must be at most {MAX_DESCRIPTION_LEN} characters.")
        else:
            cleaned["description"] = description

    if present("severity"):
        if data.get("severity") not in SEVERITIES:
            errors.append(f"Severity must be one of: {', '.join(SEVERITIES)}.")
        else:
            cleaned["severity"] = data["severity"]

    if present("status"):
        if data.get("status") not in STATUSES:
            errors.append(f"Status must be one of: {', '.join(STATUSES)}.")
        else:
            cleaned["status"] = data["status"]

    if errors:
        raise IncidentValidationError(errors)
    return cleaned


def create_incident(db_path: Path | str, data: dict[str, Any], *, is_sample: bool = False) -> int:
    payload = {"status": "Open", "description": "", **data}
    cleaned = validate_incident(payload)
    incident_id = db.insert_incident(db_path, is_sample=is_sample, **cleaned)
    db.log_event(db_path, "INFO", "incident_created", f"id={incident_id} type={cleaned['incident_type']}")
    return incident_id


def edit_incident(db_path: Path | str, incident_id: int, changes: dict[str, Any]) -> None:
    cleaned = validate_incident(changes, partial=True)
    if not db.update_incident(db_path, incident_id, **cleaned):
        raise LookupError(f"Incident {incident_id} does not exist.")
    db.log_event(db_path, "INFO", "incident_updated", f"id={incident_id} fields={sorted(cleaned)}")


def set_status(db_path: Path | str, incident_id: int, status: str) -> None:
    edit_incident(db_path, incident_id, {"status": status})


def is_authorized(entered_pin: str, admin_pin: str) -> bool:
    """Simple shared-PIN check for editing incidents.

    If ADMIN_PIN is not configured, editing is open to anyone using the app
    (suitable for a local demo). This is a basic gate, NOT user authentication.
    """
    if not admin_pin:
        return True
    return hmac.compare_digest((entered_pin or "").encode(), admin_pin.encode())
