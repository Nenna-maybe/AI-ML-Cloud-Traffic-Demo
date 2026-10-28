"""
fusion.py - Adaptive Cross-Signal Fusion Engine.

This is a deterministic, rule-based DECISION-SUPPORT algorithm.
It is NOT a trained machine-learning model.

Signals combined
----------------
* Congestion score (0-100) from traffic.py   -> congestion component C
* Incident severity, status and recency       -> incident component I
* Vehicle count is NOT added again: the congestion score is already
  calculated from the vehicle count, so adding it twice would double-count
  the same evidence. The count is only used as a sanity check and shown
  in the explanation.
* The congestion *category* is also derived from the score, so it is not
  added separately either.

Algorithm
---------
1. C = congestion_score (if available)
2. I = severity_points[severity] x status_factor[status] x recency_factor
      severity_points : Low 25, Medium 50, High 75, Critical 100
      status_factor   : Open 1.0, In Progress 0.7, Resolved 0.0
      recency_factor  : 1.0 for the first hour, then linear decay to 0.5
                        at 24 hours, and 0.5 after that
3. If both C and I are available:  score = wC x C + wI x I
   (default wC = 0.45, wI = 0.55; weights are normalised to sum to 1)
   If only one is available, that one is used alone (weight 1.0) and the
   explanation says so.
   If neither is available -> ValueError (not enough information).
4. Safety rule: an active (not resolved) Critical incident never scores
   below `critical_incident_floor` (75).
5. Category: score < 30 Low, < 55 Medium, < 75 High, otherwise Critical.

Same inputs (including the `now` time) always give the same output, so any
stored result can be reproduced exactly from its saved inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from config import FusionSettings

PRIORITY_CATEGORIES = ["Low", "Medium", "High", "Critical"]

RECOMMENDED_ACTIONS = {
    "Low": "Continue routine monitoring.",
    "Medium": "Monitor closely and re-check this location within 30 minutes.",
    "High": "Notify the traffic-control team and prepare diversion signage.",
    "Critical": "Escalate immediately: dispatch response and activate the diversion plan.",
}


@dataclass
class FusionInput:
    congestion_score: Optional[float] = None
    congestion_category: Optional[str] = None
    vehicle_count: Optional[float] = None
    incident_severity: Optional[str] = None
    incident_status: Optional[str] = None
    incident_reported_at: Optional[str] = None  # ISO timestamp
    now: Optional[str] = None  # ISO timestamp used for recency; stored for reproducibility

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FusionInput":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class FusionResult:
    priority_score: float
    priority_category: str
    recommended_action: str
    reasons: list[str] = field(default_factory=list)
    components: dict[str, Optional[float]] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def explanation(self) -> str:
        return " ".join(self.reasons)


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc


def recency_factor(age_hours: Optional[float], settings: FusionSettings) -> float:
    """1.0 for fresh incidents, decaying linearly to the floor value."""
    if age_hours is None or age_hours <= settings.recency_full_hours:
        return 1.0
    if age_hours >= settings.recency_decay_hours:
        return settings.recency_floor
    span = settings.recency_decay_hours - settings.recency_full_hours
    progress = (age_hours - settings.recency_full_hours) / span
    return round(1.0 - progress * (1.0 - settings.recency_floor), 4)


def priority_category(score: float, settings: FusionSettings) -> str:
    if score < settings.medium_threshold:
        return "Low"
    if score < settings.high_threshold:
        return "Medium"
    if score < settings.critical_threshold:
        return "High"
    return "Critical"


def compute_priority(data: FusionInput, settings: Optional[FusionSettings] = None) -> FusionResult:
    settings = settings or FusionSettings()
    reasons: list[str] = []

    # ---- Validate inputs -------------------------------------------------
    if data.congestion_score is not None and not 0 <= float(data.congestion_score) <= 100:
        raise ValueError("congestion_score must be between 0 and 100")
    if data.vehicle_count is not None and float(data.vehicle_count) < 0:
        raise ValueError("vehicle_count cannot be negative")
    if data.incident_severity is not None and data.incident_severity not in settings.severity_points:
        raise ValueError(f"Unknown incident severity: {data.incident_severity}")
    if data.incident_status is not None and data.incident_status not in settings.status_factor:
        raise ValueError(f"Unknown incident status: {data.incident_status}")
    if settings.congestion_weight < 0 or settings.incident_weight < 0:
        raise ValueError("Fusion weights cannot be negative")
    if settings.congestion_weight + settings.incident_weight == 0:
        raise ValueError("At least one fusion weight must be greater than 0")

    # ---- Congestion component ------------------------------------------
    congestion = float(data.congestion_score) if data.congestion_score is not None else None
    if congestion is not None:
        label = f" ({data.congestion_category})" if data.congestion_category else ""
        count_text = (
            f" from {data.vehicle_count:g} observed vehicles" if data.vehicle_count is not None else ""
        )
        reasons.append(f"Estimated congestion score is {congestion:g}/100{label}{count_text}.")
    else:
        reasons.append("No congestion data was provided.")

    # ---- Incident component --------------------------------------------
    incident: Optional[float] = None
    if data.incident_severity:
        status = data.incident_status or "Open"
        if data.incident_status is None:
            reasons.append("Incident status missing, so it was treated as Open (cautious default).")
        reported = _parse_time(data.incident_reported_at)
        now = _parse_time(data.now) or datetime.now()
        age_hours = None
        if reported is not None:
            age_hours = max(0.0, (now - reported).total_seconds() / 3600)
        factor = recency_factor(age_hours, settings)
        points = settings.severity_points[data.incident_severity]
        status_factor = settings.status_factor[status]
        incident = round(points * status_factor * factor, 2)
        age_text = f"{age_hours:.1f} h old" if age_hours is not None else "age unknown"
        reasons.append(
            f"Incident: {data.incident_severity} severity ({points:g} pts) x status '{status}' "
            f"({status_factor:g}) x recency ({age_text}, {factor:g}) = {incident:g}."
        )
        if status == "Resolved":
            reasons.append("The incident is resolved, so it adds no urgency.")
    else:
        reasons.append("No incident linked.")

    if congestion is None and incident is None:
        raise ValueError("Not enough information: provide congestion data and/or an incident.")

    # ---- Combine ----------------------------------------------------------
    if congestion is not None and incident is not None:
        total_w = settings.congestion_weight + settings.incident_weight
        wc = settings.congestion_weight / total_w
        wi = settings.incident_weight / total_w
        score = wc * congestion + wi * incident
        reasons.append(
            f"Priority = {wc:.2f} x {congestion:g} + {wi:.2f} x {incident:g} = {score:.2f}."
        )
    elif congestion is not None:
        score = congestion
        reasons.append("Only congestion data available, so priority = congestion score.")
    else:
        score = float(incident)  # type: ignore[arg-type]
        reasons.append("Only incident data available, so priority = incident score.")

    # ---- Safety floor for active critical incidents --------------------
    active_critical = (
        data.incident_severity == "Critical" and (data.incident_status or "Open") != "Resolved"
    )
    if active_critical and score < settings.critical_incident_floor:
        reasons.append(
            f"Safety rule: an active Critical incident raises the score to at least "
            f"{settings.critical_incident_floor:g}."
        )
        score = settings.critical_incident_floor

    if data.vehicle_count == 0 and incident is None:
        reasons.append("No vehicles were observed; the result reflects an empty scene.")

    score = round(min(100.0, max(0.0, score)), 2)
    category = priority_category(score, settings)
    reasons.append(f"Final priority: {category} ({score:g}/100).")

    return FusionResult(
        priority_score=score,
        priority_category=category,
        recommended_action=RECOMMENDED_ACTIONS[category],
        reasons=reasons,
        components={"congestion": congestion, "incident": incident},
        inputs=data.to_dict(),
    )
