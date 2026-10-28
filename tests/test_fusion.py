import pytest

from config import FusionSettings
from fusion import FusionInput, compute_priority, priority_category, recency_factor

NOW = "2026-10-27T12:00:00"


def incident(severity, status="Open", reported="2026-10-27T11:30:00", **extra):
    return FusionInput(incident_severity=severity, incident_status=status,
                       incident_reported_at=reported, now=NOW, **extra)


def test_normal_case_combines_both_signals():
    data = FusionInput(congestion_score=40, congestion_category="Moderate", vehicle_count=12,
                       incident_severity="Medium", incident_status="Open",
                       incident_reported_at="2026-10-27T11:30:00", now=NOW)
    result = compute_priority(data)
    # 0.45*40 + 0.55*50 = 18 + 27.5 = 45.5
    assert result.priority_score == 45.5
    assert result.priority_category == "Medium"
    assert result.components == {"congestion": 40.0, "incident": 50.0}
    assert "Priority = 0.45 x 40 + 0.55 x 50" in result.explanation


def test_high_congestion_only():
    result = compute_priority(FusionInput(congestion_score=90, congestion_category="Severe", vehicle_count=40))
    assert result.priority_score == 90
    assert result.priority_category == "Critical"
    assert "Only congestion data" in result.explanation


def test_severe_incident_floor_applies():
    data = incident("Critical", congestion_score=0, vehicle_count=0)
    result = compute_priority(data)
    # weighted = 0.55*100 = 55 -> raised to floor 75
    assert result.priority_score == 75
    assert result.priority_category == "Critical"
    assert "Safety rule" in result.explanation


def test_resolved_incident_adds_no_urgency():
    data = incident("Critical", status="Resolved", congestion_score=20)
    result = compute_priority(data)
    assert result.components["incident"] == 0
    assert result.priority_score == pytest.approx(0.45 * 20)
    assert result.priority_category == "Low"
    assert "resolved" in result.explanation.lower()


def test_in_progress_reduces_incident_component():
    result = compute_priority(incident("High", status="In Progress"))
    assert result.components["incident"] == pytest.approx(75 * 0.7)


def test_missing_status_treated_as_open():
    result = compute_priority(FusionInput(incident_severity="High", now=NOW))
    assert result.components["incident"] == 75
    assert "treated as Open" in result.explanation


def test_no_information_raises():
    with pytest.raises(ValueError, match="Not enough information"):
        compute_priority(FusionInput(vehicle_count=5))


@pytest.mark.parametrize("bad", [
    FusionInput(congestion_score=101),
    FusionInput(congestion_score=-1),
    FusionInput(congestion_score=10, vehicle_count=-3),
    FusionInput(incident_severity="Extreme"),
    FusionInput(incident_severity="Low", incident_status="Closed"),
    FusionInput(incident_severity="Low", incident_reported_at="yesterday"),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        compute_priority(bad)


@pytest.mark.parametrize(
    "score, expected",
    [(0, "Low"), (29.99, "Low"), (30, "Medium"), (54.99, "Medium"),
     (55, "High"), (74.99, "High"), (75, "Critical"), (100, "Critical")],
)
def test_priority_boundaries(score, expected):
    assert priority_category(score, FusionSettings()) == expected


def test_boundary_through_full_computation():
    # congestion only, score exactly at High threshold
    assert compute_priority(FusionInput(congestion_score=55)).priority_category == "High"
    assert compute_priority(FusionInput(congestion_score=54.99)).priority_category == "Medium"


def test_recency_factor_decay():
    s = FusionSettings()
    assert recency_factor(None, s) == 1.0
    assert recency_factor(0.5, s) == 1.0
    assert recency_factor(1.0, s) == 1.0
    assert recency_factor(24, s) == 0.5
    assert recency_factor(100, s) == 0.5
    assert recency_factor(12.5, s) == pytest.approx(0.75)


def test_old_incident_counts_less():
    fresh = compute_priority(incident("High", reported="2026-10-27T11:45:00"))
    old = compute_priority(incident("High", reported="2026-10-24T12:00:00"))
    assert fresh.components["incident"] == 75
    assert old.components["incident"] == 37.5


def test_vehicle_count_not_double_counted():
    a = compute_priority(FusionInput(congestion_score=50, vehicle_count=1))
    b = compute_priority(FusionInput(congestion_score=50, vehicle_count=500))
    assert a.priority_score == b.priority_score


def test_result_reproducible_from_stored_inputs():
    data = FusionInput(congestion_score=63.3, vehicle_count=20, incident_severity="High",
                       incident_status="Open", incident_reported_at="2026-10-27T05:00:00", now=NOW)
    first = compute_priority(data)
    again = compute_priority(FusionInput.from_dict(first.inputs))
    assert again.priority_score == first.priority_score
    assert again.reasons == first.reasons


def test_custom_weights():
    settings = FusionSettings(congestion_weight=1, incident_weight=1)
    result = compute_priority(incident("Medium", congestion_score=30), settings)
    assert result.priority_score == 40  # (30 + 50) / 2


def test_zero_weights_rejected():
    with pytest.raises(ValueError):
        compute_priority(FusionInput(congestion_score=10), FusionSettings(congestion_weight=0, incident_weight=0))


def test_score_never_exceeds_100():
    result = compute_priority(incident("Critical", congestion_score=100))
    assert result.priority_score == 100
