import pytest

from config import CongestionSettings
from traffic import categorize, estimate_congestion


def test_empty_scene_is_low():
    result = estimate_congestion({})
    assert result.score == 0
    assert result.category == "Low"
    assert result.total_vehicles == 0


def test_weighted_units_and_score():
    # 10 cars (10) + 4 motorcycles (2) + 1 bus (3) + 2 trucks (5) = 20 units -> 20/30*100
    result = estimate_congestion({"car": 10, "motorcycle": 4, "bus": 1, "truck": 2})
    assert result.weighted_units == 20
    assert result.total_vehicles == 17
    assert result.score == pytest.approx(66.67, abs=0.01)
    assert result.category == "High"


def test_score_is_capped_at_100():
    assert estimate_congestion({"bus": 50}).score == 100
    assert estimate_congestion({"bus": 50}).category == "Severe"


@pytest.mark.parametrize(
    "score, expected",
    [(0, "Low"), (24.99, "Low"), (25, "Moderate"), (49.99, "Moderate"),
     (50, "High"), (74.99, "High"), (75, "Severe"), (100, "Severe")],
)
def test_category_boundaries(score, expected):
    assert categorize(score, CongestionSettings()) == expected


def test_configurable_thresholds_and_capacity():
    settings = CongestionSettings(reference_capacity=10, moderate_threshold=10,
                                  high_threshold=20, severe_threshold=90)
    result = estimate_congestion({"car": 2}, settings)  # 2/10*100 = 20
    assert result.score == 20
    assert result.category == "High"


def test_unknown_class_uses_weight_one():
    assert estimate_congestion({"bicycle": 3}).weighted_units == 3


def test_fractional_counts_from_video_averages():
    result = estimate_congestion({"car": 1.5, "motorcycle": 0.5})
    assert result.weighted_units == 1.75


def test_negative_count_rejected():
    with pytest.raises(ValueError):
        estimate_congestion({"car": -1})


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        estimate_congestion({"car": 1}, CongestionSettings(moderate_threshold=60, high_threshold=50))
    with pytest.raises(ValueError):
        estimate_congestion({"car": 1}, CongestionSettings(reference_capacity=0))


def test_explanation_mentions_estimate():
    assert "estimated indicator" in estimate_congestion({"car": 3}).explanation
