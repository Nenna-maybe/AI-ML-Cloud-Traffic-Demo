"""
traffic.py - Estimated congestion indicator.

IMPORTANT: one camera image cannot measure real traffic density, speed,
road capacity or queue length. This module produces an *estimated
congestion indicator based on observed vehicle counts*. It is a simple,
transparent rule, not a validated traffic-engineering measurement.

How the score is calculated
---------------------------
1. Each detected vehicle is converted to "vehicle units" using a weight
   (car 1.0, motorcycle 0.5, bus 3.0, truck 2.5) because larger vehicles
   occupy more road space. Unknown classes get weight 1.0.
2. weighted_units = sum(count x weight)
3. score = min(100, weighted_units / reference_capacity x 100)
   reference_capacity (default 30) is the number of vehicle units that we
   treat as a "full" camera view.
4. Category from the score:
      score <  moderate_threshold (25) -> Low
      score <  high_threshold     (50) -> Moderate
      score <  severe_threshold   (75) -> High
      otherwise                        -> Severe
All numbers are configurable in config.py / .env.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import CongestionSettings

CATEGORIES = ["Low", "Moderate", "High", "Severe"]


@dataclass
class CongestionResult:
    total_vehicles: float
    weighted_units: float
    score: float
    category: str
    explanation: str


def validate_thresholds(settings: CongestionSettings) -> None:
    if settings.reference_capacity <= 0:
        raise ValueError("reference_capacity must be greater than 0")
    if not (0 < settings.moderate_threshold < settings.high_threshold < settings.severe_threshold <= 100):
        raise ValueError("Thresholds must satisfy 0 < moderate < high < severe <= 100")


def categorize(score: float, settings: CongestionSettings) -> str:
    if score < settings.moderate_threshold:
        return "Low"
    if score < settings.high_threshold:
        return "Moderate"
    if score < settings.severe_threshold:
        return "High"
    return "Severe"


def estimate_congestion(
    counts: dict[str, float], settings: Optional[CongestionSettings] = None
) -> CongestionResult:
    """Turn per-class vehicle counts into a congestion score and category.

    `counts` can hold whole numbers (one image) or averages (video frames).
    """
    settings = settings or CongestionSettings()
    validate_thresholds(settings)

    weighted_units = 0.0
    total = 0.0
    parts = []
    for vehicle_class, count in counts.items():
        if count is None:
            continue
        count = float(count)
        if count < 0:
            raise ValueError(f"Vehicle count for '{vehicle_class}' cannot be negative")
        weight = settings.vehicle_weights.get(vehicle_class, 1.0)
        weighted_units += count * weight
        total += count
        if count:
            parts.append(f"{count:g} {vehicle_class} x {weight:g}")

    score = round(min(100.0, weighted_units / settings.reference_capacity * 100.0), 2)
    category = categorize(score, settings)
    breakdown = " + ".join(parts) if parts else "no vehicles"
    explanation = (
        f"Vehicle units = {breakdown} = {weighted_units:g}. "
        f"Score = min(100, {weighted_units:g} / {settings.reference_capacity:g} x 100) = {score:g}. "
        f"Category '{category}' (Low < {settings.moderate_threshold:g} <= Moderate < "
        f"{settings.high_threshold:g} <= High < {settings.severe_threshold:g} <= Severe). "
        "This is an estimated indicator based on observed vehicle counts only."
    )
    return CongestionResult(
        total_vehicles=round(total, 2),
        weighted_units=round(weighted_units, 2),
        score=score,
        category=category,
        explanation=explanation,
    )
