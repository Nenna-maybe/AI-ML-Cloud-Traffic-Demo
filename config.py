"""
config.py - Central configuration for the Traffic Monitoring app.

All settings come from environment variables (or a local .env file).
Every setting has a safe default, so the app runs with NO configuration.

On Streamlit Community Cloud, values from the "Secrets" panel are copied
into environment variables by `load_streamlit_secrets()` (called in app.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is optional: the app still works without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else BASE_DIR / path


# COCO class IDs used by the pretrained YOLO model for vehicles.
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class CongestionSettings:
    """Settings for the estimated congestion indicator (see traffic.py)."""

    # "Vehicle units": a bus takes more road space than a motorcycle.
    vehicle_weights: dict[str, float] = field(
        default_factory=lambda: {
            "car": 1.0,
            "motorcycle": 0.5,
            "bus": 3.0,
            "truck": 2.5,
        }
    )
    # Number of weighted vehicle units that counts as a "full" scene (score 100).
    reference_capacity: float = 30.0
    # Score boundaries (0-100). score < moderate -> Low, < high -> Moderate, etc.
    moderate_threshold: float = 25.0
    high_threshold: float = 50.0
    severe_threshold: float = 75.0


@dataclass
class FusionSettings:
    """Weights and thresholds for the Adaptive Cross-Signal Fusion Engine."""

    congestion_weight: float = 0.45
    incident_weight: float = 0.55
    severity_points: dict[str, float] = field(
        default_factory=lambda: {"Low": 25.0, "Medium": 50.0, "High": 75.0, "Critical": 100.0}
    )
    status_factor: dict[str, float] = field(
        default_factory=lambda: {"Open": 1.0, "In Progress": 0.7, "Resolved": 0.0}
    )
    # Recency: full weight for the first `recency_full_hours`, then a linear
    # decay down to `recency_floor` at `recency_decay_hours`.
    recency_full_hours: float = 1.0
    recency_decay_hours: float = 24.0
    recency_floor: float = 0.5
    # Priority boundaries (0-100).
    medium_threshold: float = 30.0
    high_threshold: float = 55.0
    critical_threshold: float = 75.0
    # An active (not resolved) Critical incident never scores below this.
    critical_incident_floor: float = 75.0


@dataclass
class Settings:
    db_path: Path
    upload_dir: Path
    output_dir: Path
    models_dir: Path
    yolo_model: str
    default_confidence: float
    max_image_mb: float
    max_image_side: int
    max_video_mb: float
    max_video_seconds: float
    max_video_frames: int
    use_s3: bool
    s3_bucket: str
    s3_prefix: str
    aws_region: str
    admin_pin: str
    log_level: str
    congestion: CongestionSettings
    fusion: FusionSettings

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.yolo_model

    def ensure_dirs(self) -> None:
        for folder in (self.db_path.parent, self.upload_dir, self.output_dir, self.models_dir):
            folder.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build a Settings object from environment variables."""
    congestion = CongestionSettings(
        reference_capacity=_env_float("CONGESTION_REFERENCE_CAPACITY", 30.0),
        moderate_threshold=_env_float("CONGESTION_MODERATE_THRESHOLD", 25.0),
        high_threshold=_env_float("CONGESTION_HIGH_THRESHOLD", 50.0),
        severe_threshold=_env_float("CONGESTION_SEVERE_THRESHOLD", 75.0),
    )
    fusion = FusionSettings(
        congestion_weight=_env_float("FUSION_CONGESTION_WEIGHT", 0.45),
        incident_weight=_env_float("FUSION_INCIDENT_WEIGHT", 0.55),
        medium_threshold=_env_float("FUSION_MEDIUM_THRESHOLD", 30.0),
        high_threshold=_env_float("FUSION_HIGH_THRESHOLD", 55.0),
        critical_threshold=_env_float("FUSION_CRITICAL_THRESHOLD", 75.0),
    )
    return Settings(
        db_path=_resolve(_env_str("TRAFFIC_DB_PATH", "data/traffic.db")),
        upload_dir=_resolve(_env_str("UPLOAD_DIR", "data/uploads")),
        output_dir=_resolve(_env_str("OUTPUT_DIR", "outputs")),
        models_dir=_resolve(_env_str("MODELS_DIR", "models")),
        yolo_model=_env_str("YOLO_MODEL", "yolov8n.pt"),
        default_confidence=_env_float("DEFAULT_CONFIDENCE", 0.35),
        max_image_mb=_env_float("MAX_IMAGE_MB", 10.0),
        max_image_side=_env_int("MAX_IMAGE_SIDE", 1920),
        max_video_mb=_env_float("MAX_VIDEO_MB", 50.0),
        max_video_seconds=_env_float("MAX_VIDEO_SECONDS", 120.0),
        max_video_frames=_env_int("MAX_VIDEO_FRAMES", 30),
        use_s3=_env_bool("USE_S3", False),
        s3_bucket=_env_str("S3_BUCKET", ""),
        s3_prefix=_env_str("S3_PREFIX", "traffic-demo"),
        aws_region=_env_str("AWS_DEFAULT_REGION", "ap-south-1"),
        admin_pin=_env_str("ADMIN_PIN", ""),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        congestion=congestion,
        fusion=fusion,
    )


def load_streamlit_secrets() -> None:
    """Copy simple string values from Streamlit secrets into os.environ.

    Only keys that are NOT already set in the environment are copied.
    Safe to call when no secrets file exists.
    """
    try:
        import streamlit as st

        secrets = dict(st.secrets)
    except Exception:  # no secrets.toml, or not running under Streamlit
        return
    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)
