"""
detector.py - Vehicle detection with a pretrained Ultralytics YOLO model.

The model is pretrained on the COCO dataset. We keep only the vehicle
classes car, motorcycle, bus and truck. Every number shown in the app comes
from the real model output - nothing is invented.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# BGR colours for drawing each class
CLASS_COLORS = {
    "car": (255, 140, 0),
    "motorcycle": (0, 200, 255),
    "bus": (60, 200, 60),
    "truck": (200, 60, 200),
}


class ValidationError(ValueError):
    """Bad upload (wrong type, too large, unreadable)."""


class ModelLoadError(RuntimeError):
    """YOLO weights could not be loaded or downloaded."""


class DetectionError(RuntimeError):
    """Inference failed."""


@dataclass
class DetectionResult:
    detections: list[dict[str, Any]]
    counts: dict[str, int]
    annotated_rgb: np.ndarray
    image_size: tuple[int, int]  # (width, height)
    confidence_threshold: float
    model_name: str
    per_class_confidence: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------
def validate_image_upload(file_name: str, data: bytes, max_mb: float) -> Image.Image:
    """Check extension, size and that the bytes are a real image. Returns RGB image."""
    extension = Path(file_name or "").suffix.lower()
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValidationError("Unsupported file type. Please upload a JPG, JPEG or PNG image.")
    if not data:
        raise ValidationError("The uploaded file is empty.")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(f"Image is {size_mb:.1f} MB; the limit is {max_mb:g} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()  # detects truncated / fake files
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValidationError("The file could not be read as an image.") from exc
    if image.format not in {"JPEG", "PNG"}:
        raise ValidationError("The file content is not a JPEG or PNG image.")
    return image.convert("RGB")


def limit_image_size(image: Image.Image, max_side: int) -> Image.Image:
    """Shrink very large images to keep memory and inference time reasonable."""
    if max(image.size) <= max_side:
        return image
    image = image.copy()
    image.thumbnail((max_side, max_side))
    return image


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
class VehicleDetector:
    """Wraps a YOLO model. Weights are loaded lazily on first use."""

    def __init__(self, model_path: Path | str, model=None):
        self.model_path = Path(model_path)
        self._model = model  # tests can inject a fake model

    @property
    def model_name(self) -> str:
        return self.model_path.name

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "The 'ultralytics' package is not installed. Run: pip install -r requirements.txt"
            ) from exc
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            # If the file is missing, Ultralytics downloads official weights to this path.
            self._model = YOLO(str(self.model_path))
        except Exception as exc:  # network errors, corrupt weights, unknown names
            logger.exception("Model load failed")
            raise ModelLoadError(
                f"Could not load YOLO weights '{self.model_path.name}'. Check your internet "
                "connection for the first download, or place the .pt file in the models/ folder."
            ) from exc
        return self._model

    def detect(self, image_rgb: np.ndarray, confidence: float = 0.35) -> DetectionResult:
        if not 0.05 <= confidence <= 0.95:
            raise ValueError("Confidence threshold must be between 0.05 and 0.95")
        if image_rgb is None or image_rgb.ndim != 3:
            raise ValidationError("Expected a colour image")
        model = self.load()
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        try:
            results = model.predict(
                image_bgr, conf=confidence, classes=list(VEHICLE_CLASSES), verbose=False
            )
        except Exception as exc:
            logger.exception("Inference failed")
            raise DetectionError(f"Inference failed: {exc.__class__.__name__}") from exc

        detections = parse_results(results)
        counts = {name: 0 for name in VEHICLE_CLASSES.values()}
        best: dict[str, float] = {}
        for det in detections:
            counts[det["class_name"]] += 1
            best[det["class_name"]] = max(best.get(det["class_name"], 0.0), det["confidence"])
        annotated = draw_detections(image_bgr, detections)
        height, width = image_rgb.shape[:2]
        return DetectionResult(
            detections=detections,
            counts=counts,
            annotated_rgb=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            image_size=(width, height),
            confidence_threshold=confidence,
            model_name=self.model_name,
            per_class_confidence=best,
        )


def parse_results(results) -> list[dict[str, Any]]:
    """Convert Ultralytics results into plain dictionaries (vehicle classes only)."""
    detections: list[dict[str, Any]] = []
    if not results:
        return detections
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections
    xyxy = _to_numpy(boxes.xyxy)
    confs = _to_numpy(boxes.conf)
    classes = _to_numpy(boxes.cls).astype(int)
    for box, conf, cls_id in zip(xyxy, confs, classes):
        name = VEHICLE_CLASSES.get(int(cls_id))
        if name is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        detections.append(
            {"class_name": name, "confidence": round(float(conf), 3), "bbox": [x1, y1, x2, y2]}
        )
    return detections


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def draw_detections(image_bgr: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes and 'class confidence' labels on a copy of the image."""
    canvas = image_bgr.copy()
    thickness = max(1, round(max(canvas.shape[:2]) / 500))
    font_scale = max(0.4, max(canvas.shape[:2]) / 1600)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det["class_name"], (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness + 1)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(canvas, (x1, top), (x1 + tw + 4, top + th + baseline + 4), color, -1)
        cv2.putText(
            canvas, label, (x1 + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return canvas


def encode_png(image_rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def save_image(image_rgb: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)
    return path


def detections_table(result: DetectionResult) -> list[dict[str, Any]]:
    """Rows for display: one per detected vehicle."""
    return [
        {
            "#": i + 1,
            "class": d["class_name"],
            "confidence": d["confidence"],
            "x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3],
        }
        for i, d in enumerate(result.detections)
    ]
