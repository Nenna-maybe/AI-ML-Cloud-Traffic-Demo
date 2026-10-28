"""Detector tests use a fake YOLO model (no weight download needed).

For a REAL inference check, run:  python manual_inference_test.py
"""

import io

import numpy as np
import pytest
from PIL import Image

from detector import (
    DetectionError, ModelLoadError, ValidationError, VehicleDetector,
    detections_table, limit_image_size, validate_image_upload,
)
from tests.conftest import FakeYOLO


def image_bytes(fmt="PNG", size=(64, 48)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buffer, format=fmt)
    return buffer.getvalue()


def blank(h=100, w=160):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_counts_only_vehicle_classes_above_threshold(fake_yolo):
    result = VehicleDetector("unused.pt", model=fake_yolo).detect(blank(), confidence=0.35)
    assert result.counts == {"car": 2, "motorcycle": 1, "bus": 1, "truck": 0}
    assert result.total == 4
    assert all(d["class_name"] != "person" for d in result.detections)
    assert fake_yolo.calls[0]["classes"] == [2, 3, 5, 7]


def test_lower_threshold_includes_more(fake_yolo):
    result = VehicleDetector("unused.pt", model=fake_yolo).detect(blank(), confidence=0.15)
    assert result.counts["truck"] == 1


def test_annotated_image_has_boxes(fake_yolo):
    image = blank()
    result = VehicleDetector("unused.pt", model=fake_yolo).detect(image, 0.35)
    assert result.annotated_rgb.shape == image.shape
    assert result.annotated_rgb.any()  # boxes were drawn on the black image
    assert not image.any()  # original is not modified
    assert detections_table(result)[0]["confidence"] == 0.9


def test_no_detections():
    result = VehicleDetector("unused.pt", model=FakeYOLO([])).detect(blank(), 0.35)
    assert result.total == 0 and result.detections == []


def test_inference_error_is_wrapped():
    with pytest.raises(DetectionError):
        VehicleDetector("unused.pt", model=FakeYOLO(fail=True)).detect(blank(), 0.35)


@pytest.mark.parametrize("conf", [0.0, 0.01, 0.99, 1.5])
def test_invalid_confidence(conf, fake_yolo):
    with pytest.raises(ValueError):
        VehicleDetector("unused.pt", model=fake_yolo).detect(blank(), conf)


def test_missing_weights_give_friendly_error(tmp_path, monkeypatch):
    import ultralytics

    def broken(*args, **kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(ultralytics, "YOLO", broken)
    with pytest.raises(ModelLoadError, match="Could not load YOLO weights"):
        VehicleDetector(tmp_path / "models" / "yolov8n.pt").load()


def test_validate_good_png_and_jpeg():
    assert validate_image_upload("a.png", image_bytes("PNG"), 5).mode == "RGB"
    assert validate_image_upload("a.JPG", image_bytes("JPEG"), 5).size == (64, 48)


@pytest.mark.parametrize("name, data, message", [
    ("a.gif", image_bytes(), "Unsupported"),
    ("a.png", b"", "empty"),
    ("a.png", b"not an image at all", "could not be read"),
    ("a.jpg", image_bytes("BMP"), "not a JPEG or PNG"),
])
def test_validate_rejects_bad_files(name, data, message):
    with pytest.raises(ValidationError, match=message):
        validate_image_upload(name, data, 5)


def test_validate_rejects_large_files():
    with pytest.raises(ValidationError, match="limit"):
        validate_image_upload("a.png", image_bytes(size=(64, 48)), max_mb=0.0001)


def test_limit_image_size():
    big = Image.new("RGB", (4000, 2000))
    assert max(limit_image_size(big, 1920).size) == 1920
    small = Image.new("RGB", (640, 480))
    assert limit_image_size(small, 1920) is small
