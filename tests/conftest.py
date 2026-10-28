"""Shared pytest fixtures."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database as db  # noqa: E402


@pytest.fixture
def db_path(tmp_path):
    """A fresh, empty database for every test."""
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


class FakeBoxes:
    """Mimics ultralytics Results.boxes (xyxy, conf, cls arrays)."""

    def __init__(self, rows):
        arr = np.array(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = arr[:, :4], arr[:, 4], arr[:, 5]

    def __len__(self):
        return len(self.conf)


class FakeYOLO:
    """Stand-in for a YOLO model so tests never download weights.

    rows: list of [x1, y1, x2, y2, confidence, class_id]
    """

    def __init__(self, rows=None, fail=False):
        self.rows = rows or []
        self.fail = fail
        self.calls = []

    def predict(self, image, conf, classes, verbose):
        self.calls.append({"conf": conf, "classes": classes})
        if self.fail:
            raise RuntimeError("GPU exploded")
        rows = [r for r in self.rows if r[4] >= conf and int(r[5]) in classes]
        return [SimpleNamespace(boxes=FakeBoxes(rows))]


@pytest.fixture
def fake_yolo():
    return FakeYOLO(
        rows=[
            [10, 10, 60, 50, 0.90, 2],   # car
            [70, 10, 120, 50, 0.80, 2],  # car
            [10, 60, 40, 90, 0.55, 3],   # motorcycle
            [50, 55, 150, 95, 0.70, 5],  # bus
            [5, 5, 20, 20, 0.95, 0],     # person - must be ignored
            [100, 60, 140, 90, 0.20, 7], # truck below default threshold
        ]
    )
