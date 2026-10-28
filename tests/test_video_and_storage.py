import io

import cv2
import numpy as np
import pytest

from cloud_storage import StorageManager
from detector import ValidationError, VehicleDetector
from tests.conftest import FakeYOLO
from video_processor import (
    analyze_video, choose_frame_indices, probe_video, validate_video_upload, write_temp_video,
)


def make_video(path, frames=30, fps=10, size=(160, 120)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i * 5 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------- video ---------------------------------------
def test_frame_index_sampling():
    assert choose_frame_indices(100, 5) == [0, 25, 50, 74, 99]
    assert choose_frame_indices(3, 10) == [0, 1, 2]
    assert choose_frame_indices(50, 1) == [0]
    assert choose_frame_indices(0, 5) == []


def test_validate_video_upload():
    validate_video_upload("clip.MP4", 1024, 10)
    with pytest.raises(ValidationError):
        validate_video_upload("clip.gif", 1024, 10)
    with pytest.raises(ValidationError):
        validate_video_upload("clip.mp4", 0, 10)
    with pytest.raises(ValidationError):
        validate_video_upload("clip.mp4", 20 * 1024 * 1024, 10)


def test_analyze_video_with_fake_model(tmp_path, fake_yolo):
    video = make_video(tmp_path / "clip.mp4")
    fps, count, duration = probe_video(video)
    assert count == 30 and fps == pytest.approx(10)
    calls = []
    result = analyze_video(video, VehicleDetector("x.pt", model=fake_yolo), confidence=0.35,
                           max_frames=5, max_seconds=60, progress=lambda d, t: calls.append((d, t)))
    assert result.frames_processed == 5
    assert result.average_total == 4 and result.peak_total == 4
    assert result.average_counts["car"] == 2
    assert calls[-1] == (5, 5)
    assert 1 <= len(result.sample_frames) <= 4


def test_video_too_long_rejected(tmp_path):
    video = make_video(tmp_path / "clip.mp4", frames=30, fps=10)  # 3 seconds
    with pytest.raises(ValidationError, match="limit"):
        analyze_video(video, VehicleDetector("x.pt", model=FakeYOLO()), confidence=0.35,
                      max_frames=5, max_seconds=1)


def test_unreadable_video_rejected(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(ValidationError):
        probe_video(bad)


def test_write_temp_video_streams_file(tmp_path):
    path = write_temp_video(io.BytesIO(b"abc" * 1000), ".mp4", tmp_path / "tmp")
    assert path.read_bytes() == b"abc" * 1000
    assert path.suffix == ".mp4"


# ---------------------------- storage -------------------------------------
class FakeS3:
    def __init__(self, fail=False):
        self.fail = fail
        self.uploads = []

    def upload_file(self, local, bucket, key):
        if self.fail:
            raise PermissionError("AccessDenied")
        self.uploads.append((local, bucket, key))

    def head_bucket(self, Bucket):
        if self.fail:
            raise PermissionError("AccessDenied")


def test_storage_disabled_keeps_local(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    storage = StorageManager(use_s3=False, bucket="", prefix="p", region="ap-south-1")
    stored = storage.save(f, "a.png")
    assert stored.backend == "local" and stored.location == str(f)
    assert storage.status()[0] is False
    assert storage.download(stored.location, tmp_path / "copy.png") == f


def test_storage_uploads_to_s3(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    client = FakeS3()
    storage = StorageManager(use_s3=True, bucket="my-bucket", prefix="demo", region="ap-south-1", client=client)
    stored = storage.save(f, "annotated/a.png")
    assert stored.backend == "s3"
    assert stored.location == "s3://my-bucket/demo/annotated/a.png"
    assert client.uploads == [(str(f), "my-bucket", "demo/annotated/a.png")]
    assert storage.status()[0] is True


def test_storage_falls_back_on_error(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    storage = StorageManager(use_s3=True, bucket="b", prefix="", region="r", client=FakeS3(fail=True))
    stored = storage.save(f, "a.png")
    assert stored.backend == "local"
    assert "saved locally" in stored.warning
    ok, message = storage.status()
    assert not ok and "PermissionError" in message


def test_enabled_without_bucket_is_disabled():
    assert StorageManager(use_s3=True, bucket="", prefix="", region="r").enabled is False
