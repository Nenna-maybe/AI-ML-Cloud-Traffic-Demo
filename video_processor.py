"""
video_processor.py - Frame-sampled vehicle analysis for short videos.

How it works
------------
* The uploaded video is written to a temporary file (never fully loaded into
  memory) and deleted afterwards.
* Up to `max_frames` frames are picked evenly across the video. Each frame
  is read one at a time and passed to the YOLO detector.
* We report the vehicle count per sampled frame, the average and the peak.

Limitations (also shown in the app)
-----------------------------------
* Frame sampling can miss short events between sampled frames.
* There is no object tracking: the same vehicle visible in several frames
  is counted again in each frame. So the numbers describe "vehicles visible
  per frame", NOT the number of unique vehicles that passed.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from detector import DetectionResult, ValidationError, VehicleDetector

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class FrameSummary:
    frame_index: int
    timestamp_s: float
    total: int
    counts: dict[str, int]


@dataclass
class VideoAnalysisResult:
    fps: float
    total_frames: int
    duration_s: float
    frames: list[FrameSummary]
    average_counts: dict[str, float]
    average_total: float
    peak_total: int
    peak_frame_index: int
    sample_frames: list[tuple[str, np.ndarray]] = field(default_factory=list)

    @property
    def frames_processed(self) -> int:
        return len(self.frames)


def validate_video_upload(file_name: str, size_bytes: int, max_mb: float) -> None:
    extension = Path(file_name or "").suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValidationError("Unsupported video type. Please upload MP4, MOV, AVI or MKV.")
    if size_bytes <= 0:
        raise ValidationError("The uploaded video is empty.")
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_mb:
        raise ValidationError(f"Video is {size_mb:.1f} MB; the limit is {max_mb:g} MB.")


def write_temp_video(file_obj, suffix: str, temp_dir: Optional[Path] = None) -> Path:
    """Stream an uploaded file object to a temp file in 1 MB chunks."""
    if temp_dir:
        temp_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=suffix, dir=str(temp_dir) if temp_dir else None)
    with os.fdopen(fd, "wb") as out:
        file_obj.seek(0)
        while chunk := file_obj.read(1024 * 1024):
            out.write(chunk)
    return Path(name)


def probe_video(path: Path) -> tuple[float, int, float]:
    """Return (fps, frame_count, duration_seconds). Raises ValidationError if unreadable."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValidationError("The video could not be opened. It may be corrupted or use an unsupported codec.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValidationError("The video has no readable frames.")
    return fps, frame_count, frame_count / fps


def choose_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    """Evenly spaced frame numbers, always including the first frame."""
    if frame_count <= 0 or max_frames <= 0:
        return []
    n = min(frame_count, max_frames)
    if n == 1:
        return [0]
    step = (frame_count - 1) / (n - 1)
    return sorted({int(round(i * step)) for i in range(n)})


def analyze_video(
    path: Path,
    detector: VehicleDetector,
    *,
    confidence: float,
    max_frames: int,
    max_seconds: float,
    progress: Optional[Callable[[int, int], None]] = None,
    keep_samples: int = 3,
) -> VideoAnalysisResult:
    fps, frame_count, duration = probe_video(path)
    if duration > max_seconds:
        raise ValidationError(
            f"Video is {duration:.0f} s long; the limit is {max_seconds:g} s. Please trim it."
        )
    indices = choose_frame_indices(frame_count, max_frames)
    frames: list[FrameSummary] = []
    samples: list[tuple[str, np.ndarray]] = []
    peak: Optional[tuple[int, DetectionResult]] = None

    cap = cv2.VideoCapture(str(path))
    try:
        for position, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                continue  # skip unreadable frames instead of failing the whole video
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(frame_rgb, confidence)
            frames.append(FrameSummary(index, round(index / fps, 2), result.total, dict(result.counts)))
            if len(samples) < keep_samples:
                samples.append((f"Frame {index} ({index / fps:.1f} s)", result.annotated_rgb))
            if peak is None or result.total > peak[1].total:
                peak = (index, result)
            if progress:
                progress(position + 1, len(indices))
    finally:
        cap.release()

    if not frames:
        raise ValidationError("No frames could be decoded from this video.")

    classes = sorted({c for f in frames for c in f.counts})
    average_counts = {c: round(sum(f.counts.get(c, 0) for f in frames) / len(frames), 2) for c in classes}
    totals = [f.total for f in frames]
    peak_index, peak_result = peak  # type: ignore[misc]
    peak_label = f"Peak frame {peak_index} ({peak_index / fps:.1f} s)"
    if all(not label.startswith(f"Frame {peak_index} ") for label, _ in samples):
        samples.append((peak_label, peak_result.annotated_rgb))

    return VideoAnalysisResult(
        fps=round(fps, 2),
        total_frames=frame_count,
        duration_s=round(duration, 2),
        frames=frames,
        average_counts=average_counts,
        average_total=round(sum(totals) / len(totals), 2),
        peak_total=max(totals),
        peak_frame_index=peak_index,
        sample_frames=samples,
    )
