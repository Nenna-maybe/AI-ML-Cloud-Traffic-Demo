"""
manual_inference_test.py - Runs the REAL YOLO model on an image.

Usage:
    python manual_inference_test.py                 # uses Ultralytics' bundled bus.jpg
    python manual_inference_test.py path/to/img.jpg

The first run downloads the YOLO weights (~6 MB) into the models/ folder.
The annotated image is written to outputs/manual_test_annotated.png.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from config import load_settings
from detector import VehicleDetector, save_image
from traffic import estimate_congestion


def main() -> int:
    settings = load_settings()
    settings.ensure_dirs()
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        from ultralytics.utils import ASSETS

        image_path = Path(ASSETS) / "bus.jpg"
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    detector = VehicleDetector(settings.model_path)
    image = np.array(Image.open(image_path).convert("RGB"))
    result = detector.detect(image, settings.default_confidence)
    congestion = estimate_congestion(result.counts, settings.congestion)

    print(f"Image      : {image_path}")
    print(f"Model      : {result.model_name}  (confidence >= {result.confidence_threshold})")
    print(f"Counts     : {result.counts}  total={result.total}")
    for det in result.detections:
        print(f"  - {det['class_name']:<10} conf={det['confidence']:.3f} bbox={det['bbox']}")
    print(f"Congestion : {congestion.score} ({congestion.category})")
    out = save_image(result.annotated_rgb, settings.output_dir / "manual_test_annotated.png")
    print(f"Annotated  : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
