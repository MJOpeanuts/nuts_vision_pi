#!/usr/bin/env python3
"""
main.py – Entry point for nuts_vision_pi.

Usage:
    python3 rpi_app/main.py [--model PATH] [--conf FLOAT]
"""

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from .camera import CameraManager
from .detector_onnx import OnnxDetector
from .sqlite_db import NutsDB
from .storage import get_db_path, find_external_root
from .ui import NutsVisionApp


# Default model path: best.onnx sits at the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = str(_REPO_ROOT / "best.onnx")

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nuts_vision_pi – embedded IC detector")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to ONNX model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # -- Storage / DB --
    db_path = get_db_path()
    db = NutsDB(db_path)
    db.connect()

    # -- Camera --
    camera = CameraManager()
    camera.start()

    # -- Detector --
    model_path = args.model
    detector = OnnxDetector(model_path, conf_threshold=args.conf)

    # -- UI --
    window = NutsVisionApp(camera, detector, db, model_path)
    window.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    window.showFullScreen()

    def on_exit() -> None:
        camera.stop()
        db.close()

    app.aboutToQuit.connect(on_exit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
