#!/usr/bin/env python3
"""
camera.py – Picamera2 wrapper for nuts_vision_pi.

Provides:
  - get_preview_frame()  → numpy BGR frame (for Qt display)
  - capture_still()      → save full-resolution JPEG and return path
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# picamera2 is only available on Raspberry Pi OS.
# We guard the import so the rest of the app can be imported on other
# platforms (e.g. for unit tests).
try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    PICAMERA2_AVAILABLE = True
except ImportError:  # pragma: no cover
    PICAMERA2_AVAILABLE = False


# Preview resolution (fits in 800×480 panel alongside UI elements)
PREVIEW_SIZE = (640, 480)
# Full-resolution still capture
STILL_SIZE = (4056, 3040)  # 12 MP sensor; adjust for your module


class CameraManager:
    """Manages the CSI camera via Picamera2."""

    def __init__(
        self,
        preview_size: tuple[int, int] = PREVIEW_SIZE,
        still_size: tuple[int, int] = STILL_SIZE,
    ):
        self.preview_size = preview_size
        self.still_size = still_size
        self._cam: Optional["Picamera2"] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise and start the camera."""
        if not PICAMERA2_AVAILABLE:
            return
        self._cam = Picamera2()
        preview_cfg = self._cam.create_preview_configuration(
            main={"size": self.preview_size, "format": "BGR888"}
        )
        self._cam.configure(preview_cfg)
        self._cam.start()
        # Short warm-up so AEC/AWB can settle
        time.sleep(0.5)

    def stop(self) -> None:
        """Stop and release the camera."""
        if self._cam is not None:
            self._cam.stop()
            self._cam.close()
            self._cam = None

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """
        Return the latest preview frame as a BGR numpy array, or None if
        Picamera2 is not available.
        """
        if self._cam is None:
            return None
        return self._cam.capture_array()

    def capture_still(self, output_path: Path) -> Path:
        """
        Capture a full-resolution JPEG still.

        Temporarily switches to the still configuration, captures, then
        returns to preview.

        Args:
            output_path: Where to save the JPEG file.

        Returns:
            The output_path (same value, for convenience).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not PICAMERA2_AVAILABLE or self._cam is None:
            # Create a placeholder frame for testing without hardware
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(output_path), placeholder)
            return output_path

        # Switch to still config
        self._cam.stop()
        still_cfg = self._cam.create_still_configuration(
            main={"size": self.still_size}
        )
        self._cam.configure(still_cfg)
        self._cam.start()
        time.sleep(0.2)  # let AEC settle again

        self._cam.capture_file(str(output_path))

        # Return to preview config
        self._cam.stop()
        preview_cfg = self._cam.create_preview_configuration(
            main={"size": self.preview_size, "format": "BGR888"}
        )
        self._cam.configure(preview_cfg)
        self._cam.start()

        return output_path
