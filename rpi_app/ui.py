#!/usr/bin/env python3
"""
ui.py – PyQt6 touch-friendly UI for nuts_vision_pi (800×480).

Pages:
  MainPage   – live camera preview + SCAN button
  HistoryPage – list of past jobs; click a row to view annotated image + crops
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QMessageBox,
    QSizePolicy,
)

from .camera import CameraManager
from .detector_onnx import OnnxDetector
from .sqlite_db import NutsDB
from .storage import create_job_dirs, get_db_path, find_external_root
from crop import ComponentCropper


# ---------------------------------------------------------------------------
# Helper – numpy BGR → QPixmap
# ---------------------------------------------------------------------------

def _bgr_to_pixmap(frame: np.ndarray) -> QPixmap:
    h, w, ch = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Background scan worker
# ---------------------------------------------------------------------------

class ScanWorker(QThread):
    """Runs detection + crop pipeline in a background thread."""

    finished = pyqtSignal(dict)   # emits result dict on success
    error = pyqtSignal(str)       # emits error message on failure

    def __init__(
        self,
        still_path: Path,
        detector: OnnxDetector,
        db: NutsDB,
        model_path: str,
    ):
        super().__init__()
        self.still_path = still_path
        self.detector = detector
        self.db = db
        self.model_path = model_path

    def run(self) -> None:  # noqa: D102
        try:
            img_path = self.still_path
            now = datetime.now()
            job_name = f"scan_{now.strftime('%Y%m%d_%H%M%S')}"
            job_dir, crops_dir = create_job_dirs(job_name)

            # Copy original still into job folder
            input_copy = job_dir / f"input{img_path.suffix}"
            shutil.copy2(str(img_path), str(input_copy))

            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                self.error.emit(f"Cannot load image: {img_path}")
                return

            # Detect
            detections = self.detector.detect(image)

            # Annotated result image
            annotated = self.detector.draw_detections(image, detections)
            result_path = job_dir / "result.jpg"
            cv2.imwrite(str(result_path), annotated)

            # Crop components
            cropper = ComponentCropper(padding=10)
            crop_paths: list[str] = []
            for i, det in enumerate(detections):
                cropped = cropper.crop_component(image, det["bbox"])
                fname = f"{i:03d}_{det['class_name']}.jpg"
                cpath = crops_dir / fname
                cv2.imwrite(str(cpath), cropped)
                crop_paths.append(str(cpath))

            # Metadata JSON
            metadata = {
                "job_name": job_name,
                "input_file": str(img_path),
                "date": now.isoformat(),
                "model": self.model_path,
                "total_detections": len(detections),
                "detections": [
                    {
                        "index": i,
                        "class_name": d["class_name"],
                        "confidence": round(d["confidence"], 4),
                        "bbox": d["bbox"],
                        "crop_file": Path(crop_paths[i]).name if i < len(crop_paths) else None,
                    }
                    for i, d in enumerate(detections)
                ],
            }
            meta_path = job_dir / "metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Database
            image_id = self.db.log_image(
                img_path.name, str(img_path), img_path.suffix.lstrip(".")
            )
            job_id = self.db.start_job(
                image_id, self.model_path, job_name, str(job_dir)
            )
            det_ids: list[int] = []
            for det in detections:
                did = self.db.log_detection(
                    job_id, det["class_name"], det["confidence"], det["bbox"]
                )
                det_ids.append(did)
            for i, cp in enumerate(crop_paths):
                if i < len(det_ids):
                    self.db.log_crop(job_id, det_ids[i], cp)
            self.db.end_job(job_id)

            self.finished.emit(
                {
                    "job_name": job_name,
                    "job_folder": str(job_dir),
                    "result_photo": str(result_path),
                    "crop_photos": crop_paths,
                    "metadata": metadata,
                }
            )

        except Exception as exc:  # pylint: disable=broad-except
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Main page – live preview + SCAN button
# ---------------------------------------------------------------------------

class MainPage(QWidget):
    switch_to_history = pyqtSignal()

    def __init__(
        self,
        camera: CameraManager,
        detector: OnnxDetector,
        db: NutsDB,
        model_path: str,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._camera = camera
        self._detector = detector
        self._db = db
        self._model_path = model_path
        self._worker: Optional[ScanWorker] = None
        self._scanning = False

        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_preview)
        self._timer.start(50)  # ~20 fps

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Preview label
        self._preview_label = QLabel("Initialisation caméra…")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(640, 400)
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._preview_label.setStyleSheet("background: black; color: white;")
        root.addWidget(self._preview_label, stretch=1)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #00cc44; font-size: 14px;")
        root.addWidget(self._status_label)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._scan_btn = QPushButton("📷  SCAN")
        self._scan_btn.setMinimumHeight(60)
        self._scan_btn.setFont(QFont("Sans", 16, QFont.Weight.Bold))
        self._scan_btn.setStyleSheet(
            "QPushButton { background:#0066cc; color:white; border-radius:8px; }"
            "QPushButton:disabled { background:#444; color:#888; }"
        )
        self._scan_btn.clicked.connect(self._on_scan)
        btn_row.addWidget(self._scan_btn, stretch=2)

        hist_btn = QPushButton("📋  Historique")
        hist_btn.setMinimumHeight(60)
        hist_btn.setFont(QFont("Sans", 14))
        hist_btn.setStyleSheet(
            "QPushButton { background:#444; color:white; border-radius:8px; }"
        )
        hist_btn.clicked.connect(self.switch_to_history.emit)
        btn_row.addWidget(hist_btn, stretch=1)

        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        if self._scanning:
            return
        frame = self._camera.get_preview_frame()
        if frame is not None:
            pix = _bgr_to_pixmap(frame)
            self._preview_label.setPixmap(
                pix.scaled(
                    self._preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _on_scan(self) -> None:
        if self._scanning:
            return
        self._scanning = True
        self._scan_btn.setEnabled(False)
        self._status_label.setText("Capture en cours…")

        # Capture full-resolution still
        data_root = find_external_root()
        tmp_still = data_root / "tmp_still.jpg"
        try:
            self._camera.capture_still(tmp_still)
        except Exception as exc:
            self._status_label.setText(f"Erreur capture: {exc}")
            self._scanning = False
            self._scan_btn.setEnabled(True)
            return

        self._status_label.setText("Détection en cours…")
        self._worker = ScanWorker(tmp_still, self._detector, self._db, self._model_path)
        self._worker.finished.connect(self._on_scan_done)
        self._worker.error.connect(self._on_scan_error)
        self._worker.start()

    def _on_scan_done(self, result: dict) -> None:
        n = result["metadata"]["total_detections"]
        self._status_label.setText(
            f"✅  {n} composant(s) détecté(s) — {result['job_name']}"
        )
        self._scanning = False
        self._scan_btn.setEnabled(True)

    def _on_scan_error(self, msg: str) -> None:
        self._status_label.setText(f"❌  Erreur: {msg}")
        self._scanning = False
        self._scan_btn.setEnabled(True)


# ---------------------------------------------------------------------------
# History page
# ---------------------------------------------------------------------------

class JobDetailPage(QWidget):
    """Shows annotated result image + crop thumbnails for one job."""

    back = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Back button + title
        top = QHBoxLayout()
        back_btn = QPushButton("← Retour")
        back_btn.setMinimumHeight(44)
        back_btn.setStyleSheet("QPushButton { background:#555; color:white; border-radius:6px; }")
        back_btn.clicked.connect(self.back.emit)
        top.addWidget(back_btn)

        self._title_label = QLabel("")
        self._title_label.setFont(QFont("Sans", 13, QFont.Weight.Bold))
        self._title_label.setStyleSheet("color: white;")
        top.addWidget(self._title_label, stretch=1)
        root.addLayout(top)

        # Result image
        self._result_label = QLabel()
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_label.setStyleSheet("background: black;")
        self._result_label.setMinimumHeight(280)
        root.addWidget(self._result_label, stretch=1)

        # Crops scroll area
        crop_label = QLabel("Crops :")
        crop_label.setStyleSheet("color: #aaa;")
        root.addWidget(crop_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(110)
        scroll.setStyleSheet("background: #111; border: none;")

        self._crops_container = QWidget()
        self._crops_layout = QHBoxLayout(self._crops_container)
        self._crops_layout.setSpacing(6)
        self._crops_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self._crops_container)
        root.addWidget(scroll)

    def load_job(self, job_folder: str, job_name: str) -> None:
        """Populate the page with data from *job_folder*."""
        self._title_label.setText(job_name)
        job_dir = Path(job_folder)

        # Result image
        result_path = job_dir / "result.jpg"
        if result_path.exists():
            pix = QPixmap(str(result_path))
            self._result_label.setPixmap(
                pix.scaled(
                    QSize(760, 320),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            self._result_label.setText("Image non disponible")

        # Clear old crops
        while self._crops_layout.count():
            item = self._crops_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Load crops
        crops_dir = job_dir / "crops"
        if crops_dir.exists():
            for img_file in sorted(crops_dir.iterdir()):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    lbl = QLabel()
                    pix = QPixmap(str(img_file))
                    lbl.setPixmap(
                        pix.scaled(
                            QSize(90, 90),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                    lbl.setToolTip(img_file.name)
                    self._crops_layout.addWidget(lbl)
        self._crops_layout.addStretch()


class HistoryPage(QWidget):
    """Lists all jobs from the database; lets user view details."""

    switch_to_main = pyqtSignal()

    def __init__(self, db: NutsDB, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._db = db
        self._stacked = QStackedWidget()
        self._list_page = self._build_list_page()
        self._detail_page = JobDetailPage()
        self._detail_page.back.connect(lambda: self._stacked.setCurrentIndex(0))
        self._stacked.addWidget(self._list_page)
        self._stacked.addWidget(self._detail_page)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._stacked)

    def _build_list_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)

        # Header
        top = QHBoxLayout()
        back_btn = QPushButton("← Caméra")
        back_btn.setMinimumHeight(44)
        back_btn.setStyleSheet("QPushButton { background:#555; color:white; border-radius:6px; }")
        back_btn.clicked.connect(self.switch_to_main.emit)
        top.addWidget(back_btn)

        title = QLabel("Historique des scans")
        title.setFont(QFont("Sans", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        top.addWidget(title, stretch=1)

        refresh_btn = QPushButton("↻")
        refresh_btn.setMinimumHeight(44)
        refresh_btn.setStyleSheet("QPushButton { background:#555; color:white; border-radius:6px; }")
        refresh_btn.clicked.connect(self.refresh)
        top.addWidget(refresh_btn)
        lay.addLayout(top)

        # Job list
        self._job_list = QListWidget()
        self._job_list.setStyleSheet(
            "QListWidget { background:#111; color:white; font-size:14px; }"
            "QListWidget::item:selected { background:#0066cc; }"
        )
        self._job_list.itemDoubleClicked.connect(self._on_item_activated)
        lay.addWidget(self._job_list, stretch=1)

        hint = QLabel("Double-tap pour voir les détails")
        hint.setStyleSheet("color: #555; font-size: 11px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(hint)

        return page

    def refresh(self) -> None:
        """Reload job list from DB."""
        self._job_list.clear()
        try:
            rows = self._db.list_jobs()
        except Exception:
            rows = []
        for row in rows:
            n_det = row["detection_count"]
            text = f"{row['job_name']}  [{n_det} détection(s)]  {row['started_at']}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, dict(row))
            self._job_list.addItem(item)

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        folder = data.get("job_folder_path", "")
        name = data.get("job_name", "")
        if folder and Path(folder).exists():
            self._detail_page.load_job(folder, name)
            self._stacked.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, "Dossier introuvable", f"Le dossier du job est introuvable:\n{folder}")

    def showEvent(self, event) -> None:  # noqa: N802
        """Refresh list every time this page becomes visible."""
        super().showEvent(event)
        self.refresh()


# ---------------------------------------------------------------------------
# Root widget
# ---------------------------------------------------------------------------

class NutsVisionApp(QWidget):
    """Top-level 800×480 full-screen application widget."""

    def __init__(
        self,
        camera: CameraManager,
        detector: OnnxDetector,
        db: NutsDB,
        model_path: str,
    ):
        super().__init__()
        self.setWindowTitle("nuts_vision_pi")
        self.setStyleSheet("background: #1a1a1a;")

        self._stacked = QStackedWidget()

        self._main_page = MainPage(camera, detector, db, model_path)
        self._history_page = HistoryPage(db)

        self._main_page.switch_to_history.connect(self._show_history)
        self._history_page.switch_to_main.connect(self._show_main)

        self._stacked.addWidget(self._main_page)
        self._stacked.addWidget(self._history_page)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._stacked)

    def _show_history(self) -> None:
        self._stacked.setCurrentIndex(1)

    def _show_main(self) -> None:
        self._stacked.setCurrentIndex(0)
