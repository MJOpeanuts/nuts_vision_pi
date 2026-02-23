#!/usr/bin/env python3
"""
sqlite_db.py – SQLite database manager for nuts_vision_pi.

Schema mirrors init.sql but uses SQLite-compatible types.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


DDL = """
CREATE TABLE IF NOT EXISTS images_input (
    image_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name  TEXT NOT NULL,
    file_path  TEXT NOT NULL,
    upload_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    format     TEXT
);

CREATE TABLE IF NOT EXISTS log_jobs (
    job_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id         INTEGER NOT NULL,
    job_name         TEXT,
    job_folder_path  TEXT,
    started_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    ended_at         TEXT,
    model            TEXT,
    FOREIGN KEY (image_id) REFERENCES images_input(image_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS detections (
    detection_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        INTEGER NOT NULL,
    class_name    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    bbox_x1       REAL NOT NULL,
    bbox_y1       REAL NOT NULL,
    bbox_x2       REAL NOT NULL,
    bbox_y2       REAL NOT NULL,
    FOREIGN KEY (job_id) REFERENCES log_jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ics_cropped (
    cropped_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id            INTEGER NOT NULL,
    detection_id      INTEGER NOT NULL,
    cropped_file_path TEXT NOT NULL,
    created_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    FOREIGN KEY (job_id)       REFERENCES log_jobs(job_id)     ON DELETE CASCADE,
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_log_jobs_image_id       ON log_jobs(image_id);
CREATE INDEX IF NOT EXISTS idx_detections_job_id       ON detections(job_id);
CREATE INDEX IF NOT EXISTS idx_ics_cropped_job_id      ON ics_cropped(job_id);
CREATE INDEX IF NOT EXISTS idx_ics_cropped_detection_id ON ics_cropped(detection_id);
"""


class NutsDB:
    """Thin wrapper around the SQLite database."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(DDL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def log_image(self, file_name: str, file_path: str, fmt: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO images_input (file_name, file_path, format) VALUES (?, ?, ?)",
            (file_name, file_path, fmt),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def start_job(
        self,
        image_id: int,
        model: str,
        job_name: str,
        job_folder_path: str,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO log_jobs (image_id, job_name, job_folder_path, model)
               VALUES (?, ?, ?, ?)""",
            (image_id, job_name, job_folder_path, model),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def end_job(self, job_id: int) -> None:
        self.conn.execute(
            "UPDATE log_jobs SET ended_at = strftime('%Y-%m-%dT%H:%M:%S', 'now') WHERE job_id = ?",
            (job_id,),
        )
        self.conn.commit()

    def log_detection(
        self,
        job_id: int,
        class_name: str,
        confidence: float,
        bbox: list,
    ) -> int:
        x1, y1, x2, y2 = bbox
        cur = self.conn.execute(
            """INSERT INTO detections (job_id, class_name, confidence,
                                       bbox_x1, bbox_y1, bbox_x2, bbox_y2)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, class_name, confidence, x1, y1, x2, y2),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def log_crop(self, job_id: int, detection_id: int, crop_path: str) -> int:
        cur = self.conn.execute(
            """INSERT INTO ics_cropped (job_id, detection_id, cropped_file_path)
               VALUES (?, ?, ?)""",
            (job_id, detection_id, crop_path),
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def list_jobs(self) -> list[sqlite3.Row]:
        """Return all jobs ordered by most recent first."""
        return self.conn.execute(
            """SELECT j.job_id, j.job_name, j.job_folder_path,
                      j.started_at, j.ended_at,
                      COUNT(d.detection_id) AS detection_count
               FROM log_jobs j
               LEFT JOIN detections d ON d.job_id = j.job_id
               GROUP BY j.job_id
               ORDER BY j.started_at DESC"""
        ).fetchall()

    def get_job(self, job_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM log_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()

    def get_detections(self, job_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM detections WHERE job_id = ? ORDER BY detection_id",
            (job_id,),
        ).fetchall()

    def get_crops(self, job_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM ics_cropped WHERE job_id = ? ORDER BY cropped_id",
            (job_id,),
        ).fetchall()
