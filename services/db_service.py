from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SQLiteService:
    def __init__(self, db_path: Path, schema_path: Path):
        self.db_path = db_path
        self.schema_path = schema_path
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            schema_sql = self.schema_path.read_text(encoding="utf-8")
            conn.executescript(schema_sql)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def insert_violation(self, plate_number: str, image_path: str, email: str = "") -> int:
        timestamp = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO violations (plate_number, image_path, email, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (plate_number, image_path, email, timestamp),
                )
                conn.commit()
                return int(cursor.lastrowid)

    def get_violations(self, limit: int = 200) -> List[Dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, plate_number, image_path, email, timestamp
                FROM violations
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_violation_by_id(self, violation_id: int) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, plate_number, image_path, email, timestamp
                FROM violations
                WHERE id = ?
                """,
                (violation_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_violation(self, violation_id: int, plate_number: str, email: str) -> Optional[Dict]:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE violations
                    SET plate_number = ?, email = ?
                    WHERE id = ?
                    """,
                    (plate_number, email, violation_id),
                )
                conn.commit()
        return self.get_violation_by_id(violation_id)

    def delete_violation(self, violation_id: int) -> Optional[Dict]:
        existing = self.get_violation_by_id(violation_id)
        if not existing:
            return None
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
                conn.commit()
        return existing

