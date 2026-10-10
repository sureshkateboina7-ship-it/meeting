from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from services.db_service import SQLiteService
from services.email_service import EmailService
from utils.file_utils import secure_filename_with_timestamp
from utils.image_utils import crop_image

LOGGER = logging.getLogger(__name__)


class ViolationService:
    def __init__(self, db_service: SQLiteService, email_service: EmailService, output_dir: Path, cooldown_sec: int):
        self.db_service = db_service
        self.email_service = email_service
        self.output_dir = output_dir
        self.cooldown_sec = cooldown_sec
        self._recent_keys: Dict[str, float] = {}

    def _violation_key(self, plate_number: str, rider_box: Tuple[int, int, int, int]) -> str:
        if plate_number and plate_number not in {"UNREADABLE", "NOTFOUND", "NUMBER PLATE NOT FOUND"}:
            return plate_number
        x1, y1, x2, y2 = rider_box
        return f"bbox_{x1//20}_{y1//20}_{x2//20}_{y2//20}"

    def _allow_record(self, plate_number: str, rider_box: Tuple[int, int, int, int]) -> bool:
        key = self._violation_key(plate_number, rider_box)
        now = time.time()
        last_seen = self._recent_keys.get(key, 0)
        if now - last_seen < self.cooldown_sec:
            return False
        self._recent_keys[key] = now
        return True

    def _build_violation_crop(self, frame: np.ndarray, violation: Dict) -> np.ndarray:
        boxes = [
            box
            for box in (
                violation.get("rider_box"),
                violation.get("motorcycle_box"),
                violation.get("plate_box"),
            )
            if box
        ]
        if boxes:
            x1 = min(box[0] for box in boxes)
            y1 = min(box[1] for box in boxes)
            x2 = max(box[2] for box in boxes)
            y2 = max(box[3] for box in boxes)
            crop = crop_image(frame, (x1, y1, x2, y2))
            if crop.size > 3:
                return crop
        return frame.copy()

    def persist_violations(
        self,
        frame: np.ndarray,
        violations: List[Dict],
        email: str = "",
        source_type: str = "image",
        deduplicate: bool = True,
    ) -> List[Dict]:
        saved: List[Dict] = []
        for violation in violations:
            plate_number = violation.get("plate_number", "UNREADABLE")
            rider_box = violation.get("rider_box")
            if not rider_box:
                continue

            if deduplicate and not self._allow_record(plate_number, rider_box):
                LOGGER.info("duplicate violation skipped for plate=%s", plate_number)
                continue

            violation_crop = self._build_violation_crop(frame, violation)
            filename = secure_filename_with_timestamp(f"{source_type}_{plate_number}.jpg")
            out_path = self.output_dir / filename
            rel_path = str(Path("static") / "outputs" / filename).replace("\\", "/")
            if not cv2.imwrite(str(out_path), violation_crop):
                LOGGER.warning("cv2.imwrite failed for path=%s, falling back to full frame", out_path)
                cv2.imwrite(str(out_path), frame)

            record_id = self.db_service.insert_violation(
                plate_number=plate_number,
                image_path=rel_path,
                email=email or "",
            )

            stored = dict(violation)
            stored["id"] = record_id
            stored["image_path"] = rel_path
            saved.append(stored)
            LOGGER.info("violation stored id=%s plate=%s path=%s", record_id, plate_number, rel_path)

            if email:
                db_row = self.db_service.get_violation_by_id(record_id)
                timestamp = (db_row or {}).get("timestamp", "")
                try:
                    self.email_service.send_violation_email(
                        to_email=email,
                        plate_number=plate_number,
                        image_path=str(out_path),
                        timestamp=timestamp,
                    )
                    stored["email_status"] = "sent"
                except Exception as exc:
                    stored["email_status"] = "failed"
                    stored["email_error"] = str(exc)
                    LOGGER.warning("violation email failed id=%s plate=%s error=%s", record_id, plate_number, exc)
        return saved

    def update_violation(self, violation_id: int, plate_number: str, email: str):
        return self.db_service.update_violation(violation_id, plate_number, email)

    def delete_violation(self, violation_id: int, base_dir: Path) -> bool:
        deleted = self.db_service.delete_violation(violation_id)
        if not deleted:
            return False
        image_path = deleted.get("image_path", "")
        if image_path:
            abs_path = Path(image_path)
            if not abs_path.is_absolute():
                abs_path = base_dir / image_path
            if abs_path.exists():
                abs_path.unlink()
        LOGGER.info("violation deleted id=%s", violation_id)
        return True
