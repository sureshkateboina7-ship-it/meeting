import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg"}


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def secure_filename_with_timestamp(filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem).strip("_") or "file"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_stem}_{ts}_{uuid.uuid4().hex[:8]}{ext.lower()}"


def is_allowed_image(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def is_allowed_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def normalize_plate_text(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    return cleaned

