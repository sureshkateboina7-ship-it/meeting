from __future__ import annotations

from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve


class ModelService:
    def __init__(self, helmet_model_path: Path, plate_model_path: Path, traffic_model_path: Path, plate_model_url: str = ""):
        self.helmet_model_path = helmet_model_path
        self.plate_model_path = plate_model_path
        self.traffic_model_path = traffic_model_path
        self.plate_model_url = plate_model_url
        self._helmet_model = None
        self._plate_model = None
        self._traffic_model = None

    def _ensure_plate_model(self) -> None:
        if self.plate_model_path.exists():
            return
        if not self.plate_model_url:
            return
        self.plate_model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(self.plate_model_url, self.plate_model_path)

    def _load_model(self, path: Path):
        if path == self.plate_model_path:
            self._ensure_plate_model()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics") from exc
        return YOLO(str(path))

    def get_models(self) -> Tuple[object, object, object]:
        if self._helmet_model is None:
            self._helmet_model = self._load_model(self.helmet_model_path)
        if self._plate_model is None:
            self._plate_model = self._load_model(self.plate_model_path)
        if self._traffic_model is None:
            self._traffic_model = self._load_model(self.traffic_model_path)
        return self._helmet_model, self._plate_model, self._traffic_model

    def healthcheck(self) -> dict:
        return {
            "helmet_model_exists": self.helmet_model_path.exists(),
            "plate_model_exists": self.plate_model_path.exists(),
            "traffic_model_exists": self.traffic_model_path.exists(),
            "helmet_model_path": str(self.helmet_model_path),
            "plate_model_path": str(self.plate_model_path),
            "traffic_model_path": str(self.traffic_model_path),
        }
