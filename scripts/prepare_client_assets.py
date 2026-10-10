from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))


REQUIRED_MODELS = {
    "helmet_model.pt": "helmet detection model",
    "plate_model.pt": "license plate detection model",
    "traffic_model.pt": "traffic/rider detection model",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_python() -> None:
    if sys.version_info[:2] != (3, 9):
        raise RuntimeError(
            f"Python 3.9 is required. Current Python is {sys.version.split()[0]}."
        )


def _ensure_dirs() -> None:
    for relative in (
        ".yolo_config",
        "static/uploads",
        "static/outputs",
        "database",
    ):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)


def _verify_models() -> None:
    models_dir = ROOT / "models"
    missing = []
    for filename, label in REQUIRED_MODELS.items():
        path = models_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(f"- {filename} ({label})")
            continue
        print(f"OK model: {filename} size={path.stat().st_size} sha256={_sha256(path)[:12]}")

    if missing:
        joined = "\n".join(missing)
        raise RuntimeError(
            "Required local YOLO model files are missing from the package:\n"
            f"{joined}\n"
            "Use the original client zip that includes the models folder."
        )


def _initialize_database() -> None:
    from config import Config
    from services.db_service import SQLiteService

    cfg = Config()
    SQLiteService(cfg.database_path, cfg.schema_path).initialize()
    print(f"OK database: {cfg.database_path}")


def _warm_paddleocr() -> None:
    from services.ocr_service import OCRService

    service = OCRService(languages="en", gpu=False)
    blank = np.zeros((48, 160, 3), dtype=np.uint8)
    text, confidence = service.read_plate(blank)
    print(f"OK PaddleOCR warmup: text={text} confidence={confidence}")


def _verify_yolo_loads() -> None:
    from config import Config
    from services.model_service import ModelService

    cfg = Config()
    service = ModelService(
        cfg.helmet_model_path,
        cfg.plate_model_path,
        cfg.traffic_model_path,
        cfg.plate_model_url,
    )
    service.get_models()
    print("OK YOLO models loaded")


def main() -> None:
    os.chdir(ROOT)
    _verify_python()
    _ensure_dirs()
    _verify_models()
    _initialize_database()
    _warm_paddleocr()
    _verify_yolo_loads()
    print("Client assets are ready.")


if __name__ == "__main__":
    main()
