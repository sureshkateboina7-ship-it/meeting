import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    base_dir: Path = Path(__file__).resolve().parent
    upload_dir: Path = base_dir / "static" / "uploads"
    output_dir: Path = base_dir / "static" / "outputs"
    database_path: Path = base_dir / "database" / "violations.db"
    schema_path: Path = base_dir / "database" / "schema.sql"

    helmet_model_path: Path = base_dir / "models" / "helmet_model.pt"
    plate_model_path: Path = base_dir / "models" / "plate_model.pt"
    traffic_model_path: Path = base_dir / "models" / "traffic_model.pt"
    plate_model_url: str = os.getenv(
        "PLATE_MODEL_URL",
        "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt",
    )
    inference_size: int = int(os.getenv("INFERENCE_SIZE", "640"))
    alpr_engine: str = os.getenv("ALPR_ENGINE", "auto").lower()
    fast_alpr_detector_model: str = os.getenv("FAST_ALPR_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end")
    fast_alpr_ocr_model: str = os.getenv("FAST_ALPR_OCR_MODEL", "cct-s-v2-global-model")

    image_confidence: float = float(os.getenv("IMAGE_CONFIDENCE", "0.40"))
    plate_confidence: float = float(os.getenv("PLATE_CONFIDENCE", "0.35"))
    live_confidence: float = float(os.getenv("LIVE_CONFIDENCE", "0.45"))
    frame_skip: int = max(0, int(os.getenv("FRAME_SKIP", "2")))
    duplicate_cooldown_sec: int = max(1, int(os.getenv("DUPLICATE_COOLDOWN_SEC", "5")))

    rider_labels: str = os.getenv("RIDER_LABELS", "rider,person,motorcyclist,biker")
    motorcycle_labels: str = os.getenv("MOTORCYCLE_LABELS", "motorcycle,bike,motorbike,scooter")
    no_helmet_labels: str = os.getenv(
        "NO_HELMET_LABELS",
        "no_helmet,without_helmet,withouthelmet,nohelmet,unhelmeted,no-hardhat,nohardhat",
    )
    helmet_labels: str = os.getenv("HELMET_LABELS", "helmet,with_helmet,hardhat")

    ocr_languages: str = os.getenv("OCR_LANGUAGES", "en")
    ocr_gpu: bool = os.getenv("OCR_GPU", "false").lower() == "true"

    smtp_server: str = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    sender_email: str = os.getenv("SENDER_EMAIL", "")
    smtp_use_tls: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    auto_email_recipient: str = os.getenv("AUTO_EMAIL_RECIPIENT", "")

    max_content_length: int = int(os.getenv("MAX_CONTENT_LENGTH", str(500 * 1024 * 1024)))
