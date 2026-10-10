from __future__ import annotations

from flask import Flask

from config import Config
from services.db_service import SQLiteService
from services.detection_service import DetectionLabelConfig, DetectionService
from services.email_service import EmailService
from services.alpr_service import FastALPRService
from services.live_service import LiveDetectionService
from services.model_service import ModelService
from services.ocr_service import OCRService
from services.video_service import VideoProcessingService
from services.violation_service import ViolationService
from utils.file_utils import ensure_directories


def build_services(app: Flask, cfg: Config) -> None:
    ensure_directories([cfg.upload_dir, cfg.output_dir, cfg.database_path.parent])

    db_service = SQLiteService(cfg.database_path, cfg.schema_path)
    db_service.initialize()

    model_service = ModelService(cfg.helmet_model_path, cfg.plate_model_path, cfg.traffic_model_path, cfg.plate_model_url)
    ocr_service = OCRService(cfg.ocr_languages, cfg.ocr_gpu)
    alpr_service = FastALPRService(cfg.fast_alpr_detector_model, cfg.fast_alpr_ocr_model)
    label_config = DetectionLabelConfig(
        rider_keywords=[x.strip() for x in cfg.rider_labels.split(",") if x.strip()],
        motorcycle_keywords=[x.strip() for x in cfg.motorcycle_labels.split(",") if x.strip()],
        no_helmet_keywords=[x.strip() for x in cfg.no_helmet_labels.split(",") if x.strip()],
        helmet_keywords=[x.strip() for x in cfg.helmet_labels.split(",") if x.strip()],
    )
    detection_service = DetectionService(
        model_service,
        ocr_service,
        label_config,
        inference_size=cfg.inference_size,
        alpr_service=alpr_service,
        alpr_engine=cfg.alpr_engine,
    )

    email_service = EmailService(
        smtp_server=cfg.smtp_server,
        smtp_port=cfg.smtp_port,
        username=cfg.smtp_username,
        password=cfg.smtp_password,
        sender_email=cfg.sender_email,
        use_tls=cfg.smtp_use_tls,
    )
    violation_service = ViolationService(
        db_service=db_service,
        email_service=email_service,
        output_dir=cfg.output_dir,
        cooldown_sec=cfg.duplicate_cooldown_sec,
    )
    video_service = VideoProcessingService(
        detection_service=detection_service,
        violation_service=violation_service,
        frame_skip=cfg.frame_skip,
    )
    live_service = LiveDetectionService(
        detection_service=detection_service,
        violation_service=violation_service,
        frame_skip=cfg.frame_skip,
    )

    app.extensions["services"] = {
        "db_service": db_service,
        "model_service": model_service,
        "detection_service": detection_service,
        "ocr_service": ocr_service,
        "alpr_service": alpr_service,
        "email_service": email_service,
        "violation_service": violation_service,
        "video_service": video_service,
        "live_service": live_service,
    }
