from __future__ import annotations

import logging
from pathlib import Path

import cv2
from flask import Blueprint, Response, current_app, jsonify, request

from config import Config
from utils.file_utils import is_allowed_image, is_allowed_video, normalize_plate_text, secure_filename_with_timestamp

LOGGER = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)


def _services():
    return current_app.extensions["services"]


def _with_image_status(rows, base_dir: Path):
    enriched = []
    for row in rows:
        image_path = row.get("image_path", "")
        abs_path = Path(image_path)
        if not abs_path.is_absolute():
            abs_path = base_dir / image_path
        row = dict(row)
        row["image_exists"] = abs_path.exists()
        enriched.append(row)
    return enriched


@api_bp.post("/upload-image")
def upload_image():
    cfg: Config = current_app.config["APP_CONFIG"]
    file = request.files.get("file")
    email = request.form.get("email", "").strip() or cfg.auto_email_recipient
    if not file or not file.filename:
        return jsonify({"error": "Image file is required."}), 400
    if not is_allowed_image(file.filename):
        return jsonify({"error": "Unsupported image format."}), 400

    saved_name = secure_filename_with_timestamp(file.filename)
    upload_path = cfg.upload_dir / saved_name
    file.save(upload_path)

    image = cv2.imread(str(upload_path))
    if image is None:
        return jsonify({"error": "Uploaded image could not be decoded."}), 400

    LOGGER.info("image upload started file=%s", saved_name)
    try:
        annotated, violations = _services()["detection_service"].process_frame(
            frame=image,
            helmet_confidence=cfg.image_confidence,
            plate_confidence=cfg.plate_confidence,
        )
    except Exception as exc:
        LOGGER.exception("image processing failed")
        return jsonify({"error": str(exc)}), 500

    output_name = secure_filename_with_timestamp(f"processed_{file.filename}")
    output_path = cfg.output_dir / output_name
    output_rel = str(Path("static") / "outputs" / output_name).replace("\\", "/")
    cv2.imwrite(str(output_path), annotated)

    persisted = _services()["violation_service"].persist_violations(
        frame=annotated,
        violations=violations,
        email=email,
        source_type="image",
        deduplicate=False,
    )
    LOGGER.info("image upload completed file=%s violations=%s", saved_name, len(persisted))
    return jsonify(
        {
            "message": "Image processed successfully.",
            "processed_image": output_rel,
            "violations": persisted,
            "violation_count": len(persisted),
        }
    )


@api_bp.post("/upload-video")
def upload_video():
    cfg: Config = current_app.config["APP_CONFIG"]
    file = request.files.get("file")
    email = request.form.get("email", "").strip() or cfg.auto_email_recipient
    if not file or not file.filename:
        return jsonify({"error": "Video file is required."}), 400
    if not is_allowed_video(file.filename):
        return jsonify({"error": "Unsupported video format."}), 400

    video_name = secure_filename_with_timestamp(file.filename)
    input_path = cfg.upload_dir / video_name
    file.save(input_path)

    output_name = secure_filename_with_timestamp(f"processed_{Path(file.filename).stem}.mp4")
    output_path = cfg.output_dir / output_name
    LOGGER.info("video upload started file=%s", video_name)

    try:
        stats = _services()["video_service"].process_video(
            input_path=input_path,
            output_path=output_path,
            helmet_confidence=cfg.image_confidence,
            plate_confidence=cfg.plate_confidence,
            email=email,
        )
    except Exception as exc:
        LOGGER.exception("video processing failed")
        return jsonify({"error": str(exc)}), 500

    stats["output_path"] = str(Path("static") / "outputs" / output_name).replace("\\", "/")
    LOGGER.info("video upload completed file=%s stats=%s", video_name, stats)
    return jsonify({"message": "Video processed successfully.", "stats": stats})


@api_bp.get("/live-detection")
def live_detection():
    cfg: Config = current_app.config["APP_CONFIG"]
    source = request.args.get("source", "0")
    email = request.args.get("email", "").strip() or cfg.auto_email_recipient
    LOGGER.info("live detection started source=%s", source)
    try:
        stream = _services()["live_service"].stream(
            source=source,
            helmet_confidence=cfg.live_confidence,
            plate_confidence=cfg.plate_confidence,
            email=email,
        )
    except Exception as exc:
        LOGGER.exception("live detection failed to start")
        return jsonify({"error": str(exc)}), 500

    return Response(stream, mimetype="multipart/x-mixed-replace; boundary=frame")


@api_bp.get("/get-violations")
def get_violations():
    limit = int(request.args.get("limit", "200"))
    cfg: Config = current_app.config["APP_CONFIG"]
    rows = _services()["db_service"].get_violations(limit=limit)
    rows = _with_image_status(rows, cfg.base_dir)
    return jsonify({"violations": rows, "count": len(rows)})


@api_bp.patch("/update-violation/<int:violation_id>")
def update_violation(violation_id: int):
    payload = request.get_json(silent=True) or {}
    plate_number = normalize_plate_text((payload.get("plate_number") or "NOTFOUND").strip()) or "NOTFOUND"
    email = (payload.get("email") or "").strip()
    row = _services()["violation_service"].update_violation(violation_id, plate_number, email)
    if not row:
        return jsonify({"error": "Violation not found."}), 404
    return jsonify({"message": "Violation updated successfully.", "violation": row})


@api_bp.post("/rerun-ocr/<int:violation_id>")
def rerun_ocr(violation_id: int):
    cfg: Config = current_app.config["APP_CONFIG"]
    services = _services()
    row = services["db_service"].get_violation_by_id(violation_id)
    if not row:
        return jsonify({"error": "Violation not found."}), 404

    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = cfg.base_dir / image_path
    if not image_path.exists():
        return jsonify({"error": "Violation image not found."}), 404

    image = cv2.imread(str(image_path))
    if image is None:
        return jsonify({"error": "Violation image could not be loaded."}), 400

    plate_model = services["model_service"].get_models()[1]
    result = plate_model.predict(source=image, conf=cfg.plate_confidence, imgsz=cfg.inference_size, verbose=False)[0]
    best_crop = None
    best_conf = 0.0
    for box in result.boxes:
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size > 0 and conf >= best_conf:
            best_crop = crop
            best_conf = conf

    if best_crop is None:
        best_crop = image

    text, ocr_conf = services["ocr_service"].read_plate(best_crop)
    text = normalize_plate_text(text) or "NOTFOUND"
    updated = services["violation_service"].update_violation(violation_id, text, row.get("email", ""))
    return jsonify(
        {
            "message": "OCR re-run completed.",
            "plate_number": text,
            "ocr_confidence": ocr_conf,
            "violation": updated,
        }
    )


@api_bp.delete("/delete-violation/<int:violation_id>")
def delete_violation(violation_id: int):
    cfg: Config = current_app.config["APP_CONFIG"]
    deleted = _services()["violation_service"].delete_violation(violation_id, cfg.base_dir)
    if not deleted:
        return jsonify({"error": "Violation not found."}), 404
    return jsonify({"message": "Violation deleted successfully.", "id": violation_id})


@api_bp.post("/send-email")
def send_email():
    services = _services()
    db_service = services["db_service"]
    email_service = services["email_service"]

    payload = request.get_json(silent=True) or {}
    violation_id = payload.get("violation_id")
    to_email = (payload.get("email") or "").strip()
    plate_number = (payload.get("plate_number") or "").strip()
    image_path = (payload.get("image_path") or "").strip()
    timestamp = (payload.get("timestamp") or "").strip()

    if violation_id:
        row = db_service.get_violation_by_id(int(violation_id))
        if not row:
            return jsonify({"error": "Violation not found."}), 404
        plate_number = row["plate_number"]
        image_path = row["image_path"]
        timestamp = row["timestamp"]
        to_email = to_email or row.get("email", "")

    if not to_email:
        return jsonify({"error": "Email is required."}), 400
    if not image_path:
        return jsonify({"error": "image_path is required."}), 400
    if not timestamp:
        timestamp = "N/A"

    try:
        if not Path(image_path).exists():
            image_path = str((cfg := current_app.config["APP_CONFIG"]).base_dir / image_path)
        email_service.send_violation_email(
            to_email=to_email,
            plate_number=plate_number or "NOTFOUND",
            image_path=image_path,
            timestamp=timestamp,
        )
    except Exception as exc:
        LOGGER.exception("manual email send failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify({"message": "Email sent successfully."})


@api_bp.get("/smtp-config-status")
def smtp_config_status():
    return jsonify(_services()["email_service"].config_status())
