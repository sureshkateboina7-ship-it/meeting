import io
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np

from app import create_app


class AppRoutesTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_pages_are_available(self):
        paths = [
            "/",
            "/image-detection",
            "/video-detection",
            "/live-camera",
            "/violations-dashboard",
            "/email-page",
            "/about",
        ]
        for path in paths:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)

    def test_upload_image_requires_file(self):
        response = self.client.post("/upload-image", data={})
        self.assertEqual(response.status_code, 400)

    def test_upload_video_requires_file(self):
        response = self.client.post("/upload-video", data={})
        self.assertEqual(response.status_code, 400)

    def test_upload_image_invalid_extension(self):
        data = {"file": (io.BytesIO(b"abc"), "x.txt")}
        response = self.client.post("/upload-image", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_get_violations_api(self):
        response = self.client.get("/get-violations")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("violations", payload)

    def test_send_email_requires_email(self):
        response = self.client.post("/send-email", json={"plate_number": "ABC123", "image_path": "x.jpg"})
        self.assertEqual(response.status_code, 400)

    def test_smtp_config_status_does_not_expose_password(self):
        response = self.client.get("/smtp-config-status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("password_length", payload)
        self.assertNotIn("password", payload)

    def test_delete_violation_not_found(self):
        response = self.client.delete("/delete-violation/999999")
        self.assertEqual(response.status_code, 404)

    def test_send_email_with_violation_id_uses_service(self):
        db_service = self.app.extensions["services"]["db_service"]
        record_id = db_service.insert_violation("KA01AB1234", "static/outputs/test.jpg", "saved@example.com")
        email_service = self.app.extensions["services"]["email_service"]
        original = email_service.send_violation_email
        email_service.send_violation_email = MagicMock()
        try:
            response = self.client.post("/send-email", json={"violation_id": record_id, "email": "to@example.com"})
            self.assertEqual(response.status_code, 200)
            email_service.send_violation_email.assert_called_once()
        finally:
            email_service.send_violation_email = original
            db_service.delete_violation(record_id)

    def test_persisted_violation_keeps_record_when_email_fails(self):
        services = self.app.extensions["services"]
        violation_service = services["violation_service"]
        db_service = services["db_service"]
        original = services["email_service"].send_violation_email
        services["email_service"].send_violation_email = MagicMock(side_effect=RuntimeError("smtp failed"))
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        saved = []
        try:
            saved = violation_service.persist_violations(
                frame=frame,
                violations=[
                    {
                        "plate_number": "KA01AB1234",
                        "rider_box": (1, 1, 10, 10),
                        "motorcycle_box": (1, 8, 18, 18),
                    }
                ],
                email="to@example.com",
                source_type="test",
                deduplicate=False,
            )
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["email_status"], "failed")
            self.assertIn("email_error", saved[0])
        finally:
            services["email_service"].send_violation_email = original
            if saved:
                db_service.delete_violation(saved[0]["id"])

    def test_delete_violation_success(self):
        base_dir = Path(self.app.config["APP_CONFIG"].base_dir)
        output_dir = base_dir / "static" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_file = output_dir / "delete_me_test.jpg"
        image_file.write_bytes(b"123")

        db_service = self.app.extensions["services"]["db_service"]
        record_id = db_service.insert_violation("KA01AB1234", "static/outputs/delete_me_test.jpg", "")
        response = self.client.delete(f"/delete-violation/{record_id}")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(image_file.exists())

    def test_update_violation_success(self):
        db_service = self.app.extensions["services"]["db_service"]
        record_id = db_service.insert_violation("OLD1234", "static/outputs/test.jpg", "")
        try:
            response = self.client.patch(
                f"/update-violation/{record_id}",
                json={"plate_number": "NEW1234", "email": "edited@example.com"},
            )
            self.assertEqual(response.status_code, 200)
            updated = db_service.get_violation_by_id(record_id)
            self.assertEqual(updated["plate_number"], "NEW1234")
            self.assertEqual(updated["email"], "edited@example.com")
        finally:
            db_service.delete_violation(record_id)

    def test_rerun_ocr_updates_plate(self):
        base_dir = Path(self.app.config["APP_CONFIG"].base_dir)
        output_dir = base_dir / "static" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        image_file = output_dir / "rerun_ocr_test.jpg"
        sample = np.zeros((20, 40, 3), dtype=np.uint8)
        cv2.imwrite(str(image_file), sample)

        db_service = self.app.extensions["services"]["db_service"]
        record_id = db_service.insert_violation("NOTFOUND", "static/outputs/rerun_ocr_test.jpg", "")

        model_service = self.app.extensions["services"]["model_service"]
        ocr_service = self.app.extensions["services"]["ocr_service"]

        class _FakeXY:
            def __init__(self, vals):
                self._vals = vals
            def tolist(self):
                return list(self._vals)

        class _FakeBox:
            def __init__(self):
                self.conf = [0.9]
                self.xyxy = [_FakeXY((0, 0, 10, 10))]

        class _FakeResult:
            def __init__(self):
                self.boxes = [_FakeBox()]

        class _FakePlateModel:
            def predict(self, source, conf, imgsz=None, verbose=False):
                return [_FakeResult()]

        original_get_models = model_service.get_models
        original_read_plate = ocr_service.read_plate
        model_service.get_models = MagicMock(return_value=(None, _FakePlateModel(), None))
        ocr_service.read_plate = MagicMock(return_value=("AP09AB1234", 0.95))
        try:
            response = self.client.post(f"/rerun-ocr/{record_id}")
            self.assertEqual(response.status_code, 200)
            updated = db_service.get_violation_by_id(record_id)
            self.assertEqual(updated["plate_number"], "AP09AB1234")
        finally:
            model_service.get_models = original_get_models
            ocr_service.read_plate = original_read_plate
            db_service.delete_violation(record_id)
            if image_file.exists():
                image_file.unlink()


if __name__ == "__main__":
    unittest.main()
