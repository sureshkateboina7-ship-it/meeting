from __future__ import annotations

import logging
from typing import Generator

import cv2

from services.detection_service import DetectionService
from services.violation_service import ViolationService

LOGGER = logging.getLogger(__name__)


class LiveDetectionService:
    def __init__(
        self,
        detection_service: DetectionService,
        violation_service: ViolationService,
        frame_skip: int = 1,
    ):
        self.detection_service = detection_service
        self.violation_service = violation_service
        self.frame_skip = max(0, frame_skip)

    def stream(
        self,
        source: str,
        helmet_confidence: float,
        plate_confidence: float,
        email: str = "",
    ) -> Generator[bytes, None, None]:
        source_input = int(source) if str(source).isdigit() else source
        cap = cv2.VideoCapture(source_input)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera/stream source: {source}")

        frame_idx = 0
        LOGGER.info("live stream opened source=%s", source)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    LOGGER.warning("live stream frame read failed source=%s", source)
                    break

                frame_idx += 1
                should_process = (frame_idx == 1) or (frame_idx % (self.frame_skip + 1) == 0)
                if should_process:
                    annotated, violations = self.detection_service.process_frame(
                        frame=frame,
                        helmet_confidence=helmet_confidence,
                        plate_confidence=plate_confidence,
                    )
                    self.violation_service.persist_violations(
                        frame=annotated,
                        violations=violations,
                        email=email,
                        source_type="live",
                        deduplicate=True,
                    )
                    if violations:
                        LOGGER.info("live frame=%s violations=%s", frame_idx, len(violations))
                else:
                    annotated = frame

                ok, buffer = cv2.imencode(".jpg", annotated)
                if not ok:
                    continue

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        finally:
            cap.release()
            LOGGER.info("live stream released source=%s", source)
