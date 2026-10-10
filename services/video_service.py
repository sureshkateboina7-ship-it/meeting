from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import cv2

from services.detection_service import DetectionService
from services.violation_service import ViolationService

LOGGER = logging.getLogger(__name__)


class VideoProcessingService:
    def __init__(
        self,
        detection_service: DetectionService,
        violation_service: ViolationService,
        frame_skip: int = 1,
    ):
        self.detection_service = detection_service
        self.violation_service = violation_service
        self.frame_skip = max(0, frame_skip)

    def process_video(
        self,
        input_path: Path,
        output_path: Path,
        helmet_confidence: float,
        plate_confidence: float,
        email: str = "",
    ) -> Dict:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {input_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        processed_frames = 0
        total_violations = 0
        detected_frames = 0
        last_annotated = None
        LOGGER.info("video processing open input=%s fps=%s size=%sx%s", input_path, fps, width, height)

        try:
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                frame_count += 1
                should_process = (frame_count == 1) or (frame_count % (self.frame_skip + 1) == 0)
                if should_process:
                    annotated, violations = self.detection_service.process_frame(
                        frame=frame,
                        helmet_confidence=helmet_confidence,
                        plate_confidence=plate_confidence,
                    )
                    persisted = self.violation_service.persist_violations(
                        frame=annotated,
                        violations=violations,
                        email=email,
                        source_type="video",
                        deduplicate=True,
                    )
                    if violations:
                        detected_frames += 1
                    total_violations += len(persisted)
                    processed_frames += 1
                    last_annotated = annotated
                    writer.write(annotated)
                else:
                    writer.write(last_annotated if last_annotated is not None else frame)
        finally:
            capture.release()
            writer.release()

        stats = {
            "frames_total": frame_count,
            "frames_processed": processed_frames,
            "frames_with_detections": detected_frames,
            "violations_saved": total_violations,
            "output_path": str(output_path).replace("\\", "/"),
        }
        LOGGER.info("video processing complete stats=%s", stats)
        return stats
