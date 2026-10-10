from __future__ import annotations

import logging
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.file_utils import normalize_plate_text

Box = Tuple[int, int, int, int]
LOGGER = logging.getLogger(__name__)


class FastALPRService:
    def __init__(
        self,
        detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        ocr_model: str = "cct-s-v2-global-model",
    ):
        self.detector_model = detector_model
        self.ocr_model = ocr_model
        self._alpr = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import fast_alpr  # noqa: F401
        except ImportError:
            self._available = False
            LOGGER.info("FastALPR is not installed; using YOLO/PaddleOCR fallback")
            return False
        self._available = True
        return True

    def _get_alpr(self):
        if not self.is_available():
            return None
        if self._alpr is None:
            from fast_alpr import ALPR

            self._alpr = ALPR(detector_model=self.detector_model, ocr_model=self.ocr_model)
            LOGGER.info("FastALPR initialized detector=%s ocr=%s", self.detector_model, self.ocr_model)
        return self._alpr

    @staticmethod
    def _result_box(result) -> Optional[Box]:
        detection = getattr(result, "detection", None)
        candidates = [
            getattr(result, "box", None),
            getattr(result, "bbox", None),
            getattr(result, "det_box", None),
            getattr(detection, "box", None),
            getattr(detection, "bbox", None),
        ]
        for raw_box in candidates:
            if raw_box is None:
                continue
            values = list(raw_box)
            if len(values) >= 4:
                x1, y1, x2, y2 = values[:4]
                return int(x1), int(y1), int(x2), int(y2)
        return None

    @staticmethod
    def _ocr_payload(result) -> Tuple[str, float, str]:
        ocr = getattr(result, "ocr", None)
        if ocr is None:
            return "NOTFOUND", 0.0, ""

        text = normalize_plate_text(str(getattr(ocr, "text", "") or ""))
        confidence = getattr(ocr, "confidence", 0.0)
        if isinstance(confidence, list):
            confidence = mean([float(value) for value in confidence]) if confidence else 0.0
        region = str(getattr(ocr, "region", "") or "")
        return text or "NOTFOUND", float(confidence or 0.0), region

    def detect(self, image: np.ndarray) -> List[Dict]:
        alpr = self._get_alpr()
        if alpr is None or image is None or image.size == 0:
            return []

        try:
            results = alpr.predict(image)
        except Exception:
            LOGGER.exception("FastALPR prediction failed; using fallback")
            return []

        detections: List[Dict] = []
        for result in results or []:
            text, confidence, region = self._ocr_payload(result)
            box = self._result_box(result)
            detections.append(
                {
                    "plate_number": text,
                    "ocr_confidence": confidence,
                    "plate_box": box,
                    "plate_scope": f"fast_alpr:{region}" if region else "fast_alpr",
                }
            )
        return detections
