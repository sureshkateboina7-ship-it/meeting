from __future__ import annotations

import logging
import os
import re
from typing import Tuple

import cv2
import numpy as np

from utils.file_utils import normalize_plate_text
from utils.image_utils import enhance_plate_for_ocr

LOGGER = logging.getLogger(__name__)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"


class OCRService:
    def __init__(self, languages: str = "en", gpu: bool = False):
        self.languages = languages
        self.gpu = gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            try:
                from paddleocr import PaddleOCR
                import paddle
            except ImportError as exc:
                raise RuntimeError("PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle") from exc

            use_gpu = self.gpu and bool(getattr(paddle.device, "is_compiled_with_cuda", lambda: False)())
            device = "gpu:0" if use_gpu else "cpu"
            self._reader = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device=device,
            )
            LOGGER.info("PaddleOCR initialized device=%s", device)
        return self._reader

    @staticmethod
    def _append_candidate(candidates: list[Tuple[str, float]], raw_text: str, score) -> None:
        text = normalize_plate_text(str(raw_text))
        try:
            conf = float(score)
        except (TypeError, ValueError):
            conf = 0.0
        if text:
            candidates.append((text, conf))

    @classmethod
    def _append_combined_candidates(cls, candidates: list[Tuple[str, float]], lines: list[Tuple[str, float]]) -> None:
        normalized_lines = [(normalize_plate_text(text), conf) for text, conf in lines]
        normalized_lines = [(text, conf) for text, conf in normalized_lines if text]

        for window_size in (2, 3):
            if len(normalized_lines) < window_size:
                continue
            for start in range(0, len(normalized_lines) - window_size + 1):
                window = normalized_lines[start : start + window_size]
                text = "".join(part for part, _ in window)
                if not 6 <= len(text) <= 12:
                    continue
                confidence = sum(conf for _, conf in window) / window_size
                cls._append_candidate(candidates, text, confidence)

    @classmethod
    def _extract_candidates(cls, result) -> list[Tuple[str, float]]:
        candidates: list[Tuple[str, float]] = []
        if not result:
            return candidates

        for page in result:
            if isinstance(page, dict):
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                page_lines: list[Tuple[str, float]] = []
                for index, raw_text in enumerate(texts):
                    score = scores[index] if index < len(scores) else 0.0
                    try:
                        conf = float(score)
                    except (TypeError, ValueError):
                        conf = 0.0
                    page_lines.append((str(raw_text), conf))
                    cls._append_candidate(candidates, str(raw_text), conf)
                cls._append_combined_candidates(candidates, page_lines)
                continue

            if isinstance(page, list):
                page_lines: list[Tuple[str, float]] = []
                for line in page:
                    if not line or len(line) < 2:
                        continue
                    value = line[1]
                    if not isinstance(value, (list, tuple)) or len(value) < 2:
                        continue
                    try:
                        conf = float(value[1])
                    except (TypeError, ValueError):
                        conf = 0.0
                    page_lines.append((str(value[0]), conf))
                    cls._append_candidate(candidates, str(value[0]), conf)
                cls._append_combined_candidates(candidates, page_lines)

        return candidates

    @staticmethod
    def _plate_score(text: str, confidence: float) -> float:
        normalized = normalize_plate_text(text)
        if not normalized:
            return 0.0

        has_letter = any(ch.isalpha() for ch in normalized)
        has_digit = any(ch.isdigit() for ch in normalized)
        if not has_letter or not has_digit:
            return confidence * 0.20

        length_score = 1.0 if 6 <= len(normalized) <= 10 else 0.35
        digit_count = sum(ch.isdigit() for ch in normalized)
        letter_count = sum(ch.isalpha() for ch in normalized)
        balance_score = 1.0 if digit_count >= 3 and letter_count >= 2 else 0.50
        indian_plate_bonus = 1.25 if re.match(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$", normalized) else 0.0
        return confidence + length_score + balance_score + indian_plate_bonus

    @classmethod
    def _select_best_plate(cls, candidates: list[Tuple[str, float]]) -> Tuple[str, float]:
        if not candidates:
            return "NOTFOUND", 0.0
        text, confidence = max(candidates, key=lambda item: cls._plate_score(item[0], item[1]))
        return normalize_plate_text(text) or "NOTFOUND", float(confidence)

    @staticmethod
    def _ocr_variants(image: np.ndarray) -> list[np.ndarray]:
        variants = [image, enhance_plate_for_ocr(image)]
        return variants

    def read_plate(self, plate_image: np.ndarray) -> Tuple[str, float]:
        if plate_image is None or plate_image.size == 0:
            LOGGER.warning("OCR skipped because plate image was empty")
            return "NOTFOUND", 0.0

        reader = self._get_reader()
        candidates: list[Tuple[str, float]] = []
        for variant in self._ocr_variants(plate_image):
            result = reader.ocr(variant)
            candidates.extend(self._extract_candidates(result))

        if not candidates:
            LOGGER.warning("PaddleOCR returned no text for plate crop")
            return "NOTFOUND", 0.0

        return self._select_best_plate(candidates)
