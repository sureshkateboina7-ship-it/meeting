from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from services.alpr_service import FastALPRService
from services.model_service import ModelService
from services.ocr_service import OCRService
from utils.file_utils import normalize_plate_text
from utils.image_utils import crop_image, iou

Box = Tuple[int, int, int, int]
LOGGER = logging.getLogger(__name__)
NUMBER_PLATE_NOT_FOUND = "NUMBER PLATE NOT FOUND"
MISSING_PLATE_VALUES = {"UNREADABLE", "NOTFOUND", NUMBER_PLATE_NOT_FOUND}


@dataclass
class DetectionLabelConfig:
    rider_keywords: List[str]
    motorcycle_keywords: List[str]
    no_helmet_keywords: List[str]
    helmet_keywords: List[str]


class DetectionService:
    def __init__(
        self,
        model_service: ModelService,
        ocr_service: OCRService,
        label_config: DetectionLabelConfig,
        inference_size: int = 640,
        alpr_service: Optional[FastALPRService] = None,
        alpr_engine: str = "auto",
    ):
        self.model_service = model_service
        self.ocr_service = ocr_service
        self.label_config = label_config
        self.inference_size = inference_size
        self.alpr_service = alpr_service
        self.alpr_engine = alpr_engine

    @staticmethod
    def _normalize_label(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    def _match_keywords(self, label: str, keywords: List[str]) -> bool:
        normalized = self._normalize_label(label)
        return any(self._normalize_label(keyword) == normalized for keyword in keywords)

    def _box_to_int(self, box) -> Box:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        return int(x1), int(y1), int(x2), int(y2)

    @staticmethod
    def _box_center(box: Box) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _horizontal_overlap_ratio(box_a: Box, box_b: Box) -> float:
        ax1, _, ax2, _ = box_a
        bx1, _, bx2, _ = box_b
        overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        width = max(1, min(ax2 - ax1, bx2 - bx1))
        return overlap / width

    @staticmethod
    def _expand_box(box: Box, width: int, height: int, pad_ratio: float = 0.15) -> Box:
        x1, y1, x2, y2 = box
        bw, bh = (x2 - x1), (y2 - y1)
        px, py = int(bw * pad_ratio), int(bh * pad_ratio)
        return (
            max(0, x1 - px),
            max(0, y1 - py),
            min(width - 1, x2 + px),
            min(height - 1, y2 + py),
        )

    @staticmethod
    def _clamp_box(box: Box, width: int, height: int) -> Box:
        x1, y1, x2, y2 = box
        return (
            max(0, min(x1, width - 1)),
            max(0, min(y1, height - 1)),
            max(0, min(x2, width - 1)),
            max(0, min(y2, height - 1)),
        )

    def _predict(self, model, source, confidence: float):
        return model.predict(source=source, conf=confidence, imgsz=self.inference_size, verbose=False)[0]

    def _no_helmet_is_on_rider_head(self, no_helmet_box: Box, rider_box: Box) -> bool:
        rx1, ry1, rx2, ry2 = rider_box
        _, ny1, _, ny2 = no_helmet_box
        nx_c, ny_c = self._box_center(no_helmet_box)
        rider_h = max(1, ry2 - ry1)
        inside_rider = rx1 <= nx_c <= rx2 and ry1 <= ny_c <= ry2
        in_upper_body = ny2 <= (ry1 + int(rider_h * 0.45))
        helmet_small_enough = (ny2 - ny1) <= int(rider_h * 0.25)
        return inside_rider and in_upper_body and helmet_small_enough

    def _helmet_is_on_rider_head(self, helmet_box: Box, rider_box: Box) -> bool:
        rx1, ry1, rx2, ry2 = rider_box
        hx_c, hy_c = self._box_center(helmet_box)
        rider_h = max(1, ry2 - ry1)
        return rx1 <= hx_c <= rx2 and ry1 <= hy_c <= (ry1 + int(rider_h * 0.38))

    def _head_region_from_rider(self, rider_box: Box) -> Box:
        x1, y1, x2, y2 = rider_box
        rider_h = max(1, y2 - y1)
        return x1, y1, x2, y1 + int(rider_h * 0.32)

    def _motorcycle_match_score(self, rider_box: Box, motorcycle_box: Box) -> float:
        rider_bottom = rider_box[3]
        motorcycle_top = motorcycle_box[1]
        vertical_gap = abs(rider_bottom - motorcycle_top)
        rider_h = max(1, rider_box[3] - rider_box[1])
        rider_w = max(1, rider_box[2] - rider_box[0])
        bike_w = max(1, motorcycle_box[2] - motorcycle_box[0])

        vertical_score = max(0.0, 1.0 - (vertical_gap / (rider_h * 1.4)))
        overlap_score = self._horizontal_overlap_ratio(rider_box, motorcycle_box)
        iou_score = iou(rider_box, motorcycle_box)

        rider_cx, _ = self._box_center(rider_box)
        bike_cx, _ = self._box_center(motorcycle_box)
        center_gap = abs(rider_cx - bike_cx)
        center_scale = max(rider_w, bike_w)
        center_score = max(0.0, 1.0 - (center_gap / (center_scale * 1.3)))
        return (0.35 * overlap_score) + (0.25 * vertical_score) + (0.30 * center_score) + (0.10 * iou_score)

    def _find_associated_motorcycle(self, rider_box: Box, motorcycle_boxes: List[Box]) -> Tuple[Optional[Box], float]:
        if not motorcycle_boxes:
            return None, 0.0
        scored = [(self._motorcycle_match_score(rider_box, bike), bike) for bike in motorcycle_boxes]
        best_score, best_bike = max(scored, key=lambda x: x[0])
        if best_score < 0.35:
            return None, 0.0
        return best_bike, float(best_score)

    def _find_best_rider_for_no_helmet(self, no_helmet_box: Box, rider_boxes: List[Box], motorcycle_boxes: List[Box]) -> Tuple[Optional[Box], Optional[Box]]:
        candidates: List[Tuple[float, Box, Optional[Box]]] = []
        for rider_box in rider_boxes:
            if not self._no_helmet_is_on_rider_head(no_helmet_box, rider_box):
                continue
            bike_box, bike_score = self._find_associated_motorcycle(rider_box, motorcycle_boxes)
            if bike_box is None:
                continue
            overlap = iou(no_helmet_box, rider_box)
            score = (0.6 * overlap) + (0.4 * bike_score)
            candidates.append((score, rider_box, bike_box))

        if not candidates:
            return None, None
        _, best_rider, best_bike = max(candidates, key=lambda x: x[0])
        return best_rider, best_bike

    def _nearest_box(self, source_box: Box, boxes: List[Box]) -> Optional[Box]:
        if not boxes:
            return None
        sx, sy = self._box_center(source_box)
        return min(boxes, key=lambda box: np.hypot(self._box_center(box)[0] - sx, self._box_center(box)[1] - sy))

    def _infer_rider_box(self, no_helmet_box: Box, width: int, height: int) -> Box:
        x1, y1, x2, y2 = no_helmet_box
        head_w = max(1, x2 - x1)
        head_h = max(1, y2 - y1)
        cx, _ = self._box_center(no_helmet_box)
        inferred = (
            int(cx - head_w * 1.8),
            max(0, y1 - int(head_h * 0.35)),
            int(cx + head_w * 1.8),
            y2 + int(head_h * 5.0),
        )
        return self._clamp_box(inferred, width, height)

    def _infer_motorcycle_box(self, rider_box: Box, width: int, height: int) -> Box:
        x1, y1, x2, y2 = rider_box
        rider_w = max(1, x2 - x1)
        rider_h = max(1, y2 - y1)
        cx, _ = self._box_center(rider_box)
        inferred = (
            int(cx - rider_w * 1.25),
            y1 + int(rider_h * 0.55),
            int(cx + rider_w * 1.25),
            y2 + int(rider_h * 0.25),
        )
        return self._clamp_box(inferred, width, height)

    def _resolve_violation_context(
        self,
        no_helmet_box: Box,
        rider_boxes: List[Box],
        motorcycle_boxes: List[Box],
        width: int,
        height: int,
    ) -> Tuple[Optional[Box], Optional[Box], bool]:
        rider_box, motorcycle_box = self._find_best_rider_for_no_helmet(no_helmet_box, rider_boxes, motorcycle_boxes)
        if rider_box is not None and motorcycle_box is not None:
            return rider_box, motorcycle_box, True

        # A traffic challan should be created only when the no-helmet head can
        # be tied to a motorcycle. Do not invent a bike context for pedestrians.
        inferred_rider = self._infer_rider_box(no_helmet_box, width, height)
        inferred_motorcycle, bike_score = self._find_associated_motorcycle(inferred_rider, motorcycle_boxes)
        if inferred_motorcycle is not None and bike_score >= 0.35:
            return inferred_rider, inferred_motorcycle, False

        return None, None, False

    def _draw_box(self, image: np.ndarray, box: Box, label: str, color: Tuple[int, int, int]) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def _plate_search_regions(self, frame: np.ndarray, rider_box: Box, motorcycle_box: Box) -> List[Tuple[str, Box]]:
        h, w = frame.shape[:2]
        mx1, my1, mx2, my2 = motorcycle_box
        motorcycle_h = max(1, my2 - my1)
        return [
            ("motorcycle_crop", self._expand_box(motorcycle_box, w, h, 0.18)),
            ("motorcycle_lower_half", self._expand_box((mx1, my1 + int(motorcycle_h * 0.35), mx2, my2), w, h, 0.16)),
            ("motorcycle_plate_band", self._expand_box((mx1, my1 + int(motorcycle_h * 0.50), mx2, my2), w, h, 0.12)),
        ]

    def _detect_best_plate(self, plate_model, frame: np.ndarray, rider_box: Box, motorcycle_box: Box, plate_confidence: float) -> Tuple[Optional[Box], float, str]:
        h, w = frame.shape[:2]
        regions = self._plate_search_regions(frame, rider_box, motorcycle_box)

        rider_cx, rider_cy = self._box_center(rider_box)
        best_box = None
        best_score = -1.0
        best_conf = 0.0
        best_scope = "none"
        confidence_levels = sorted({plate_confidence, max(0.12, plate_confidence * 0.60)}, reverse=True)

        for stage_idx, (scope, region_box) in enumerate(regions):
            rx1, ry1, _, _ = region_box
            region_crop = crop_image(frame, region_box)
            if region_crop.size == 0:
                continue

            for confidence in confidence_levels:
                result = self._predict(plate_model, region_crop, confidence)
                names = result.names if isinstance(result.names, dict) else {}
                for plate_box in result.boxes:
                    conf = float(plate_box.conf[0])
                    cls_id = int(plate_box.cls[0])
                    cls_label = str(names.get(cls_id, "plate")).lower()
                    if all(token not in cls_label for token in ("plate", "license", "number")) and len(result.boxes) > 1:
                        continue

                    px1, py1, px2, py2 = self._box_to_int(plate_box)
                    abs_box = (rx1 + px1, ry1 + py1, rx1 + px2, ry1 + py2)
                    pcx, pcy = self._box_center(abs_box)
                    dist = np.hypot(pcx - rider_cx, pcy - rider_cy)
                    dist_norm = dist / max(1.0, np.hypot(w, h))
                    locality_bonus = max(0.0, 1.0 - dist_norm)
                    stage_bonus = max(0.0, 0.25 - (stage_idx * 0.04))
                    score = (0.65 * conf) + (0.25 * locality_bonus) + stage_bonus
                    if score > best_score:
                        best_score = score
                        best_conf = conf
                        best_box = abs_box
                        best_scope = f"{scope}@{confidence:.2f}"

            if best_box is not None and scope != "full_frame":
                break

        return best_box, best_conf, best_scope

    @staticmethod
    def _box_distance_score(reference_box: Box, candidate_box: Box, width: int, height: int) -> float:
        ref_cx, ref_cy = DetectionService._box_center(reference_box)
        cand_cx, cand_cy = DetectionService._box_center(candidate_box)
        dist = np.hypot(cand_cx - ref_cx, cand_cy - ref_cy)
        return max(0.0, 1.0 - (dist / max(1.0, np.hypot(width, height))))

    def _fast_alpr_plate(self, frame: np.ndarray, rider_box: Box, motorcycle_box: Box) -> Tuple[Optional[Box], float, str, str, float]:
        if self.alpr_engine == "disabled" or self.alpr_service is None:
            return None, 0.0, "none", "UNREADABLE", 0.0
        if self.alpr_engine == "fast_alpr" and not self.alpr_service.is_available():
            LOGGER.warning("ALPR_ENGINE=fast_alpr requested, but fast-alpr is not installed")
            return None, 0.0, "none", "UNREADABLE", 0.0
        if not self.alpr_service.is_available():
            return None, 0.0, "none", "UNREADABLE", 0.0

        h, w = frame.shape[:2]
        regions = self._plate_search_regions(frame, rider_box, motorcycle_box)
        best: Tuple[float, Optional[Box], float, str, str, float] = (-1.0, None, 0.0, "none", "UNREADABLE", 0.0)
        for index, (scope, region_box) in enumerate(regions):
            rx1, ry1, _, _ = region_box
            crop = crop_image(frame, region_box)
            for result in self.alpr_service.detect(crop):
                raw_box = result.get("plate_box")
                plate_text = normalize_plate_text(result.get("plate_number", "")) or "UNREADABLE"
                ocr_conf = float(result.get("ocr_confidence", 0.0) or 0.0)
                abs_box = None
                locality = 0.35
                if raw_box:
                    px1, py1, px2, py2 = raw_box
                    abs_box = (rx1 + px1, ry1 + py1, rx1 + px2, ry1 + py2)
                    locality = self._box_distance_score(motorcycle_box, abs_box, w, h)
                text_bonus = 0.25 if plate_text not in MISSING_PLATE_VALUES else 0.0
                score = ocr_conf + text_bonus + locality + max(0.0, 0.10 - index * 0.02)
                if score > best[0]:
                    best = (
                        score,
                        abs_box,
                        float(result.get("plate_confidence", ocr_conf) or ocr_conf),
                        str(result.get("plate_scope", "fast_alpr")),
                        plate_text,
                        ocr_conf,
                    )
                if ocr_conf >= 0.75 and plate_text not in MISSING_PLATE_VALUES:
                    break

        _, box, plate_conf, scope, text, ocr_conf = best
        return box, plate_conf, scope, text, ocr_conf

    def _ocr_plate_from_regions(self, frame: np.ndarray, rider_box: Box, motorcycle_box: Box) -> Tuple[str, float, str, Optional[Box]]:
        best_text = "NOTFOUND"
        best_conf = 0.0
        best_scope = "none"
        best_region = None
        ocr_scopes = {"motorcycle_crop", "motorcycle_lower_half", "motorcycle_plate_band"}
        attempts = 0
        for index, (scope, region_box) in enumerate(self._plate_search_regions(frame, rider_box, motorcycle_box)):
            if scope == "full_frame":
                continue
            if scope not in ocr_scopes:
                continue
            if attempts >= 2:
                break
            attempts += 1
            region_crop = crop_image(frame, region_box)
            if region_crop.size == 0:
                continue
            text, confidence = self.ocr_service.read_plate(region_crop)
            text = normalize_plate_text(text) or "NOTFOUND"
            if text not in {"NOTFOUND", "UNREADABLE"}:
                weighted_conf = min(1.0, float(confidence) + max(0.0, 0.12 - (index * 0.02)))
                if weighted_conf > best_conf:
                    best_text = text
                    best_conf = weighted_conf
                    best_scope = f"ocr_{scope}"
                    best_region = region_box
                if confidence >= 0.75 and 6 <= len(text) <= 10:
                    break
        return best_text, best_conf, best_scope, best_region

    def process_frame(self, frame: np.ndarray, helmet_confidence: float, plate_confidence: float) -> Tuple[np.ndarray, List[Dict]]:
        helmet_model, plate_model, traffic_model = self.model_service.get_models()
        annotated = frame.copy()
        height, width = frame.shape[:2]

        helmet_result = self._predict(helmet_model, frame, helmet_confidence)
        traffic_result = self._predict(traffic_model, frame, min(helmet_confidence, 0.25))
        helmet_names = helmet_result.names if isinstance(helmet_result.names, dict) else {}
        traffic_names = traffic_result.names if isinstance(traffic_result.names, dict) else {}

        rider_boxes: List[Box] = []
        motorcycle_boxes: List[Box] = []
        no_helmet_boxes: List[Box] = []
        helmet_boxes: List[Box] = []
        inferred_contexts: Dict[Box, Tuple[Box, Box]] = {}

        for box in traffic_result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = str(traffic_names.get(class_id, class_id))
            x1, y1, x2, y2 = self._box_to_int(box)
            coords = (
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)),
                max(0, min(y2, height - 1)),
            )
            if self._match_keywords(label, self.label_config.rider_keywords):
                rider_boxes.append(coords)
            elif self._match_keywords(label, self.label_config.motorcycle_keywords):
                motorcycle_boxes.append(coords)

        for box in helmet_result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = str(helmet_names.get(class_id, class_id))
            x1, y1, x2, y2 = self._box_to_int(box)
            coords = (
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(0, min(x2, width - 1)),
                max(0, min(y2, height - 1)),
            )
            if self._match_keywords(label, self.label_config.no_helmet_keywords):
                no_helmet_boxes.append(coords)
            elif self._match_keywords(label, self.label_config.helmet_keywords):
                helmet_boxes.append(coords)

        explicit_no_helmet_boxes = list(no_helmet_boxes)
        for rider_box in rider_boxes:
            if any(self._no_helmet_is_on_rider_head(explicit_box, rider_box) for explicit_box in explicit_no_helmet_boxes):
                continue
            motorcycle_box, _ = self._find_associated_motorcycle(rider_box, motorcycle_boxes)
            if motorcycle_box is None:
                continue
            has_helmet = any(self._helmet_is_on_rider_head(helmet_box, rider_box) for helmet_box in helmet_boxes)
            if not has_helmet:
                inferred_head = self._head_region_from_rider(rider_box)
                no_helmet_boxes.append(inferred_head)
                inferred_contexts[inferred_head] = (rider_box, motorcycle_box)
                LOGGER.info("no-helmet inferred from rider without helmet on head rider=%s", rider_box)

        LOGGER.info(
            "frame processed: riders=%s motorcycles=%s helmets=%s no_helmet=%s",
            len(rider_boxes),
            len(motorcycle_boxes),
            len(helmet_boxes),
            len(no_helmet_boxes),
        )

        violations: List[Dict] = []
        used_motorcycle_boxes: List[Box] = []
        for no_helmet_box in no_helmet_boxes:
            if no_helmet_box in inferred_contexts:
                rider_box, motorcycle_box = inferred_contexts[no_helmet_box]
                strict_match = True
            else:
                rider_box, motorcycle_box, strict_match = self._resolve_violation_context(
                    no_helmet_box=no_helmet_box,
                    rider_boxes=rider_boxes,
                    motorcycle_boxes=motorcycle_boxes,
                    width=width,
                    height=height,
                )
            if rider_box is None or motorcycle_box is None:
                LOGGER.info("no-helmet detection ignored because no motorcycle rider context was found box=%s", no_helmet_box)
                continue
            if any(iou(motorcycle_box, used_box) >= 0.55 for used_box in used_motorcycle_boxes):
                LOGGER.info("duplicate no-helmet rider context skipped for same motorcycle=%s", motorcycle_box)
                continue
            used_motorcycle_boxes.append(motorcycle_box)
            if not strict_match:
                LOGGER.info("no-helmet violation saved with inferred context rider=%s motorcycle=%s", rider_box, motorcycle_box)

            plate_text = NUMBER_PLATE_NOT_FOUND
            ocr_conf = 0.0
            plate_box, plate_conf, plate_scope, fast_text, fast_ocr_conf = self._fast_alpr_plate(
                frame=frame,
                rider_box=rider_box,
                motorcycle_box=motorcycle_box,
            )
            if fast_text not in MISSING_PLATE_VALUES:
                plate_text = fast_text
                ocr_conf = fast_ocr_conf
                if plate_box is not None:
                    self._draw_box(annotated, plate_box, f"Plate: {plate_text} ({plate_scope})", (0, 255, 0))

            if plate_text in MISSING_PLATE_VALUES:
                plate_box, plate_conf, plate_scope = self._detect_best_plate(
                    plate_model=plate_model,
                    frame=frame,
                    rider_box=rider_box,
                    motorcycle_box=motorcycle_box,
                    plate_confidence=plate_confidence,
                )

            if plate_text in MISSING_PLATE_VALUES and plate_box is not None:
                plate_crop = crop_image(frame, plate_box)
                plate_text, ocr_conf = self.ocr_service.read_plate(plate_crop)
                plate_text = normalize_plate_text(plate_text) or NUMBER_PLATE_NOT_FOUND
                if plate_text in MISSING_PLATE_VALUES:
                    plate_text = NUMBER_PLATE_NOT_FOUND
                else:
                    self._draw_box(annotated, plate_box, f"Plate: {plate_text} ({plate_scope})", (0, 255, 0))

            if plate_text in MISSING_PLATE_VALUES:
                fallback_text, fallback_conf, fallback_scope, fallback_region = self._ocr_plate_from_regions(
                    frame=frame,
                    rider_box=rider_box,
                    motorcycle_box=motorcycle_box,
                )
                if fallback_text not in MISSING_PLATE_VALUES:
                    plate_text = fallback_text
                    ocr_conf = fallback_conf
                    plate_scope = fallback_scope
                    plate_box = fallback_region
                    plate_conf = fallback_conf
                    if fallback_region is not None:
                        self._draw_box(annotated, fallback_region, f"OCR Plate: {plate_text}", (0, 255, 0))
            if plate_text in MISSING_PLATE_VALUES:
                plate_text = NUMBER_PLATE_NOT_FOUND

            self._draw_box(annotated, no_helmet_box, "NO HELMET", (0, 0, 255))
            self._draw_box(annotated, rider_box, "NO HELMET VIOLATION", (0, 0, 255))
            self._draw_box(annotated, motorcycle_box, "ASSOCIATED BIKE", (0, 165, 255))
            violations.append(
                {
                    "plate_number": plate_text,
                    "plate_confidence": float(plate_conf),
                    "rider_box": rider_box,
                    "motorcycle_box": motorcycle_box,
                    "plate_box": plate_box,
                    "ocr_confidence": float(ocr_conf),
                    "plate_scope": plate_scope,
                    "context_inferred": not strict_match,
                }
            )

        LOGGER.info("violations detected in frame=%s", len(violations))
        return annotated, violations
