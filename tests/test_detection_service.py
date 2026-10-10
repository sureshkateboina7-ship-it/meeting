import unittest

import numpy as np

from services.detection_service import DetectionLabelConfig, DetectionService


class _TensorList:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeDetBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [_TensorList(xyxy)]


class _FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


class _FakeModel:
    def __init__(self, result):
        self._result = result

    def predict(self, source, conf, imgsz=None, verbose=False):
        return [self._result]


class _FakePlateModel:
    def __init__(self, stage_results):
        self._stage_results = stage_results
        self.calls = 0

    def predict(self, source, conf, imgsz=None, verbose=False):
        index = min(self.calls, len(self._stage_results) - 1)
        self.calls += 1
        return [self._stage_results[index]]


class _FakeModelService:
    def __init__(self, helmet_model, plate_model, traffic_model):
        self.helmet_model = helmet_model
        self.plate_model = plate_model
        self.traffic_model = traffic_model

    def get_models(self):
        return self.helmet_model, self.plate_model, self.traffic_model


class _FakeOCR:
    def read_plate(self, plate_image):
        return "KA01AB1234", 0.93


class _SequenceOCR:
    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def read_plate(self, plate_image):
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]


class DetectionServiceLogicTest(unittest.TestCase):
    def setUp(self):
        self.label_cfg = DetectionLabelConfig(
            rider_keywords=["person", "rider"],
            motorcycle_keywords=["motorcycle", "bike"],
            no_helmet_keywords=["no-hardhat", "no_helmet"],
            helmet_keywords=["hardhat", "helmet"],
        )
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_no_helmet_pedestrian_is_ignored_when_bike_match_is_weak(self):
        helmet_names = {0: "Hardhat", 1: "NO-Hardhat"}
        traffic_names = {0: "person", 3: "motorcycle"}
        helmet_boxes = [_FakeDetBox(1, 0.95, (145, 170, 210, 240))]
        traffic_boxes = [
            _FakeDetBox(0, 0.91, (100, 160, 260, 520)),
            _FakeDetBox(3, 0.88, (900, 420, 1200, 670)),
        ]
        helmet_model = _FakeModel(_FakeResult(helmet_names, helmet_boxes))
        traffic_model = _FakeModel(_FakeResult(traffic_names, traffic_boxes))
        plate_model = _FakePlateModel([_FakeResult({0: "license_plate"}, [])])
        service = DetectionService(
            _FakeModelService(helmet_model, plate_model, traffic_model),
            _FakeOCR(),
            self.label_cfg,
        )

        _, violations = service.process_frame(self.frame, helmet_confidence=0.3, plate_confidence=0.3)
        self.assertEqual(len(violations), 0)

    def test_no_helmet_is_ignored_when_traffic_model_misses_motorcycle(self):
        helmet_names = {0: "Hardhat", 1: "NO-Hardhat"}
        traffic_names = {0: "person", 3: "motorcycle"}
        helmet_boxes = [_FakeDetBox(1, 0.95, (410, 150, 470, 220))]
        traffic_boxes = []
        helmet_model = _FakeModel(_FakeResult(helmet_names, helmet_boxes))
        traffic_model = _FakeModel(_FakeResult(traffic_names, traffic_boxes))
        plate_model = _FakePlateModel([_FakeResult({0: "license_plate"}, [])])
        service = DetectionService(
            _FakeModelService(helmet_model, plate_model, traffic_model),
            _FakeOCR(),
            self.label_cfg,
        )

        _, violations = service.process_frame(self.frame, helmet_confidence=0.3, plate_confidence=0.3)

        self.assertEqual(len(violations), 0)

    def test_rider_on_motorcycle_without_detected_helmet_is_saved_as_violation(self):
        helmet_names = {0: "Hardhat", 1: "NO-Hardhat"}
        traffic_names = {0: "person", 3: "motorcycle"}
        helmet_boxes = []
        traffic_boxes = [
            _FakeDetBox(0, 0.93, (350, 180, 550, 620)),
            _FakeDetBox(3, 0.90, (320, 430, 650, 700)),
        ]
        helmet_model = _FakeModel(_FakeResult(helmet_names, helmet_boxes))
        traffic_model = _FakeModel(_FakeResult(traffic_names, traffic_boxes))
        plate_model = _FakePlateModel([_FakeResult({0: "license_plate"}, [])])
        service = DetectionService(
            _FakeModelService(helmet_model, plate_model, traffic_model),
            _FakeOCR(),
            self.label_cfg,
        )

        _, violations = service.process_frame(self.frame, helmet_confidence=0.3, plate_confidence=0.3)

        self.assertEqual(len(violations), 1)
        self.assertFalse(violations[0]["context_inferred"])
        self.assertEqual(violations[0]["plate_number"], "KA01AB1234")

    def test_plate_detection_fallback_attempts_on_valid_no_helmet_rider(self):
        helmet_names = {0: "Hardhat", 1: "NO-Hardhat"}
        traffic_names = {0: "person", 3: "motorcycle"}
        helmet_boxes = [_FakeDetBox(1, 0.96, (410, 220, 470, 280))]
        traffic_boxes = [
            _FakeDetBox(0, 0.93, (350, 180, 550, 620)),
            _FakeDetBox(3, 0.90, (320, 430, 650, 700)),
        ]
        helmet_model = _FakeModel(_FakeResult(helmet_names, helmet_boxes))
        traffic_model = _FakeModel(_FakeResult(traffic_names, traffic_boxes))

        plate_stage_1 = _FakeResult({0: "license_plate"}, [])
        plate_stage_2 = _FakeResult({0: "license_plate"}, [_FakeDetBox(0, 0.86, (80, 70, 210, 130))])
        plate_model = _FakePlateModel([plate_stage_1, plate_stage_2])

        service = DetectionService(
            _FakeModelService(helmet_model, plate_model, traffic_model),
            _FakeOCR(),
            self.label_cfg,
        )
        _, violations = service.process_frame(self.frame, helmet_confidence=0.3, plate_confidence=0.3)

        self.assertEqual(len(violations), 1)
        self.assertIsNotNone(violations[0]["plate_box"])
        self.assertGreaterEqual(plate_model.calls, 2)
        self.assertTrue(violations[0]["plate_scope"].startswith(("motorcycle_crop", "motorcycle_lower_half", "motorcycle_plate_band")))

    def test_no_helmet_with_motorcycle_can_use_inferred_rider_context(self):
        helmet_names = {0: "Hardhat", 1: "NO-Hardhat"}
        traffic_names = {0: "person", 3: "motorcycle"}
        helmet_boxes = [_FakeDetBox(1, 0.95, (410, 150, 470, 220))]
        traffic_boxes = [_FakeDetBox(3, 0.88, (320, 430, 650, 700))]
        helmet_model = _FakeModel(_FakeResult(helmet_names, helmet_boxes))
        traffic_model = _FakeModel(_FakeResult(traffic_names, traffic_boxes))
        plate_model = _FakePlateModel([_FakeResult({0: "license_plate"}, [])])
        service = DetectionService(
            _FakeModelService(helmet_model, plate_model, traffic_model),
            _FakeOCR(),
            self.label_cfg,
        )

        _, violations = service.process_frame(self.frame, helmet_confidence=0.3, plate_confidence=0.3)

        self.assertEqual(len(violations), 1)
        self.assertTrue(violations[0]["context_inferred"])
        self.assertIsNotNone(violations[0]["rider_box"])
        self.assertIsNotNone(violations[0]["motorcycle_box"])

    def test_duplicate_no_helmet_boxes_on_same_motorcycle_create_one_violation(self):
        helmet_names = {0: "Hardhat", 1: "NO-Hardhat"}
        traffic_names = {0: "person", 3: "motorcycle"}
        helmet_boxes = [
            _FakeDetBox(1, 0.96, (410, 220, 470, 280)),
            _FakeDetBox(1, 0.91, (412, 222, 472, 282)),
        ]
        traffic_boxes = [
            _FakeDetBox(0, 0.93, (350, 180, 550, 620)),
            _FakeDetBox(3, 0.90, (320, 430, 650, 700)),
        ]
        helmet_model = _FakeModel(_FakeResult(helmet_names, helmet_boxes))
        traffic_model = _FakeModel(_FakeResult(traffic_names, traffic_boxes))
        plate_model = _FakePlateModel([_FakeResult({0: "license_plate"}, [])])
        service = DetectionService(
            _FakeModelService(helmet_model, plate_model, traffic_model),
            _FakeOCR(),
            self.label_cfg,
        )

        _, violations = service.process_frame(self.frame, helmet_confidence=0.3, plate_confidence=0.3)

        self.assertEqual(len(violations), 1)

    def test_missing_plate_uses_required_message(self):
        helmet_names = {0: "Hardhat", 1: "NO-Hardhat"}
        traffic_names = {0: "person", 3: "motorcycle"}
        helmet_boxes = [_FakeDetBox(1, 0.96, (410, 220, 470, 280))]
        traffic_boxes = [
            _FakeDetBox(0, 0.93, (350, 180, 550, 620)),
            _FakeDetBox(3, 0.90, (320, 430, 650, 700)),
        ]
        helmet_model = _FakeModel(_FakeResult(helmet_names, helmet_boxes))
        traffic_model = _FakeModel(_FakeResult(traffic_names, traffic_boxes))
        plate_model = _FakePlateModel([_FakeResult({0: "license_plate"}, [])])
        ocr = _SequenceOCR([("NOTFOUND", 0.0)])
        service = DetectionService(
            _FakeModelService(helmet_model, plate_model, traffic_model),
            ocr,
            self.label_cfg,
        )

        _, violations = service.process_frame(self.frame, helmet_confidence=0.3, plate_confidence=0.3)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["plate_number"], "NUMBER PLATE NOT FOUND")

    def test_ocr_fallback_extracts_plate_when_detector_misses_plate_box(self):
        helmet_names = {0: "Hardhat", 1: "NO-Hardhat"}
        traffic_names = {0: "person", 3: "motorcycle"}
        helmet_boxes = [_FakeDetBox(1, 0.96, (410, 220, 470, 280))]
        traffic_boxes = [
            _FakeDetBox(0, 0.93, (350, 180, 550, 620)),
            _FakeDetBox(3, 0.90, (320, 430, 650, 700)),
        ]
        helmet_model = _FakeModel(_FakeResult(helmet_names, helmet_boxes))
        traffic_model = _FakeModel(_FakeResult(traffic_names, traffic_boxes))
        plate_model = _FakePlateModel([_FakeResult({0: "license_plate"}, [])])
        ocr = _SequenceOCR([("NOTFOUND", 0.0), ("KA05MN4321", 0.81)])

        service = DetectionService(
            _FakeModelService(helmet_model, plate_model, traffic_model),
            ocr,
            self.label_cfg,
        )
        _, violations = service.process_frame(self.frame, helmet_confidence=0.3, plate_confidence=0.3)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["plate_number"], "KA05MN4321")
        self.assertTrue(violations[0]["plate_scope"].startswith("ocr_"))
        self.assertGreaterEqual(ocr.calls, 2)


if __name__ == "__main__":
    unittest.main()
