import unittest

import numpy as np

from services.ocr_service import OCRService


class OCRServiceParsingTest(unittest.TestCase):
    def test_extract_candidates_supports_structured_paddle_result(self):
        result = [
            {
                "rec_texts": ["ap09 ab 1234", "junk"],
                "rec_scores": [0.97, 0.12],
            }
        ]

        candidates = OCRService._extract_candidates(result)

        self.assertEqual(candidates[0], ("AP09AB1234", 0.97))
        self.assertEqual(candidates[1], ("JUNK", 0.12))

    def test_extract_candidates_combines_multiline_structured_plate(self):
        result = [
            {
                "rec_texts": ["DL1S", "AD2271"],
                "rec_scores": [0.81, 0.96],
            }
        ]

        candidates = OCRService._extract_candidates(result)

        self.assertIn(("DL1SAD2271", 0.885), candidates)

    def test_extract_candidates_supports_legacy_tuple_result(self):
        result = [
            [
                [[[0, 0], [1, 0], [1, 1], [0, 1]], ("ka01 cd 5678", 0.91)],
            ]
        ]

        candidates = OCRService._extract_candidates(result)

        self.assertEqual(candidates, [("KA01CD5678", 0.91)])

    def test_select_best_plate_prefers_plate_like_text(self):
        candidates = [
            ("ROAD", 0.99),
            ("KA 01 AB 1234", 0.72),
            ("A", 0.95),
        ]

        text, confidence = OCRService._select_best_plate(candidates)

        self.assertEqual(text, "KA01AB1234")
        self.assertEqual(confidence, 0.72)

    def test_select_best_plate_prefers_complete_multiline_indian_plate(self):
        candidates = [
            ("DL1S", 0.86),
            ("AD2271", 0.98),
            ("DL1SAD2271", 0.82),
        ]

        text, confidence = OCRService._select_best_plate(candidates)

        self.assertEqual(text, "DL1SAD2271")
        self.assertEqual(confidence, 0.82)

    def test_read_plate_returns_not_found_when_no_candidates(self):
        service = OCRService()
        service._reader = type("_Reader", (), {"ocr": staticmethod(lambda image: [{"rec_texts": [], "rec_scores": []}] )})()

        text, conf = service.read_plate(np.zeros((20, 40, 3), dtype=np.uint8))

        self.assertEqual(text, "NOTFOUND")
        self.assertEqual(conf, 0.0)


if __name__ == "__main__":
    unittest.main()
