"""
TAKIM_BAGLANTI_ARAYUZU payload testleri (HY köküne göre src import).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# HY/havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
_ROOT = Path(__file__).resolve().parents[2]
_TAKIM = _ROOT / "havacilikta-yapay-zeka-yarismasi" / "TAKIM_BAGLANTI_ARAYUZU"
if _TAKIM.is_dir():
    sys.path.insert(0, str(_TAKIM))


@unittest.skipUnless((_TAKIM / "src" / "detected_object.py").is_file(), "TAKIM_BAGLANTI_ARAYUZU yok")
class TestTakimPayload(unittest.TestCase):
    def test_detected_object_payload(self):
        from src.detected_object import DetectedObject

        o = DetectedObject(
            cls=1,
            landing_status=-1,
            motion_status=-1,
            top_left_x=10.0,
            top_left_y=20.0,
            bottom_right_x=100.0,
            bottom_right_y=200.0,
        )
        base = "http://127.0.0.1:5000"
        p = o.create_payload(base)
        self.assertIn("motion_status", p)
        self.assertEqual(p["motion_status"], "-1")
        self.assertTrue(p["cls"].startswith("http://127.0.0.1:5000/classes/"))
        self.assertIn("/classes/2/", p["cls"])

    def test_frame_predictions_includes_undefined(self):
        from src.frame_predictions import FramePredictions
        from src.detected_undefined import DetectedUndefinedObject

        fp = FramePredictions(
            "http://f/1",
            "/img/x.jpg",
            "vid",
            0.0,
            0.0,
            0.0,
        )
        fp.add_detected_undefined_object(
            DetectedUndefinedObject("ref_a", 1.0, 2.0, 3.0, 4.0)
        )
        payload = fp.create_payload("http://127.0.0.1:5000/")
        self.assertIn("detected_undefined_objects", payload)
        self.assertEqual(len(payload["detected_undefined_objects"]), 1)
        self.assertEqual(payload["detected_undefined_objects"][0]["object_id"], "ref_a")


if __name__ == "__main__":
    unittest.main()
