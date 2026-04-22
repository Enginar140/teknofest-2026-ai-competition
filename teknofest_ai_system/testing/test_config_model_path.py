"""models.teknofest_model_path konfigürasyon okuması."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.config_manager import ConfigManager


class TestTeknofestModelPath(unittest.TestCase):
    def test_nested_key_empty_default(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "c.json"
            cfg_path.write_text(
                json.dumps({"models": {"teknofest_model_path": ""}}),
                encoding="utf-8",
            )
            cm = ConfigManager(str(cfg_path))
            self.assertEqual(cm.get("models.teknofest_model_path"), "")

    def test_nested_key_set(self):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "c.json"
            fake_pt = Path(d) / "m.pt"
            fake_pt.write_text("x", encoding="utf-8")
            cfg_path.write_text(
                json.dumps({"models": {"teknofest_model_path": str(fake_pt)}}),
                encoding="utf-8",
            )
            cm = ConfigManager(str(cfg_path))
            self.assertEqual(cm.get("models.teknofest_model_path"), str(fake_pt))


if __name__ == "__main__":
    unittest.main()
