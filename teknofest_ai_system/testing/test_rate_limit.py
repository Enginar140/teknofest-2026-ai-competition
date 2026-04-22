"""PredictionThrottle birim testleri (stdlib unittest)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from core.rate_limit import PredictionThrottle


class TestPredictionThrottle(unittest.TestCase):
    def test_no_sleep_under_limit(self):
        """Pencere dolmadan sleep çağrılmaz."""
        t = PredictionThrottle(max_per_minute=5)
        with patch("core.rate_limit.time.monotonic", return_value=0.0):
            with patch("core.rate_limit.time.sleep") as sm:
                for _ in range(5):
                    t.wait()
                sm.assert_not_called()

    def test_sleep_when_limit_reached(self):
        """Limit aşımında sleep çağrılır (aynı zaman damgasında birikim)."""
        t = PredictionThrottle(max_per_minute=3)
        with patch("core.rate_limit.time.monotonic", return_value=0.0):
            with patch("core.rate_limit.time.sleep") as sm:
                for _ in range(4):
                    t.wait()
                sm.assert_called()


if __name__ == "__main__":
    unittest.main()
