"""
Görev 3: Referans görüntülerle ORB + homografi ile eşleme.
REFERENCES_DIR veya ./_references/ altındaki görseller kullanılır.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Task3ImageMatcher:
    def __init__(self, references_dir: Optional[str] = None):
        self.references_dir = references_dir or os.environ.get(
            "TEKNOFEST_REFERENCES_DIR",
            os.path.join(os.path.dirname(__file__), "..", "_references"),
        )
        self.references_dir = os.path.abspath(self.references_dir)
        self._refs: List[Dict[str, Any]] = []
        self._orb = cv2.ORB_create(nfeatures=2000)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._load_references()

    def _load_references(self) -> None:
        os.makedirs(self.references_dir, exist_ok=True)
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        paths: List[str] = []
        for p in patterns:
            paths.extend(glob.glob(os.path.join(self.references_dir, p)))
        for path in sorted(paths):
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self._orb.detectAndCompute(gray, None)
            if des is None:
                continue
            oid = os.path.splitext(os.path.basename(path))[0]
            self._refs.append({"id": oid, "kp": kp, "des": des, "shape": img.shape})
            logger.info("Referans yüklendi: %s", path)
        if not self._refs:
            logger.warning("Referans görüntü yok: %s", self.references_dir)

    def match(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if not self._refs:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kp2, des2 = self._orb.detectAndCompute(gray, None)
        if des2 is None:
            return []

        matches_out: List[Dict[str, Any]] = []
        h_img, w_img = gray.shape[:2]

        for ref in self._refs:
            matches = self._bf.match(ref["des"], des2)
            matches = sorted(matches, key=lambda m: m.distance)[: min(150, len(matches))]
            if len(matches) < 10:
                continue
            pts_ref = np.float32([ref["kp"][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_q = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(pts_ref, pts_q, cv2.RANSAC, 5.0)
            if H is None:
                continue
            rh, rw = ref["shape"][0], ref["shape"][1]
            corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, H)
            x1 = float(np.clip(np.min(proj[:, 0, 0]), 0, w_img))
            y1 = float(np.clip(np.min(proj[:, 0, 1]), 0, h_img))
            x2 = float(np.clip(np.max(proj[:, 0, 0]), 0, w_img))
            y2 = float(np.clip(np.max(proj[:, 0, 1]), 0, h_img))
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue
            inliers = int(mask.sum()) if mask is not None else 0
            matches_out.append(
                {
                    "object_id": ref["id"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "inliers": inliers,
                }
            )

        return matches_out
