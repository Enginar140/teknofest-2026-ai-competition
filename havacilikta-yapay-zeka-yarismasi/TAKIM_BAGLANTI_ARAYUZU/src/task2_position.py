"""
Görev 2: Görsel tabanlı kaba pozisyon kestirimi (VO ölçekli).
GPS sağlıklı karelerde kalibrasyon ile ölçek uyumu iyileştirilir.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Task2PositionEstimator:
    def __init__(self, camera_matrix: np.ndarray):
        self.K = camera_matrix.astype(np.float64)
        self.prev_gray: Optional[np.ndarray] = None
        self.R = np.eye(3, dtype=np.float64)
        self.t = np.zeros((3, 1), dtype=np.float64)
        self.scale = 0.02
        self._orb = cv2.ORB_create(nfeatures=1500)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def calibrate(self, image: np.ndarray, gx: float, gy: float, gz: float) -> None:
        """GPS güvenilir karede birikimli hatayı azaltmak için konumu sıfırla."""
        self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        self.t = np.array([[float(gx)], [float(gy)], [float(gz)]], dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)

    def estimate(self, image: np.ndarray) -> dict:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return {"x": float(self.t[0, 0]), "y": float(self.t[1, 0]), "z": float(self.t[2, 0])}

        kp1, des1 = self._orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = self._orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            self.prev_gray = gray.copy()
            return {"x": float(self.t[0, 0]), "y": float(self.t[1, 0]), "z": float(self.t[2, 0])}

        matches = self._bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[: min(200, len(matches))]
        if len(matches) < 8:
            self.prev_gray = gray.copy()
            return {"x": float(self.t[0, 0]), "y": float(self.t[1, 0]), "z": float(self.t[2, 0])}

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.size != 9:
            self.prev_gray = gray.copy()
            return {"x": float(self.t[0, 0]), "y": float(self.t[1, 0]), "z": float(self.t[2, 0])}

        _, R, t_rel, _ = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        dz = float(t_rel[2, 0]) * self.scale * 10.0
        dx = float(t_rel[0, 0]) * self.scale
        dy = float(t_rel[1, 0]) * self.scale

        self.t[0, 0] += dx
        self.t[1, 0] += dy
        self.t[2, 0] += dz
        self.R = R @ self.R
        self.prev_gray = gray.copy()

        return {
            "x": float(self.t[0, 0]),
            "y": float(self.t[1, 0]),
            "z": float(self.t[2, 0]),
        }
