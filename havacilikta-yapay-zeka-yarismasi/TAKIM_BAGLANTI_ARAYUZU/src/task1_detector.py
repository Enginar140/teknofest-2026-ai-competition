"""
Görev 1: Nesne tespiti (Taşıt, İnsan, UAP, UAİ) + hareket + iniş durumu.
4 sınıflı özel YOLO ağırlığı önerilir; yoksa COCO üzerinden kısmi eşleme yapılır.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO sınıf id → HY: 0 Taşıt, 1 İnsan (UAP/UAİ için eğitilmiş model gerekir)
COCO_TO_HY = {
    0: 1,   # person
    2: 0,   # car
    3: 0,   # motorcycle
    5: 0,   # bus
    7: 0,   # truck
}


class Task1Detector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self._prev_centers: Dict[int, Tuple[float, float]] = {}
        self._track_motion: Dict[int, List[float]] = defaultdict(list)
        self._model = None
        self._nc = 0
        self._is_hy_native = False

        try:
            from ultralytics import YOLO

            self._model = YOLO(model_path)
            names = getattr(self._model, "names", None)
            self._nc = len(names) if names is not None else 0
            self._is_hy_native = self._nc == 4
            logger.info("Task1 YOLO yüklendi: %s (nc=%s hy_native=%s)", model_path, self._nc, self._is_hy_native)
        except Exception as e:
            logger.error("YOLO yüklenemedi: %s", e)
            raise

    def detect(self, image: np.ndarray, frame_index: int) -> List[Dict[str, Any]]:
        if self._model is None:
            return []

        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        out: List[Dict[str, Any]] = []
        if not results or results[0].boxes is None:
            return out

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            coco_or_hy = int(clss[i])
            hy_cls, conf = self._map_class(coco_or_hy, float(confs[i]))
            if hy_cls is None:
                continue

            oid = i
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            motion = self._motion_status(hy_cls, oid, cx, cy, frame_index)
            landing = self._landing_status(hy_cls, (x1, y1, x2, y2), xyxy, clss, i)

            out.append(
                {
                    "cls": hy_cls,
                    "landing_status": landing,
                    "motion_status": motion,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        return out

    def _map_class(self, raw_cls: int, conf: float) -> Tuple[Optional[int], float]:
        if self._is_hy_native:
            if 0 <= raw_cls <= 3:
                return raw_cls, conf
            return None, conf
        mapped = COCO_TO_HY.get(raw_cls)
        if mapped is not None:
            return mapped, conf
        return None, conf

    def _motion_status(self, hy_cls: int, oid: int, cx: float, cy: float, frame_index: int) -> int:
        # Taşıt için: 0 hareketsiz, 1 hareketli; diğerleri -1
        if hy_cls != 0:
            return -1
        key = oid
        if key not in self._prev_centers:
            self._prev_centers[key] = (cx, cy)
            return 0
        px, py = self._prev_centers[key]
        dist = float(np.hypot(cx - px, cy - py))
        self._prev_centers[key] = (cx, cy)
        self._track_motion[key].append(dist)
        if len(self._track_motion[key]) > 5:
            self._track_motion[key].pop(0)
        avg = float(np.mean(self._track_motion[key])) if self._track_motion[key] else 0.0
        # Piksel eşiği (görüntü boyutundan bağımsız kaba ölçek)
        return 1 if avg > 2.5 else 0

    def _landing_status(
        self,
        hy_cls: int,
        box: Tuple[float, float, float, float],
        all_xyxy: np.ndarray,
        all_cls: np.ndarray,
        self_idx: int,
    ) -> int:
        if hy_cls not in (2, 3):
            return -1
        x1, y1, x2, y2 = box
        pad_cx, pad_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        pad_area = max((x2 - x1) * (y2 - y1), 1.0)
        for j in range(len(all_xyxy)):
            if j == self_idx:
                continue
            ox1, oy1, ox2, oy2 = map(float, all_xyxy[j])
            inter = self._inter_area((x1, y1, x2, y2), (ox1, oy1, ox2, oy2))
            if inter / pad_area > 0.05:
                return 0
        # Basit: daire alanı yaklaşık kare bbox → en-boy oranı
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0:
            ratio = min(w, h) / max(w, h)
            if ratio < 0.65:
                return 0
        return 1

    @staticmethod
    def _inter_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)
