"""
TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması
Ana Model Entegrasyon Dosyası

3 Görev:
  1. Nesne Tespiti (Taşıt, İnsan, UAP, UAİ) + Hareket/İniş durumu
  2. Pozisyon Kestirimi (Visual Odometry + Kalman Filter)
  3. Görüntü Eşleme (SIFT + RANSAC feature matching)
"""

import logging
import time
import os
import cv2
import numpy as np
import requests

from .detected_object import DetectedObject
from .detected_translation import DetectedTranslation
from .detected_undefined import DetectedUndefinedObject

# Görev modüllerini import et
from .task1_detector import Task1Detector
from .task2_position import Task2PositionEstimator
from .task3_matching import Task3ImageMatcher


class ObjectDetectionModel:
    """
    Ana model sınıfı — 3 görevi koordine eder.
    Resmi TAKIM_BAGLANTI_ARAYUZU ile uyumlu process/detect interface.
    """

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaluation_server = evaluation_server_url

        # ===== GÖREV 1: Nesne Tespiti =====
        self.task1 = Task1Detector(
            model_path=self._find_model_path(),
            confidence_threshold=0.25,
            iou_threshold=0.45
        )

        # ===== GÖREV 2: Pozisyon Kestirimi =====
        # RGB Kamera parametreleri (2025 kalibrasyonundan)
        camera_matrix = np.array([
            [2792.2, 0, 1988.0],
            [0, 2795.2, 1562.2],
            [0, 0, 1]
        ], dtype=np.float64)
        self.task2 = Task2PositionEstimator(camera_matrix=camera_matrix)

        # ===== GÖREV 3: Görüntü Eşleme =====
        self.task3 = Task3ImageMatcher()

        # Frame sayacı
        self.frame_count = 0

        logging.info('All 3 task modules initialized successfully')

    def _find_model_path(self):
        """En iyi mevcut YOLO model dosyasını bul"""
        # Öncelik sırası: yolov8l > yolov8m > yolov8s > yolov8n
        search_paths = [
            # Proje kök dizininden
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models'),
            os.path.join(os.path.dirname(__file__), '..', '..', '..'),
            os.path.join(os.path.dirname(__file__), '..', 'models'),
            '.',
        ]
        model_priority = ['yolov8l.pt', 'yolov8m.pt', 'yolov8s.pt', 'yolov8n.pt']

        for model_name in model_priority:
            for search_dir in search_paths:
                full_path = os.path.join(search_dir, model_name)
                if os.path.exists(full_path):
                    logging.info(f'Found model: {full_path}')
                    return os.path.abspath(full_path)

        # Fallback: ultralytics indirsin
        logging.warning('No local model found, will use ultralytics auto-download: yolov8n.pt')
        return 'yolov8n.pt'

    @staticmethod
    def download_image(img_url, images_folder, images_files, retries=3, initial_wait_time=0.1):
        """Görüntü karesi indir"""
        t1 = time.perf_counter()
        wait_time = initial_wait_time
        image_name = img_url.split("/")[-1]

        if image_name not in images_files:
            for attempt in range(retries):
                try:
                    response = requests.get(img_url, timeout=60)
                    response.raise_for_status()

                    img_bytes = response.content
                    with open(os.path.join(images_folder, image_name), 'wb') as img_file:
                        img_file.write(img_bytes)

                    t2 = time.perf_counter()
                    logging.info(f'{img_url} - Download Finished in {t2 - t1:.2f}s')
                    return os.path.join(images_folder, image_name)

                except requests.exceptions.RequestException as e:
                    logging.error(f"Download failed for {img_url} attempt {attempt + 1}: {e}")
                    time.sleep(wait_time)
                    wait_time *= 2

            logging.error(f"Failed to download {img_url} after {retries} attempts.")
            return None
        else:
            return os.path.join(images_folder, image_name)

    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files):
        """
        Ana işleme fonksiyonu — her frame için çağrılır.
        3 görevi sırasıyla çalıştırır.
        """
        self.frame_count += 1
        t_start = time.perf_counter()

        # 1) Görüntüyü indir
        img_path = self.download_image(
            evaluation_server_url + "media" + prediction.image_url,
            images_folder, images_files
        )

        # 2) Görüntüyü oku
        image = None
        if img_path and os.path.exists(img_path):
            image = cv2.imread(img_path)

        if image is None:
            logging.warning(f'Frame {self.frame_count}: Image could not be loaded, sending defaults')
            self._send_defaults(prediction, health_status)
            return prediction

        # 3) Üç görevi çalıştır
        frame_results = self.detect(prediction, health_status, image)

        t_end = time.perf_counter()
        logging.info(f'Frame {self.frame_count}: Total processing time: {(t_end - t_start)*1000:.1f}ms')

        return frame_results

    def detect(self, prediction, health_status, image):
        """
        3 görevi çalıştır ve sonuçları prediction nesnesine ekle.
        """

        # ============================================================
        # GÖREV 1: Nesne Tespiti + Hareket Durumu + İniş Durumu
        # ============================================================
        try:
            detections = self.task1.detect(image, self.frame_count)

            for det in detections:
                d_obj = DetectedObject(
                    int(det["cls"]),
                    int(det["landing_status"]),
                    int(det["motion_status"]),
                    float(det["x1"]),
                    float(det["y1"]),
                    float(det["x2"]),
                    float(det["y2"]),
                )
                prediction.add_detected_object(d_obj)

            logging.info(f'Frame {self.frame_count}: Task1 detected {len(detections)} objects')

        except Exception as e:
            logging.error(f'Frame {self.frame_count}: Task1 error: {e}')

        # ============================================================
        # GÖREV 2: Pozisyon Kestirimi
        # ============================================================
        try:
            if health_status == '0':
                # GPS çalışmıyor → kendi tahminimizi gönder
                pos = self.task2.estimate(image)
                pred_x = pos['x']
                pred_y = pos['y']
                pred_z = pos['z']
            else:
                # GPS sağlıklı → sunucu değerini geri gönder + kalibrasyon
                pred_x = float(prediction.gt_translation_x)
                pred_y = float(prediction.gt_translation_y)
                pred_z = float(prediction.gt_translation_z)

                # GPS sağlıklıyken kalibrasyon yap (VO scale öğrenme)
                self.task2.calibrate(
                    image,
                    pred_x, pred_y, pred_z
                )

            logging.info(f'Frame {self.frame_count}: Task2 position=({pred_x}, {pred_y}, {pred_z}) health={health_status}')

        except Exception as e:
            logging.error(f'Frame {self.frame_count}: Task2 error: {e}')
            pred_x = float(prediction.gt_translation_x)
            pred_y = float(prediction.gt_translation_y)
            pred_z = float(prediction.gt_translation_z)

        trans_obj = DetectedTranslation(float(pred_x), float(pred_y), float(pred_z))
        prediction.add_translation_object(trans_obj)

        # ============================================================
        # GÖREV 3: Görüntü Eşleme (referans nesneler → detected_undefined_objects)
        # ============================================================
        try:
            matches = self.task3.match(image)
            for m in matches:
                u = DetectedUndefinedObject(
                    str(m["object_id"]),
                    float(m["x1"]),
                    float(m["y1"]),
                    float(m["x2"]),
                    float(m["y2"]),
                )
                prediction.add_detected_undefined_object(u)
            if matches:
                logging.info(f'Frame {self.frame_count}: Task3 found {len(matches)} matches')
        except Exception as e:
            logging.error(f'Frame {self.frame_count}: Task3 error: {e}')

        return prediction

    def _send_defaults(self, prediction, health_status):
        """Görüntü yüklenemediğinde varsayılan değerler gönder"""
        pred_x = float(prediction.gt_translation_x)
        pred_y = float(prediction.gt_translation_y)
        pred_z = float(prediction.gt_translation_z)
        trans_obj = DetectedTranslation(pred_x, pred_y, pred_z)
        prediction.add_translation_object(trans_obj)