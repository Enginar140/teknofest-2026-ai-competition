"""
Kamera Paneli - Gerçek zamanlı kamera/video görüntüsü ve nesne tespiti
Teknofest 2026 Havacılıkta Yapay Zeka Yarışması için 3 görevli arayüz
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QFileDialog, QMessageBox, QSpinBox, QCheckBox, QTabWidget, QProgressBar,
    QTextEdit, QSlider, QScrollArea, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
import logging
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import threading
from typing import Optional

from collections import defaultdict

from core.config_manager import get_config_manager

logger = logging.getLogger(__name__)

# COCO → HY (4 sınıflı özel model kullanıldığında eşleme devre dışı)
COCO_TO_HY = {
    0: 1,   # person → İnsan
    2: 0, 3: 0, 5: 0, 7: 0,  # car, motorcycle, bus, truck → Taşıt
}

# Teknofest Sınıfları
TEKNOFEST_CLASSES = {
    0: "Taşıt",
    1: "İnsan", 
    2: "UAP (Uçan Araba Park)",
    3: "UAİ (Uçan Ambulans İniş)"
}

TEKNOFEST_COLORS = {
    0: (255, 0, 0),      # Taşıt - Kırmızı
    1: (0, 255, 0),      # İnsan - Yeşil
    2: (255, 255, 0),    # UAP - Sarı
    3: (0, 255, 255)     # UAİ - Cyan
}

# Ana penceredeki görev kimlikleri (task1–3) ile iç mod adları
TASK_INTERNAL_ALIASES = {"task1": "detection", "task2": "position", "task3": "matching"}
TASK_MAIN_NAMES = {"detection": "task1", "position": "task2", "matching": "task3"}


def _teknofest_project_root() -> Path:
    """teknofest_ai_system paket kökü (cwd'den bağımsız)."""
    return Path(__file__).resolve().parents[2]


class CameraWorker(QThread):
    """Kamera/Video işleme thread'i - Görev bazlı işlem"""
    frame_ready = pyqtSignal(np.ndarray, dict)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    stats_update = pyqtSignal(dict)
    
    def __init__(self, source_type='camera', source_id=0, video_path=None,
                 task_mode='detection', model=None, conf_threshold=0.45,
                 camera_params=None, vo_method='orb', use_ekf=True, matcher_type='orb',
                 initial_position=None, gps_health_mode='competition', target_fps=7.5,
                 use_server_position=False, server_connection=None,
                 translation_rows=None, reference_image_path=None):
        super().__init__()
        self.source_type = source_type
        self.source_id = source_id
        self.video_path = video_path
        self.task_mode = task_mode  # 'detection', 'position', 'matching'
        self.model = model
        self.conf_threshold = conf_threshold
        self.running = True
        self.cap = None
        
        # Frame Rate Limiting (Şartname: 7.5 FPS, 2250 frame / 5 dakika)
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps  # saniye cinsinden
        self.last_frame_time = None  # Son frame işleme zamanı
        
        # Sunucu Bağlantısı (Şartnameden: translation_x, translation_y, translation_z, gps_health_status)
        self.use_server_position = use_server_position
        self.server_connection = server_connection
        self.server_position_data = None  # Sunucudan gelen son konum verisi
        self.server_position_lock = threading.Lock()
        self.translation_rows = translation_rows or []
        self.reference_image_path = reference_image_path
        
        # İstatistikler - Her frame için sıfırlanacak değerler
        self.frame_count = 0
        self.current_frame_detections = {0: 0, 1: 0, 2: 0, 3: 0}  # Bu framedeki tespitler
        self.total_detections = {0: 0, 1: 0, 2: 0, 3: 0}  # Toplam kümülatif (tracking için)
        
        # FPS hesaplama için moving average
        self.fps = 0
        self.last_fps_time = datetime.now()
        self.fps_history = []  # Son 10 frame'in FPS değeri
        self.processing_times = []  # İşleme süreleri
        
        # Basit tracking için önceki frame tespitleri
        self.prev_detections_with_id = []  # [(bbox, cls_id, object_id), ...]
        self.tracked_objects = set()  # Takip edilen nesnelerin ID'leri
        self.next_object_id = 1
        
        # Görev 2 - Pozisyon Tespiti
        self.camera_params = camera_params or {}
        self.position_estimator = None
        self.vo_method = vo_method
        self.use_ekf = use_ekf
        self.last_frame_process_time = None  # Frame rate limiting için
        self.last_position_frame_time = datetime.now()  # Position estimation için
        self.initial_position = initial_position or {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.gps_health_mode = gps_health_mode  # 'always_healthy', 'always_sick', 'competition', 'random'
        self.current_gps_health = 1  # Başlangıçta sağlıklı
        
        # Görev 3 - Görüntü Eşleme
        self.matcher_type = matcher_type
        self.image_matching_pipeline = None
        self.reference_image = None
        self.reference_features = None
        self._vehicle_motion_hist = defaultdict(list)
        self._prev_vehicle_xy = {}
    
    def _sync_translation_from_rows(self):
        """Sunucudan önceden alınmış translation listesi ile kare hizala."""
        if not self.use_server_position or not self.translation_rows:
            return
        idx = self.frame_count - 1
        if idx < 0 or idx >= len(self.translation_rows):
            return
        row = self.translation_rows[idx]

        def _num(k, default=0.0):
            v = row.get(k, default)
            if v is None:
                return float(default)
            return float(v)

        h_raw = row.get('gps_health_status', row.get('health_status', 1))
        try:
            h = int(float(str(h_raw).strip()))
        except (TypeError, ValueError):
            h = 1
        with self.server_position_lock:
            self.server_position_data = {
                'x': _num('translation_x'),
                'y': _num('translation_y'),
                'z': _num('translation_z'),
                'health': h,
            }

    def _map_raw_to_hy(self, raw_cls: int) -> Optional[int]:
        """YOLO çıkış sınıfını HY 0–3 ile eşle veya None."""
        try:
            names = getattr(self.model, 'names', {})
            nc = len(names) if names is not None else 0
        except Exception:
            nc = 0
        if nc == 4:
            if 0 <= raw_cls <= 3:
                return raw_cls
            return None
        return COCO_TO_HY.get(raw_cls)

    @staticmethod
    def _inter_area_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return float((ix2 - ix1) * (iy2 - iy1))

    def run(self):
        """Thread çalıştır"""
        try:
            # Kaynağı aç
            if self.source_type == 'camera':
                self.cap = cv2.VideoCapture(self.source_id)
            elif self.source_type == 'video':
                if self.video_path and Path(self.video_path).exists():
                    self.cap = cv2.VideoCapture(self.video_path)
                else:
                    self.error_signal.emit(f"Video bulunamadı: {self.video_path}")
                    return
            else:
                self.error_signal.emit("Geçersiz kaynak tipi")
                return
            
            if not self.cap.isOpened():
                self.error_signal.emit(f"Kaynak açılamadı: {self.source_type}")
                return
            
            # Görev 2 - Pozisyon kestiricisini başlat
            if self.task_mode == 'position':
                self._init_position_estimator()
            
            # Görev 3 - Görüntü eşleme pipeline'ını başlat
            if self.task_mode == 'matching':
                self._init_image_matching()
                if self.reference_image_path:
                    rp = Path(self.reference_image_path)
                    if rp.exists():
                        ref = cv2.imread(str(rp))
                        if ref is not None:
                            self.reference_image = ref
            
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source_type == 'video' else 0
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    if self.source_type == 'video':
                        self.finished_signal.emit()
                    break
                
                # Frame Rate Limiting (Şartname: 7.5 FPS)
                current_time = datetime.now()
                
                # İlk frame ise başlat
                if self.last_frame_process_time is None:
                    self.last_frame_process_time = current_time
                
                # Frame aralığını kontrol et
                elapsed = (current_time - self.last_frame_process_time).total_seconds()
                
                # Eğer yeterli süre geçmediyse bekle (video modunda frame atla)
                if self.source_type == 'video' and elapsed < self.frame_interval:
                    # Frame'i işlemeden atla (videoda hızlı ilerleme için)
                    import time as time_module
                    sleep_time = self.frame_interval - elapsed
                    if sleep_time > 0.001:  # 1ms'den fazla bekle
                        time_module.sleep(sleep_time)
                    continue
                
                # FPS hesapla - her işlenen frame için
                if elapsed > 0:
                    current_fps = 1.0 / elapsed
                    self.fps_history.append(current_fps)
                    if len(self.fps_history) > 10:
                        self.fps_history.pop(0)
                    self.fps = sum(self.fps_history) / len(self.fps_history)
                
                self.last_frame_process_time = current_time
                self.frame_count += 1
                self._sync_translation_from_rows()
                
                # Bu framedeki tespit sayılarını sıfırla
                self.current_frame_detections = {0: 0, 1: 0, 2: 0, 3: 0}
                
                # Göreve göre işlem yap
                detections = []
                position_data = {}
                
                if self.task_mode == 'detection' and self.model is not None:
                    # Görev 1: Nesne Tespiti
                    import time
                    process_start = time.time()
                    
                    results = self.model(frame, verbose=False, conf=self.conf_threshold)
                    
                    current_detections = []  # Bu framedeki tespitler
                    if len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes.cpu().numpy()
                        raw_dets = []
                        for box in boxes:
                            raw_cls = int(box.cls[0])
                            hy_cls = self._map_raw_to_hy(raw_cls)
                            if hy_cls is None:
                                continue
                            xyxy = box.xyxy[0].astype(float)
                            raw_dets.append({
                                'xyxy': xyxy,
                                'conf': float(box.conf[0]),
                                'hy_cls': hy_cls,
                            })
                        for idx, det in enumerate(raw_dets):
                            hy_cls = det['hy_cls']
                            xyxy = det['xyxy']
                            x1, y1, x2, y2 = map(float, xyxy)
                            bbox = tuple(map(int, (x1, y1, x2, y2)))
                            matched_id = self._match_with_previous(bbox, hy_cls)
                            self.current_frame_detections[hy_cls] += 1
                            motion_s = self._motion_for_vehicle(hy_cls, matched_id, (x1 + x2) / 2, (y1 + y2) / 2)
                            landing_s = self._landing_for_pad(hy_cls, (x1, y1, x2, y2), raw_dets, idx)
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': det['conf'],
                                'class_id': hy_cls,
                                'class_name': TEKNOFEST_CLASSES[hy_cls],
                                'object_id': matched_id,
                                'motion_status': motion_s,
                                'landing_status': landing_s,
                            }
                            detections.append(detection)
                            current_detections.append((bbox, hy_cls))
                    
                    # Tracking için güncelle
                    # ID'leri de sakla
                    self.prev_detections_with_id = [(d['bbox'], d['class_id'], d['object_id']) for d in detections]
                    
                    # İşleme süresini kaydet
                    process_time = (time.time() - process_start) * 1000  # ms
                    self.processing_times.append(process_time)
                    if len(self.processing_times) > 30:
                        self.processing_times.pop(0)
                    
                    frame = results[0].plot()
                    
                elif self.task_mode == 'position':
                    # Görev 2: Pozisyon Tespiti
                    position_data = self._estimate_position(frame)
                    frame = self._draw_position_info(frame, position_data)
                    
                elif self.task_mode == 'matching':
                    # Görev 3: Görüntü Eşleme (referans dosyası yoksa ilk kareyi referans al)
                    if self.reference_image is None:
                        self.reference_image = frame.copy()
                    detections, frame = self._match_reference_objects(frame)
                
                # İstatistikleri gönder - her frame için güncel değerler
                avg_latency = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
                current_frame_total = sum(self.current_frame_detections.values())
                
                stats = {
                    'frame_count': self.frame_count,
                    'total_frames': total_frames,
                    'fps': round(self.fps, 1),
                    'detections': current_frame_total,  # Bu framedeki tespit sayısı
                    'detection_counts': self.current_frame_detections.copy(),  # Sınıf bazında
                    'tracked_objects': len(self.tracked_objects),  # Takip edilen nesne sayısı
                    'latency': round(avg_latency, 1),
                    'source_type': self.source_type
                }
                self.stats_update.emit(stats)
                
                metadata = {
                    'detections': detections,
                    'position': position_data,
                    'frame_count': self.frame_count,
                    'total_frames': total_frames,
                    'source_type': self.source_type,
                    'task_mode': self.task_mode
                }
                
                self.frame_ready.emit(frame, metadata)
        
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if self.cap:
                self.cap.release()
    
    def _update_gps_health(self):
        """GPS health durumunu güncelle (Şartname: İlk 450 frame sağlıklı, sonra sağlıksız olabilir)"""
        if self.gps_health_mode == 'always_healthy':
            self.current_gps_health = 1
        elif self.gps_health_mode == 'always_sick':
            self.current_gps_health = 0
        elif self.gps_health_mode == 'competition':
            # Yarışma modu: İlk 450 frame (1 dakika) sağlıklı, sonra sağlıksız
            if self.frame_count <= 450:
                self.current_gps_health = 1
            else:
                self.current_gps_health = 0
        elif self.gps_health_mode == 'random':
            # Rastgele: %70 sağlıklı, %30 sağlıksız
            self.current_gps_health = 1 if np.random.random() > 0.3 else 0
        
        return self.current_gps_health
    
    def _motion_for_vehicle(self, hy_cls: int, track_id: int, cx: float, cy: float) -> int:
        if hy_cls != 0:
            return -1
        if track_id not in self._prev_vehicle_xy:
            self._prev_vehicle_xy[track_id] = (cx, cy)
            return 0
        px, py = self._prev_vehicle_xy[track_id]
        d = float(np.hypot(cx - px, cy - py))
        self._prev_vehicle_xy[track_id] = (cx, cy)
        hist = self._vehicle_motion_hist[track_id]
        hist.append(d)
        if len(hist) > 5:
            hist.pop(0)
        avg = float(np.mean(hist)) if hist else 0.0
        return 1 if avg > 2.5 else 0

    def _landing_for_pad(self, hy_cls: int, box: tuple, all_raw_dets: list, self_index: int) -> int:
        if hy_cls not in (2, 3):
            return -1
        x1, y1, x2, y2 = box
        pad_area = max((x2 - x1) * (y2 - y1), 1.0)
        for j, other in enumerate(all_raw_dets):
            if j == self_index:
                continue
            ox1, oy1, ox2, oy2 = map(float, other['xyxy'])
            inter = self._inter_area_xyxy((x1, y1, x2, y2), (ox1, oy1, ox2, oy2))
            if inter / pad_area > 0.05:
                return 0
        w, h = x2 - x1, y2 - y1
        if w > 0 and h > 0 and min(w, h) / max(w, h) < 0.65:
            return 0
        return 1
    
    def _estimate_position(self, frame):
        """Pozisyon kestirimi - Önce sunucu verisi, sonra PositionEstimator"""
        if self.use_server_position:
            server_data = self._get_server_position()
            if server_data:
                h = int(server_data.get('health', 1))
                if h == 1:
                    return {
                        'x': float(server_data['x'] + self.initial_position['x']),
                        'y': float(server_data['y'] + self.initial_position['y']),
                        'z': float(server_data['z'] + self.initial_position['z']),
                        'vx': 0.0,
                        'vy': 0.0,
                        'omega': 0.0,
                        'theta': 0.0,
                        'confidence': 1.0,
                        'health': h,
                        'gps_mode': 'server',
                        'source': 'server'
                    }
        
        # Sunucu verisi yoksa veya kullanılmıyorsa, yerel VO kullan
        if self.position_estimator is None:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'health': 0, 'confidence': 0.0, 'source': 'none'}
        
        try:
            # GPS health'i güncelle (yerel mod)
            gps_health = self._update_gps_health()
            
            # Zaman adımı hesapla
            current_time = datetime.now()
            dt = (current_time - self.last_position_frame_time).total_seconds()
            if dt <= 0:
                dt = 1.0 / self.target_fps  # Target FPS'e göre
            self.last_position_frame_time = current_time
            
            # Pozisyon kestir
            pose = self.position_estimator.process_frame(frame, dt)
            
            # Hız al
            vx, vy, omega = self.position_estimator.get_velocity()
            
            # Başlangıç konumunu ekle
            x = self.initial_position['x'] + pose.x
            y = self.initial_position['y'] + pose.y
            z = self.initial_position['z'] + pose.z if hasattr(pose, 'z') else self.initial_position['z']
            
            return {
                'x': float(x),
                'y': float(y),
                'z': float(z),
                'vx': float(vx),
                'vy': float(vy),
                'omega': float(omega),
                'theta': float(pose.theta),
                'confidence': float(pose.confidence),
                'health': gps_health,
                'gps_mode': self.gps_health_mode,
                'source': 'vo'  # Visual Odometry
            }
        except Exception as e:
            logger.error(f"Pozisyon kestirimi hatası: {e}")
            return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'health': 0, 'confidence': 0.0, 'source': 'error'}
    
    def _draw_position_info(self, frame, position_data):
        """Pozisyon bilgisi çiz"""
        x = position_data.get('x', 0)
        y = position_data.get('y', 0)
        z = position_data.get('z', 100.0)
        vx = position_data.get('vx', 0)
        vy = position_data.get('vy', 0)
        theta = position_data.get('theta', 0)
        confidence = position_data.get('confidence', 0)
        health = position_data.get('health', 1)
        source = position_data.get('source', 'unknown')  # 'server', 'vo', 'none', 'error'
        
        color = (0, 255, 0) if health == 1 else (0, 0, 255)
        
        # Kaynağa göre renk belirle
        source_color_map = {
            'server': (0, 255, 255),    # Cyan - Sunucu verisi
            'vo': (255, 0, 255),        # Magenta - Visual Odometry
            'none': (128, 128, 128),    # Gray - Veri yok
            'error': (0, 0, 255)        # Red - Hata
        }
        source_color = source_color_map.get(source, (255, 255, 0))
        
        # Hız hesapla
        speed = np.sqrt(vx**2 + vy**2)
        
        # Bilgileri çiz
        y_offset = 30
        cv2.putText(frame, f"Position: X={x:.2f}m Y={y:.2f} Z={z:.1f}m",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_offset += 25
        cv2.putText(frame, f"Speed: {speed:.2f} m/s | Heading: {np.degrees(theta):.1f}°",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y_offset += 25
        conf_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.4 else (0, 0, 255)
        source_text = {'server': 'SERVER', 'vo': 'VO', 'none': 'NONE', 'error': 'ERROR'}.get(source, '?')
        cv2.putText(frame, f"Confidence: {confidence:.2f} | GPS: {'OK' if health == 1 else 'BAD'} | Source: {source_text}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, conf_color, 2)
        
        # Yön oku çiz
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] - 100
        arrow_length = 50
        end_x = int(center_x + arrow_length * np.cos(theta))
        end_y = int(center_y - arrow_length * np.sin(theta))  # Y ekseni ters
        
        cv2.circle(frame, (center_x, center_y), 5, color, -1)
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), color, 3)
        cv2.putText(frame, "N", (center_x - 10, center_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _match_reference_objects(self, frame):
        """Referans nesne eşleştirme - ImageMatchingPipeline kullanarak"""
        detections = []
        
        if self.image_matching_pipeline is None or self.reference_image is None:
            return detections, frame
        
        try:
            # Pipeline ile frame'i işle
            result = self.image_matching_pipeline.process_frame(frame)
            
            if result and result.get('confidence', 0) > 0.3:
                # Eşleşme bulundu
                H = result.get('homography', np.eye(3))
                inlier_count = result.get('inlier_count', 0)
                confidence = result.get('confidence', 0.0)
                tracked_features = result.get('tracked_features', {})
                
                # Referans görüntüsünün köşelerini dönüştür
                h, w = self.reference_image.shape[:2]
                corners = np.float32([
                    [0, 0], [w, 0], [w, h], [0, h]
                ]).reshape(-1, 1, 2)
                
                try:
                    transformed_corners = cv2.perspectiveTransform(corners, H)
                    
                    # Detection oluştur
                    detection = {
                        'bbox': [
                            float(np.min(transformed_corners[:, 0, 0])),
                            float(np.min(transformed_corners[:, 0, 1])),
                            float(np.max(transformed_corners[:, 0, 0])),
                            float(np.max(transformed_corners[:, 0, 1]))
                        ],
                        'confidence': float(confidence),
                        'class_id': 0,
                        'class_name': 'Reference_Match',
                        'inlier_count': inlier_count,
                        'tracked_features': len(tracked_features)
                    }
                    detections.append(detection)
                    
                    # Görüntüye çiz
                    frame = self._draw_matching_result(frame, transformed_corners, confidence, inlier_count)
                except Exception as e:
                    logger.warning(f"Perspektif dönüşümü hatası: {e}")
        
        except Exception as e:
            logger.error(f"Görüntü eşleme hatası: {e}")
        
        return detections, frame
    
    def _match_with_previous(self, bbox, cls_id):
        """Önceki frame'deki tespitlerle eşleştir (basit IoU tabanlı)"""
        x1, y1, x2, y2 = bbox
        best_match = None
        best_iou = 0.3  # IoU eşiği
        
        for prev_bbox, prev_cls, prev_id in list(self.prev_detections_with_id):
            if prev_cls != cls_id:
                continue
            
            # IoU hesapla
            px1, py1, px2, py2 = prev_bbox
            inter_x1 = max(x1, px1)
            inter_y1 = max(y1, py1)
            inter_x2 = min(x2, px2)
            inter_y2 = min(y2, py2)
            
            if inter_x2 < inter_x1 or inter_y2 < inter_y1:
                continue
            
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            bbox_area = (x2 - x1) * (y2 - y1)
            prev_bbox_area = (px2 - px1) * (py2 - py1)
            union_area = bbox_area + prev_bbox_area - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_match = prev_id
        
        if best_match is not None:
            # Mevcut nesne ile eşleşti
            return best_match
        else:
            # Yeni nesne - yeni ID ata
            new_id = self.next_object_id
            self.next_object_id += 1
            self.tracked_objects.add(new_id)
            return new_id
    
    def stop(self):
        """Thread'i durdur"""
        self.running = False
        self.wait()
    
    def reset_stats(self):
        """İstatistikleri sıfırla"""
        self.frame_count = 0
        self.current_frame_detections = {0: 0, 1: 0, 2: 0, 3: 0}
        self.total_detections = {0: 0, 1: 0, 2: 0, 3: 0}
        self.fps = 0
        self.fps_history = []
        self.processing_times = []
        self.prev_detections_with_id = []
        self.tracked_objects = set()
        self.next_object_id = 1
        self.last_frame_process_time = None
        self.last_position_frame_time = datetime.now()
    
    def _init_position_estimator(self):
        """Görev 2 - Pozisyon kestiricisini başlat"""
        try:
            from teknofest_ai_system.models.position import PositionEstimator
            
            if not self.camera_params:
                logger.warning("Kamera parametreleri yüklenmedi, varsayılan değerler kullanılıyor")
                # Varsayılan kamera matrisi
                focal_length = 2792.2
                principal_point = [1988.0, 1562.2]
                image_size = [3976, 3124]
            else:
                focal_length = self.camera_params.get('focal', [2792.2, 2795.2])[0]
                principal_point = self.camera_params.get('principal', [1988.0, 1562.2])
                image_size = self.camera_params.get('image_size', [3976, 3124])
            
            # Kamera matrisi oluştur
            camera_matrix = np.array([
                [focal_length, 0, principal_point[0]],
                [0, focal_length, principal_point[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Distorsiyon katsayıları
            dist_coeffs = np.array(
                self.camera_params.get('radial', [0.0, 0.0]) +
                self.camera_params.get('tangential', [0.0, 0.0]),
                dtype=np.float32
            )
            
            # PositionEstimator oluştur
            self.position_estimator = PositionEstimator(
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                ekf_process_noise=0.1,
                ekf_measurement_noise=0.5,
            )
            
            logger.info("Pozisyon Kestirici başlatıldı")
        except Exception as e:
            logger.error(f"Pozisyon Kestirici başlatma hatası: {e}")
            self.position_estimator = None
    
    def _init_image_matching(self):
        """Görev 3 - Görüntü eşleme pipeline'ını başlat"""
        try:
            from teknofest_ai_system.models.matching import ImageMatchingPipeline
            
            # Pipeline oluştur
            self.image_matching_pipeline = ImageMatchingPipeline(
                matcher_type=self.matcher_type,
                use_kalman_tracking=True,
            )
            
            logger.info(f"Görüntü Eşleme Pipeline başlatıldı: {self.matcher_type}")
        except Exception as e:
            logger.error(f"Görüntü Eşleme Pipeline başlatma hatası: {e}")
            self.image_matching_pipeline = None
    
    def _draw_matching_result(self, frame, corners, confidence, inlier_count):
        """Eşleşme sonuçlarını görüntüye çiz"""
        try:
            # Köşeleri çiz
            corners_int = np.int32(corners)
            cv2.polylines(frame, [corners_int], True, (0, 255, 0), 3)
            
            # Bilgi yazısı
            text = f"Match: {confidence:.2f} | Inliers: {inlier_count}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            logger.warning(f"Eşleşme çizimi hatası: {e}")
        
        return frame
    
    def _on_server_message(self, message: dict):
        """Sunucudan gelen mesajları işle (Şartname: translation_x, translation_y, translation_z, gps_health_status)"""
        try:
            msg_type = message.get('type', '')
            
            if msg_type == 'POSITION_DATA' or 'translation_x' in message:
                # Sunucudan konum verisi geldi
                with self.server_position_lock:
                    self.server_position_data = {
                        'x': float(message.get('translation_x', 0.0)),
                        'y': float(message.get('translation_y', 0.0)),
                        'z': float(message.get('translation_z', 0.0)),
                        'health': int(message.get('gps_health_status', 1)),
                        'timestamp': datetime.now().timestamp()
                    }
                    logger.debug(f"Sunucu konum verisi alındı: {self.server_position_data}")
        
        except Exception as e:
            logger.error(f"Sunucu mesajı işleme hatası: {e}")
    
    def _get_server_position(self):
        """Sunucudan gelen son konum verisini getir"""
        with self.server_position_lock:
            if self.server_position_data:
                return self.server_position_data.copy()
        return None


class CameraPanel(QWidget):
    """Kamera paneli - Teknofest 2026 için 3 görevli arayüz"""
    
    camera_changed = pyqtSignal(str)
    calibration_loaded = pyqtSignal(dict)
    frame_ready = pyqtSignal(np.ndarray, dict)  # Worker'dan gelen frame'i ilet
    stats_updated = pyqtSignal(dict)  # Worker'dan gelen istatistikleri ilet
    error = pyqtSignal(str)  # Worker'dan gelen hataları ilet
    # Sol panel / diğer bileşenlerle senkron (task1, task2, task3)
    camera_task_sync = pyqtSignal(str)
    processing_state_changed = pyqtSignal(bool)  # True: çalışıyor
    
    def __init__(self):
        super().__init__()
        self.config = get_config_manager()
        self.model = None
        self.worker = None
        self.current_frame = None
        self.video_path = None
        self.current_task = 'detection'  # Varsayılan görev
        self.task3_reference_path = None  # Görev 3: isteğe bağlı referans dosyası
        
        # Resmi Teknofest Sunucu Bağlantısı
        self.teknofest_connection = None  # TeknofestConnectionHandler
        
        self.init_ui()
        self.load_model()

    def set_teknofest_connection(self, handler):
        """MainWindow / Sunucu panelinden gelen resmi API bağlantısı."""
        self.teknofest_connection = handler
        self.add_log("🔗 Teknofest sunucu oturumu kamera paneline bağlandı")

    def clear_teknofest_connection(self):
        """Sunucu oturumunu kapat (şifre/token temizlenmez; yeni login gerekir)."""
        self.teknofest_connection = None
        self.add_log("🔌 Teknofest sunucu oturumu kamera panelinden kaldırıldı")
    
    def init_ui(self) -> None:
        """UI'yi başlat - Görev bazlı düzenli arayüz"""
        # Ana layout - Dikey
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Başlık
        title = QLabel("🎥 TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # ==================== GÖREV SEÇİM ALANI ====================
        task_group = QGroupBox("📋 Görev Seçimi")
        task_layout = QHBoxLayout(task_group)
        
        self.task_buttons = {}
        tasks = [
            ('detection', '1️⃣ Nesne Tespiti', 'Taşıt, İnsan, UAP, UAİ tespiti'),
            ('position', '2️⃣ Pozisyon Tespiti', 'Hava aracı konum kestirimi'),
            ('matching', '3️⃣ Görüntü Eşleme', 'Referans nesne eşleştirme')
        ]
        
        for task_id, title, desc in tasks:
            btn = QPushButton(f"{title}\n{desc}")
            btn.setCheckable(True)
            btn.setMinimumHeight(50)
            btn.clicked.connect(lambda checked, t=task_id: self._on_camera_task_button(t))
            if task_id == 'detection':
                btn.setChecked(True)
            self.task_buttons[task_id] = btn
            task_layout.addWidget(btn)
        
        main_layout.addWidget(task_group)
        
        # ==================== ANA İÇERİK ALANI (SOL-SAĞ) ====================
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        # ==================== SOL TARAF - GÖRÜNTÜLEME ====================
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        
        # Görüntü Label
        self.video_label = QLabel("Başlatıldığında görüntü burada görünecek")
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setStyleSheet("border: 3px solid #333; background-color: black; color: white; border-radius: 5px; font-size: 14px;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(False)
        left_layout.addWidget(self.video_label, 1)
        
        # Progress Bar (video için)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(25)
        left_layout.addWidget(self.progress_bar)
        
        # Durum Bilgileri
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("⚪ Hazır")
        self.status_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px; font-weight: bold; font-size: 11px;")
        status_layout.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 5px; font-size: 11px;")
        status_layout.addWidget(self.fps_label)
        
        self.frame_label = QLabel("Frame: 0")
        self.frame_label.setStyleSheet("padding: 10px; background-color: #fff3e0; border-radius: 5px; font-size: 11px;")
        status_layout.addWidget(self.frame_label)
        
        left_layout.addLayout(status_layout)
        
        content_layout.addLayout(left_layout, 3)  # 75% genişlik
        
        # ==================== SAĞ TARAF - KONTROLLER ====================
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; }")
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # ==================== KAYNAK SEÇİM ALANI ====================
        source_group = QGroupBox("📹 Video/Kamera Kaynağı")
        source_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        source_layout = QGridLayout(source_group)
        source_layout.setSpacing(5)
        
        # Kaynak Tipi
        lbl_source = QLabel("Kaynak Tipi:")
        lbl_source.setStyleSheet("font-size: 10px;")
        source_layout.addWidget(lbl_source, 0, 0)
        self.source_type_combo = QComboBox()
        self.source_type_combo.addItems(["📷 Canlı Kamera", "🎬 Video Dosyası"])
        self.source_type_combo.currentIndexChanged.connect(self.on_source_type_changed)
        self.source_type_combo.setStyleSheet("font-size: 10px;")
        source_layout.addWidget(self.source_type_combo, 0, 1)
        
        # Kamera ID (sadece kamera modunda)
        lbl_cam_id = QLabel("Kamera ID:")
        lbl_cam_id.setStyleSheet("font-size: 10px;")
        source_layout.addWidget(lbl_cam_id, 1, 0)
        self.camera_id_spin = QSpinBox()
        self.camera_id_spin.setRange(0, 10)
        self.camera_id_spin.setValue(0)
        self.camera_id_spin.setStyleSheet("font-size: 10px;")
        source_layout.addWidget(self.camera_id_spin, 1, 1)
        
        # Video Dosyası Seçim
        lbl_video = QLabel("Video:")
        lbl_video.setStyleSheet("font-size: 10px;")
        source_layout.addWidget(lbl_video, 2, 0)
        self.video_path_label = QLabel("Seçilmedi")
        self.video_path_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border-radius: 3px; font-size: 9px;")
        source_layout.addWidget(self.video_path_label, 2, 1)
        
        self.select_video_btn = QPushButton("📂 Seç")
        self.select_video_btn.clicked.connect(self.select_video)
        self.select_video_btn.setMaximumWidth(60)
        self.select_video_btn.setStyleSheet("font-size: 9px;")
        source_layout.addWidget(self.select_video_btn, 2, 2)
        
        right_layout.addWidget(source_group)
        
        # ==================== MODEL VE AYARLAR (GÖREV 1 İÇİN) ====================
        self.task1_settings_group = QGroupBox("🔍 Nesne Tespiti Ayarları")
        self.task1_settings_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        task1_settings_layout = QGridLayout(self.task1_settings_group)
        task1_settings_layout.setSpacing(5)
        
        # Model Seçimi
        lbl_model = QLabel("Model:")
        lbl_model.setStyleSheet("font-size: 10px;")
        task1_settings_layout.addWidget(lbl_model, 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l"])
        self.model_combo.setStyleSheet("font-size: 10px;")
        task1_settings_layout.addWidget(self.model_combo, 0, 1)
        
        # Confidence Threshold
        lbl_conf = QLabel("Conf:")
        lbl_conf.setStyleSheet("font-size: 10px;")
        task1_settings_layout.addWidget(lbl_conf, 1, 0)
        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(1, 100)
        self.conf_spin.setValue(45)
        self.conf_spin.setSuffix("%")
        self.conf_spin.setStyleSheet("font-size: 10px;")
        task1_settings_layout.addWidget(self.conf_spin, 1, 1)
        
        # IoU Threshold
        lbl_iou = QLabel("IoU:")
        lbl_iou.setStyleSheet("font-size: 10px;")
        task1_settings_layout.addWidget(lbl_iou, 2, 0)
        self.iou_spin = QSpinBox()
        self.iou_spin.setRange(1, 100)
        self.iou_spin.setValue(50)
        self.iou_spin.setSuffix("%")
        self.iou_spin.setStyleSheet("font-size: 10px;")
        task1_settings_layout.addWidget(self.iou_spin, 2, 1)
        
        # Seçenekler
        self.detection_check = QCheckBox("Tespit Aktif")
        self.detection_check.setChecked(True)
        self.detection_check.setStyleSheet("font-size: 10px;")
        task1_settings_layout.addWidget(self.detection_check, 3, 0, 1, 2)
        
        self.tracking_check = QCheckBox("Tracking")
        self.tracking_check.setChecked(False)
        self.tracking_check.setStyleSheet("font-size: 10px;")
        task1_settings_layout.addWidget(self.tracking_check, 4, 0, 1, 2)
        
        right_layout.addWidget(self.task1_settings_group)
        
        # ==================== POZISYON TESPİTİ AYARLARI (GÖREV 2 İÇİN) ====================
        self.task2_settings_group = QGroupBox("📍 Pozisyon Tespiti Ayarları")
        self.task2_settings_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        task2_settings_layout = QGridLayout(self.task2_settings_group)
        task2_settings_layout.setSpacing(5)
        
        # Kamera Seçimi
        lbl_camera = QLabel("Kamera:")
        lbl_camera.setStyleSheet("font-size: 10px;")
        task2_settings_layout.addWidget(lbl_camera, 0, 0)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["RGB_2025", "Thermal_2025", "RGB_4K_2025"])
        self.camera_combo.setStyleSheet("font-size: 10px;")
        self.camera_combo.currentIndexChanged.connect(self.update_camera_properties)
        task2_settings_layout.addWidget(self.camera_combo, 0, 1)
        
        # Kamera Parametreleri Görüntüleme
        self.camera_params_label = QLabel("Focal: --- | Principal: ---")
        self.camera_params_label.setStyleSheet("font-size: 9px; color: #666; padding: 5px;")
        task2_settings_layout.addWidget(self.camera_params_label, 1, 0, 1, 2)
        
        # ==================== BAŞLANGIÇ POZİSYONU (Şartname x0=0, y0=0, z0=0) ====================
        pos_group = QGroupBox("📍 Başlangıç Pozisyonu (m)")
        pos_group.setStyleSheet("QGroupBox { font-size: 10px; }")
        pos_layout = QHBoxLayout(pos_group)
        
        self.x0_spin = QDoubleSpinBox()
        self.x0_spin.setRange(-1000, 1000)
        self.x0_spin.setValue(0.0)
        self.x0_spin.setSingleStep(0.1)
        self.x0_spin.setSuffix(" m")
        self.x0_spin.setStyleSheet("font-size: 9px;")
        pos_layout.addWidget(self.x0_spin)
        
        self.y0_spin = QDoubleSpinBox()
        self.y0_spin.setRange(-1000, 1000)
        self.y0_spin.setValue(0.0)
        self.y0_spin.setSingleStep(0.1)
        self.y0_spin.setSuffix(" m")
        self.y0_spin.setStyleSheet("font-size: 9px;")
        pos_layout.addWidget(self.y0_spin)
        
        self.z0_spin = QDoubleSpinBox()
        self.z0_spin.setRange(0, 500)
        self.z0_spin.setValue(0.0)
        self.z0_spin.setSingleStep(1.0)
        self.z0_spin.setSuffix(" m")
        self.z0_spin.setStyleSheet("font-size: 9px;")
        pos_layout.addWidget(self.z0_spin)
        
        task2_settings_layout.addWidget(pos_group, 2, 0, 1, 2)
        
        # ==================== GPS HEALTH SİMÜLASYONU ====================
        gps_group = QGroupBox("🛰️ GPS Health Simülasyonu")
        gps_group.setStyleSheet("QGroupBox { font-size: 10px; }")
        gps_layout = QHBoxLayout(gps_group)
        
        self.gps_health_sim_combo = QComboBox()
        self.gps_health_sim_combo.addItems([
            "Her Zaman Sağlıklı (1)",
            "Her Zaman Sağlıksız (0)",
            "İlk 450 frame Sağlıklı (Yarışma Modu)",
            "Rastgele"
        ])
        self.gps_health_sim_combo.setCurrentIndex(2)  # Varsayılan: Yarışma modu
        self.gps_health_sim_combo.setStyleSheet("font-size: 9px;")
        gps_layout.addWidget(self.gps_health_sim_combo)
        
        self.gps_health_label = QLabel("GPS: ---")
        self.gps_health_label.setStyleSheet("font-size: 10px; color: #4CAF50; font-weight: bold;")
        gps_layout.addWidget(self.gps_health_label)
        
        task2_settings_layout.addWidget(gps_group, 3, 0, 1, 2)
        
        # Visual Odometry Method
        lbl_vo = QLabel("VO Method:")
        lbl_vo.setStyleSheet("font-size: 10px;")
        task2_settings_layout.addWidget(lbl_vo, 4, 0)
        self.vo_method_combo = QComboBox()
        self.vo_method_combo.addItems(["ORB", "SIFT", "XoFTR"])
        self.vo_method_combo.setStyleSheet("font-size: 10px;")
        task2_settings_layout.addWidget(self.vo_method_combo, 4, 1)
        
        # EKF Aktif
        self.ekf_check = QCheckBox("EKF Aktif")
        self.ekf_check.setChecked(True)
        self.ekf_check.setStyleSheet("font-size: 10px;")
        task2_settings_layout.addWidget(self.ekf_check, 5, 0, 1, 2)
        
        right_layout.addWidget(self.task2_settings_group)
        self.task2_settings_group.setVisible(False)  # Başlangıçta gizli
        
        # ==================== GÖRÜNTÜ EŞLEME AYARLARI (GÖREV 3 İÇİN) ====================
        self.task3_settings_group = QGroupBox("🔗 Görüntü Eşleme Ayarları")
        self.task3_settings_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        task3_settings_layout = QGridLayout(self.task3_settings_group)
        task3_settings_layout.setSpacing(5)
        
        # Referans Nesne Yükleme
        lbl_ref = QLabel("Referans Nesne:")
        lbl_ref.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(lbl_ref, 0, 0)
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(["Yeni Yükle..."])
        self.ref_combo.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(self.ref_combo, 0, 1)
        
        self.load_ref_btn = QPushButton("📂 Yükle")
        self.load_ref_btn.setMaximumWidth(60)
        self.load_ref_btn.setStyleSheet("font-size: 9px;")
        self.load_ref_btn.clicked.connect(self._load_task3_reference)
        task3_settings_layout.addWidget(self.load_ref_btn, 0, 2)
        
        # Feature Matcher
        lbl_matcher = QLabel("Matcher:")
        lbl_matcher.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(lbl_matcher, 1, 0)
        self.matcher_combo = QComboBox()
        self.matcher_combo.addItems(["XoFTR", "LightGlue", "ORB", "SIFT"])
        self.matcher_combo.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(self.matcher_combo, 1, 1, 1, 2)
        
        # Match Threshold
        lbl_match_thresh = QLabel("Match Thresh:")
        lbl_match_thresh.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(lbl_match_thresh, 2, 0)
        self.match_thresh_spin = QSpinBox()
        self.match_thresh_spin.setRange(1, 100)
        self.match_thresh_spin.setValue(70)
        self.match_thresh_spin.setSuffix("%")
        self.match_thresh_spin.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(self.match_thresh_spin, 2, 1, 1, 2)
        
        # Cross-modal Matching
        self.cross_modal_check = QCheckBox("Cross-modal Matching")
        self.cross_modal_check.setChecked(False)
        self.cross_modal_check.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(self.cross_modal_check, 3, 0, 1, 3)
        
        # Ensemble Model
        self.ensemble_check = QCheckBox("Ensemble Model")
        self.ensemble_check.setChecked(False)
        self.ensemble_check.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(self.ensemble_check, 4, 0, 1, 3)
        
        # Max Eşleşme Sayısı
        lbl_max_match = QLabel("Max Match:")
        lbl_max_match.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(lbl_max_match, 5, 0)
        self.max_match_spin = QSpinBox()
        self.max_match_spin.setRange(1, 100)
        self.max_match_spin.setValue(10)
        self.max_match_spin.setStyleSheet("font-size: 10px;")
        task3_settings_layout.addWidget(self.max_match_spin, 5, 1, 1, 2)
        
        right_layout.addWidget(self.task3_settings_group)
        self.task3_settings_group.setVisible(False)  # Başlangıçta gizli
        
        # ==================== KONTROL BUTONLARI ====================
        control_layout = QVBoxLayout()
        control_layout.setSpacing(5)
        
        self.start_btn = QPushButton("▶️ Başlat")
        self.start_btn.setMinimumHeight(38)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 11px;")
        self.start_btn.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ Durdur")
        self.stop_btn.setMinimumHeight(38)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; font-size: 11px;")
        self.stop_btn.clicked.connect(self.stop_processing)
        control_layout.addWidget(self.stop_btn)
        
        self.screenshot_btn = QPushButton("📸 Ekran Görüntüsü")
        self.screenshot_btn.setMinimumHeight(35)
        self.screenshot_btn.setEnabled(False)
        self.screenshot_btn.setStyleSheet("font-size: 10px;")
        control_layout.addWidget(self.screenshot_btn)
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        
        self.export_btn = QPushButton("💾 Dışa Aktar")
        self.export_btn.setMinimumHeight(35)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("font-size: 10px;")
        control_layout.addWidget(self.export_btn)
        self.export_btn.clicked.connect(self.export_results)
        
        right_layout.addLayout(control_layout)
        
        # ==================== TESPİT ÖZETİ ====================
        detection_group = QGroupBox("🎯 Tespit Özeti")
        detection_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        detection_layout = QVBoxLayout(detection_group)
        detection_layout.setSpacing(3)
        
        self.detection_labels = {}
        for cls_id, cls_name in TEKNOFEST_CLASSES.items():
            color = TEKNOFEST_COLORS[cls_id]
            label = QLabel(f"{cls_name}: 0")
            label.setStyleSheet(f"padding: 10px; background-color: rgb({color[0]}, {color[1]}, {color[2]}); color: black; border-radius: 5px; font-weight: bold; font-size: 11px;")
            self.detection_labels[cls_id] = label
            detection_layout.addWidget(label)
        
        right_layout.addWidget(detection_group)
        
        # ==================== KAMERA KONFİGÜRASYONU ====================
        config_group = QGroupBox("📷 Kamera Konfigürasyonu")
        config_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        config_layout = QGridLayout(config_group)
        config_layout.setSpacing(5)
        
        lbl_profile = QLabel("Profil:")
        lbl_profile.setStyleSheet("font-size: 10px;")
        config_layout.addWidget(lbl_profile, 0, 0)
        self.camera_combo = QComboBox()
        cameras = self.config.get_all_cameras()
        if cameras:
            self.camera_combo.addItems(list(cameras.keys()))
        else:
            self.camera_combo.addItem("varsayılan")
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        self.camera_combo.setStyleSheet("font-size: 10px;")
        config_layout.addWidget(self.camera_combo, 0, 1)
        
        # Kamera Özellikleri Tablosu
        self.properties_table = QTableWidget()
        self.properties_table.setColumnCount(2)
        self.properties_table.setHorizontalHeaderLabels(["Özellik", "Değer"])
        self.properties_table.setMaximumHeight(100)
        self.properties_table.setStyleSheet("font-size: 10px;")
        self.update_camera_properties()
        config_layout.addWidget(self.properties_table, 1, 0, 1, 2)
        
        right_layout.addWidget(config_group)
        
        # ==================== LOG ALANI ====================
        log_group = QGroupBox("📝 Loglar")
        log_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 11px; }")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(80)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 9px;")
        log_layout.addWidget(self.log_text)
        
        right_layout.addWidget(log_group)
        
        # Sağ tarafı scroll yapılabilir hale getir
        right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        
        # Ana content layout'u ekle
        content_layout.addWidget(right_scroll, 1)  # 25% genişlik
        main_layout.addLayout(content_layout, 1)
    
    def _on_camera_task_button(self, task_internal: str) -> None:
        """Kamera sekmesindeki görev butonu — iç mod + sol panel senkronu."""
        self.set_task(task_internal)
        self.camera_task_sync.emit(TASK_MAIN_NAMES[task_internal])

    @staticmethod
    def _normalize_task_mode(task_id: str) -> str:
        """task1–3 veya detection/position/matching → iç mod adı."""
        return TASK_INTERNAL_ALIASES.get(task_id, task_id)

    def set_task(self, task_id):
        """Görev seç - Görev bazlı UI görünürlük kontrolü"""
        mode = self._normalize_task_mode(task_id)
        self.current_task = mode
        
        # Buton durumlarını güncelle
        for btn_task, btn in self.task_buttons.items():
            btn.setChecked(btn_task == mode)
        
        # Görev bazlı panel görünürlüğini kontrol et
        if mode == 'detection':
            # Görev 1: Sadece Görev 1 ayarları görünür
            self.task1_settings_group.setVisible(True)
            self.task2_settings_group.setVisible(False)
            self.task3_settings_group.setVisible(False)
            self.add_log("🎯 Nesne Tespiti modu - Taşıt, İnsan, UAP, UAİ tespiti")
            
        elif mode == 'position':
            # Görev 2: Sadece Görev 2 ayarları görünür
            self.task1_settings_group.setVisible(False)
            self.task2_settings_group.setVisible(True)
            self.task3_settings_group.setVisible(False)
            self.add_log("📌 Pozisyon Tespiti modu - Kamera kalibrasyonu ve VO kullanılacak")
            # Kamera parametrelerini güncelle
            self.update_camera_properties()
            
        elif mode == 'matching':
            # Görev 3: Sadece Görev 3 ayarları görünür
            self.task1_settings_group.setVisible(False)
            self.task2_settings_group.setVisible(False)
            self.task3_settings_group.setVisible(True)
            self.add_log("🔍 Görüntü Eşleme modu - Referans nesneler eşleştirilecek")
    
    def _load_task3_reference(self) -> None:
        """Görev 3 için referans görüntü seç (yoksa akışta ilk kare kullanılır)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Referans görüntü seçin",
            str(_teknofest_project_root()),
            "Görüntü (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Tüm dosyalar (*.*)",
        )
        if not path:
            return
        self.task3_reference_path = path
        name = Path(path).name
        self.ref_combo.clear()
        self.ref_combo.addItem(name)
        self.add_log(f"📎 Görev 3 referans: {name}")

    def on_source_type_changed(self, index):
        """Kaynak tipi değişti"""
        is_camera = (index == 0)
        self.camera_id_spin.setEnabled(is_camera)
        self.select_video_btn.setEnabled(not is_camera)
    
    def load_model(self):
        """Modeli yükle (önce TEKNOFEST_MODEL / config, sonra models/yolov8*.pt)."""
        try:
            from ultralytics import YOLO
            import os

            root = _teknofest_project_root()

            override = os.environ.get("TEKNOFEST_MODEL") or (
                self.config.get("models.teknofest_model_path") or ""
            )
            if override:
                op = Path(override)
                if not op.is_absolute():
                    op = (root / op).resolve()
                if op.exists():
                    self.add_log(f"🔄 Özel model: {op}")
                    self.model = YOLO(str(op))
                    self.add_log("✅ Teknofest model yüklendi (4 sınıf önerilir)")
                    return True

            model_name = self.model_combo.currentText()
            model_path = root / "models" / f"{model_name}.pt"
            
            if model_path.exists():
                self.add_log(f"🔄 Model yükleniyor: {model_path}")
                self.model = YOLO(str(model_path))
                self.add_log(f"✅ Model yüklendi: {model_name}")
                return True
            else:
                self.add_log(f"❌ Model bulunamadı: {model_path}")
                return False
        
        except ImportError:
            self.add_log("❌ ultralytics kütüphanesi yüklü değil")
            return False
        except Exception as e:
            self.add_log(f"❌ Model yükleme hatası: {e}")
            return False
    
    def select_video(self):
        """Video dosyası seç"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Video Dosyası Seç",
            "",
            "Video Dosyaları (*.mp4 *.avi *.mov *.mkv *.wmv);;Tüm Dosyalar (*.*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.video_path_label.setText(Path(file_path).name)
            self.add_log(f"📼 Video seçildi: {Path(file_path).name}")
    
    def start_processing(self):
        """İşlemi başlat"""
        if self.worker is not None and self.worker.isRunning():
            self.add_log("⚠️ Zaten çalışıyor — önce Durdur kullanın")
            return

        # Görev 1 için model gerekli
        if self.current_task == 'detection' and not self.load_model():
            return
        
        source_type = 'camera' if self.source_type_combo.currentIndex() == 0 else 'video'
        
        if source_type == 'video' and not self.video_path:
            self.add_log("❌ Lütfen video dosyası seçin")
            return
        
        camera_id = self.camera_id_spin.value()
        conf_threshold = self.conf_spin.value() / 100.0
        model = self.model if self.current_task == 'detection' and self.detection_check.isChecked() else None
        
        # Görev 2 - Kamera parametrelerini al
        camera_params = None
        vo_method = 'orb'
        use_ekf = True
        initial_position = None
        gps_health_mode = 'competition'
        
        if self.current_task == 'position':
            vo_method = self.vo_method_combo.currentText().lower()
            use_ekf = self.ekf_check.isChecked()
            # Kamera parametrelerini oku
            camera_name = self.camera_combo.currentText()
            camera_params = self._load_camera_calibration(camera_name)
            if not camera_params:
                self.add_log("⚠️ Kamera kalibrasyon parametreleri yüklenemedi, varsayılan kullanılacak")
            
            # Başlangıç konumunu al (Şartname: x0=0, y0=0, z0=0)
            initial_position = {
                'x': self.x0_spin.value(),
                'y': self.y0_spin.value(),
                'z': self.z0_spin.value()
            }
            
            # GPS Health modunu al
            gps_mode_index = self.gps_health_sim_combo.currentIndex()
            gps_modes = ['always_healthy', 'always_sick', 'competition', 'random']
            gps_health_mode = gps_modes[gps_mode_index]
            
            self.add_log(f"📍 Başlangıç Konumu: X={initial_position['x']:.1f}m, Y={initial_position['y']:.1f}m, Z={initial_position['z']:.1f}m")
            self.add_log(f"🛰️ GPS Mode: {self.gps_health_sim_combo.currentText()}")
        
        # Görev 3 - Matcher parametrelerini al
        matcher_type = 'orb'
        if self.current_task == 'matching':
            matcher_type = self.matcher_combo.currentText().lower()
            # Referans görüntü yükleme eklenebilir
            self.add_log(f"🔍 Matcher: {matcher_type}")
        
        # Frame Rate Limiting (Şartname: 7.5 FPS = 2250 frame / 5 dakika)
        target_fps = 7.5  # Teknofest şartnamesine göre
        
        # Sunucu Bağlantısı (Şartname: translation_x, translation_y, translation_z, gps_health_status)
        use_server_position = False
        server_connection = None
        
        translation_rows = None
        if self.teknofest_connection and self.teknofest_connection.is_authenticated:
            use_server_position = True
            translations = self.teknofest_connection.get_translations()
            if translations:
                translation_rows = translations
                self.add_log(f"🌐 Teknofest: {len(translations)} translation satırı")
            else:
                self.add_log("⚠️ Teknofest translation alınamadı (pozisyon modu VO'ya düşebilir)")
        
        # Alternatif: Özel sunucu bağlantısı
        elif hasattr(self, 'server_connection') and self.server_connection:
            from server.connection import ConnectionStatus
            if self.server_connection.connection.status == ConnectionStatus.CONNECTED:
                use_server_position = True
                server_connection = self.server_connection.connection
                self.add_log(f"🌐 Özel sunucu konum verisi kullanılacak")
        
        ref_path = None
        if self.current_task == "matching" and self.task3_reference_path:
            rp = Path(self.task3_reference_path)
            if rp.exists():
                ref_path = str(rp.resolve())

        self.worker = CameraWorker(
            source_type=source_type,
            source_id=camera_id,
            video_path=self.video_path,
            task_mode=self.current_task,
            model=model,
            conf_threshold=conf_threshold,
            camera_params=camera_params,
            vo_method=vo_method,
            use_ekf=use_ekf,
            matcher_type=matcher_type,
            initial_position=initial_position,
            gps_health_mode=gps_health_mode,
            target_fps=target_fps,
            use_server_position=use_server_position,
            server_connection=server_connection,
            translation_rows=translation_rows,
            reference_image_path=ref_path,
        )
        
        # Sunucu mesaj callback'ini bağla
        if server_connection:
            server_connection.on_message_received = self.worker._on_server_message
        
        self.add_log(f"⏱️ Target FPS: {target_fps} (Teknofest Şartnamesi)")
        self.add_log(f"📍 Konum Kaynağı: {'Sunucu' if use_server_position else 'Yerel VO'}")
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.error_signal.connect(self.on_error)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.stats_update.connect(self.on_stats_update)
        self.worker.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.screenshot_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress_bar.setVisible(source_type == 'video')
        self.processing_state_changed.emit(True)
        
        self.status_label.setText(f"🟢 {self.current_task.upper()} çalışıyor...")
        self.add_log(f"▶️ {source_type.upper()} işleme başlatıldı - Görev: {self.current_task}")
    
    def stop_processing(self):
        """İşlemi durdur"""
        if self.worker:
            self.worker.stop()
            self.worker.wait(15000)
            self.worker = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.screenshot_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.processing_state_changed.emit(False)
        
        self.status_label.setText("⏹️ Durduruldu")
        self.add_log("⏹️ İşlem durduruldu")
    
    def on_frame_ready(self, frame, metadata):
        """Frame hazır"""
        self.current_frame = frame
        
        # Frame'i göster
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        
        # Resize et
        max_width = 800
        if w > max_width:
            scale = max_width / w
            rgb_frame = cv2.resize(rgb_frame, (max_width, int(h * scale)))
        
        bytes_per_line = 3 * rgb_frame.shape[1]
        qt_image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0],
                         bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
        
        # Sinyali dışarı ilet - MainWindow için
        stats_for_main = {
            'stats': {
                'fps': self.worker.fps if self.worker else 0,
                'latency': 0,  # Basit implementasyon
                'detections': len(metadata.get('detections', [])),
                'tracked_objects': len(metadata.get('detections', []))
            }
        }
        self.frame_ready.emit(frame, stats_for_main)
    
    def on_stats_update(self, stats):
        """İstatistik güncelleme"""
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
        
        frame_count = stats['frame_count']
        total_frames = stats['total_frames']
        
        if total_frames > 0:
            progress = (frame_count / total_frames) * 100
            self.progress_bar.setValue(int(progress))
            self.frame_label.setText(f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)")
        else:
            self.frame_label.setText(f"Frame: {frame_count}")
        
        # Sinyali dışarı ilet - MainWindow için
        server_st = "Bağlı Değil"
        if self.teknofest_connection and getattr(
            self.teknofest_connection, "is_authenticated", False
        ):
            server_st = "Bağlı (Teknofest API)"

        main_stats = {
            'fps': stats.get('fps', 0.0),
            'latency': stats.get('latency', 0.0),
            'gpu_usage': 0,  # GPU kullanımı için ayrı ölçüm gerekli
            'detections': stats.get('detections', 0),  # Bu framedeki tespit sayısı
            'tracked_objects': stats.get('tracked_objects', 0),  # Takip edilen nesne sayısı
            'server_status': server_st,
        }
        self.stats_updated.emit(main_stats)
        
        # Tespit sayılarını güncelle (bu framedeki tespitler)
        if self.current_task == 'detection':
            for cls_id, count in stats.get('detection_counts', {}).items():
                cls_name = TEKNOFEST_CLASSES[cls_id]
                color = TEKNOFEST_COLORS[cls_id]
                self.detection_labels[cls_id].setText(
                    f"{cls_name}: {count}"
                )
                self.detection_labels[cls_id].setStyleSheet(
                    f"padding: 10px; background-color: rgb({color[0]}, {color[1]}, {color[2]}); color: black; border-radius: 5px; font-weight: bold;"
                )
        
        # Latency bilgisini göster (varsa)
        if 'latency' in stats:
            self.status_label.setText(f"🟢 Çalışıyor | Latency: {stats['latency']:.1f}ms")
    
    def on_error(self, error_msg):
        """Hata"""
        self.status_label.setText(f"❌ Hata: {error_msg}")
        self.add_log(f"❌ {error_msg}")
        self.stop_processing()
    
    def on_finished(self):
        """İşlem bitti"""
        self.status_label.setText("✅ Tamamlandı")
        self.add_log("✅ İşlem tamamlandı")
        self.stop_processing()
    
    def take_screenshot(self):
        """Ekran görüntüsü al"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{self.current_task}_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.add_log(f"📸 Ekran görüntüsü kaydedildi: {filename}")
            self.status_label.setText(f"📸 Kaydedildi: {filename}")
    
    def export_results(self):
        """Sonuçları dışa aktar"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{self.current_task}_{timestamp}.json"
        
        results = {
            'task': self.current_task,
            'timestamp': timestamp,
            'model': self.model_combo.currentText(),
            'conf_threshold': self.conf_spin.value() / 100.0
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.add_log(f"💾 Sonuçlar kaydedildi: {filename}")
    
    def add_log(self, message):
        """Log ekle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        logger.info(message)
    
    def update_camera_properties(self) -> None:
        """Kamera özelliklerini güncelle - Görev 2 için - Kalibrasyon dosyasından okuma"""
        # Önce properties_table'ı güncelle (eğer varsa)
        if hasattr(self, 'properties_table'):
            self.properties_table.setRowCount(0)
            
            camera_name = self.camera_combo.currentText()
            camera_config = self.config.get_camera_config(camera_name)
            
            if camera_config:
                row = 0
                for key, value in camera_config.items():
                    self.properties_table.insertRow(row)
                    
                    key_item = QTableWidgetItem(key)
                    value_item = QTableWidgetItem(str(value))
                    
                    self.properties_table.setItem(row, 0, key_item)
                    self.properties_table.setItem(row, 1, value_item)
                    
                    row += 1
        
        # Kamera kalibrasyon parametrelerini göster (Görev 2 için)
        if hasattr(self, 'camera_params_label'):
            camera_name = self.camera_combo.currentText()
            
            # Kalibrasyon dosyasını oku
            camera_params = self._load_camera_calibration(camera_name)
            
            if camera_params:
                self.camera_params_label.setText(
                    f"Focal: [{camera_params['focal'][0]:.1f}, {camera_params['focal'][1]:.1f}] | "
                    f"Principal: [{camera_params['principal'][0]:.1f}, {camera_params['principal'][1]:.1f}] | "
                    f"Size: {camera_params['image_size'][0]}x{camera_params['image_size'][1]} | "
                    f"Radial: [{camera_params['radial'][0]:.3f}, {camera_params['radial'][1]:.3f}]"
                )
                # Worker'a kamera parametrelerini aktar
                if hasattr(self, 'worker') and self.worker:
                    self.worker.camera_params = camera_params
            else:
                self.camera_params_label.setText("Focal: --- | Principal: ---")
    
    def _load_camera_calibration(self, camera_name: str) -> dict:
        """Kalibrasyon dosyasından kamera parametrelerini oku"""
        root = _teknofest_project_root()
        calib_dir = root / "config" / "kamera_kalibrasyon"
        calib_file_2025 = calib_dir / "Kamera_Kalibrasyon_Parametreleri_2025.txt"
        calib_file_2024 = calib_dir / "Kamera_Kalibrasyon_Parametreleri_2024.txt"
        
        # 2025 dosyasını dene
        if calib_file_2025.exists():
            params = self._parse_calibration_file(calib_file_2025, camera_name)
            if params:
                return params
        
        # 2024 dosyasını dene
        if calib_file_2024.exists():
            return self._parse_calibration_file(calib_file_2024, camera_name)
        
        return {}
    
    def _parse_calibration_file(self, file_path: Path, camera_name: str) -> dict:
        """Kalibrasyon dosyasını ayrıştır"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Basit ayrıştırma - dosya formatını analiz et
            camera_params = {}
            current_camera = None
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Kamera tipi tespiti
                if 'RGB Camera Intrinsics' in line:
                    current_camera = 'RGB'
                elif 'Termal Camera Intrinsics' in line:
                    current_camera = 'Thermal'
                
                # Focal Length
                if 'FocalLength:' in line and current_camera:
                    values = line.split('[')[1].split(']')[0].split()
                    camera_params['focal'] = [float(v) for v in values]
                
                # Principal Point
                elif 'PrincipalPoint:' in line and current_camera:
                    values = line.split('[')[1].split(']')[0].split()
                    camera_params['principal'] = [float(v) for v in values]
                
                # Image Size
                elif 'ImageSize:' in line and current_camera:
                    values = line.split('[')[1].split(']')[0].split()
                    camera_params['image_size'] = [int(v) for v in values]
                
                # Radial Distortion
                elif 'RadialDistortion:' in line and current_camera:
                    values = line.split('[')[1].split(']')[0].split()
                    camera_params['radial'] = [float(v) for v in values]
                
                # Tangential Distortion
                elif 'TangentialDistortion:' in line and current_camera:
                    values = line.split('[')[1].split(']')[0].split()
                    camera_params['tangential'] = [float(v) for v in values]
            
            # Kamera adını eşleştir
            camera_mapping = {
                'RGB_2025': 'RGB',
                'RGB_4K_2025': 'RGB',
                'Thermal_2025': 'Thermal'
            }
            
            if camera_name in camera_mapping:
                target = camera_mapping[camera_name]
                # RGB_2025 ve RGB_4K_2025 için aynı parametreleri kullan
                if (target == 'RGB' and 'focal' in camera_params and
                    camera_params['focal'][0] > 1000):  # RGB
                    return camera_params
                elif (target == 'Thermal' and 'focal' in camera_params and
                      camera_params['focal'][0] < 1000):  # Thermal
                    return camera_params
            
            return {}
            
        except Exception as e:
            self.add_log(f"⚠️ Kalibrasyon dosyası okuma hatası: {e}")
            return {}
    
    def on_camera_changed(self, camera_name: str) -> None:
        """Kamera değiştiğinde"""
        self.config.set_selected_camera(camera_name)
        self.update_camera_properties()
        self.camera_changed.emit(camera_name)
        self.add_log(f"📷 Kamera değiştirildi: {camera_name}")
