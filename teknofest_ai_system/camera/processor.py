"""
Kamera Entegrasyonu ve Canlı İşleme Modülü
Kamera girişi, frame işleme ve tüm modüllerin entegrasyonu
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CameraSource(Enum):
    """Kamera kaynakları"""
    WEBCAM = "webcam"
    VIDEO_FILE = "video_file"
    RTSP_STREAM = "rtsp_stream"
    IMAGE_SEQUENCE = "image_sequence"


@dataclass
class CameraConfig:
    """Kamera konfigürasyonu"""
    source_type: CameraSource = CameraSource.WEBCAM
    source_path: str = "0"  # Webcam ID veya dosya yolu
    
    # Çözünürlük
    width: int = 1280
    height: int = 720
    fps: int = 30
    
    # Kamera parametreleri
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    exposure: int = -5
    
    # İşleme
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotate_90: bool = False
    
    # Buffer
    buffer_size: int = 1


class FrameBuffer:
    """
    Frame Buffer
    Thread-safe frame depolama
    """
    
    def __init__(self, max_size: int = 1):
        """
        Frame buffer başlat
        
        Args:
            max_size: Maksimum buffer boyutu
        """
        self.max_size = max_size
        self.frames = []
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
    
    def put(self, frame: np.ndarray, frame_id: int, timestamp: float):
        """
        Frame ekle
        
        Args:
            frame: Görüntü frame'i
            frame_id: Frame ID'si
            timestamp: Zaman damgası
        """
        with self.not_empty:
            # Buffer dolu ise en eski frame'i sil
            if len(self.frames) >= self.max_size:
                self.frames.pop(0)
            
            self.frames.append({
                'frame': frame,
                'frame_id': frame_id,
                'timestamp': timestamp,
            })
            
            self.not_empty.notify_all()
    
    def get(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Frame al
        
        Args:
            timeout: Bekleme timeout'u
            
        Returns:
            Frame dict'i veya None
        """
        with self.not_empty:
            if not self.frames:
                self.not_empty.wait(timeout=timeout)
            
            if self.frames:
                return self.frames.pop(0)
            
            return None
    
    def clear(self):
        """Buffer'ı temizle"""
        with self.lock:
            self.frames.clear()
    
    def size(self) -> int:
        """Buffer boyutu"""
        with self.lock:
            return len(self.frames)


class CameraCapture:
    """
    Kamera Yakalama
    Kameradan frame'leri okur
    """
    
    def __init__(
        self,
        config: CameraConfig,
        on_frame: Optional[Callable[[np.ndarray, int, float], None]] = None,
    ):
        """
        Kamera yakalama başlat
        
        Args:
            config: Kamera konfigürasyonu
            on_frame: Frame callback'i
        """
        self.config = config
        self.on_frame = on_frame
        
        self.cap = None
        self.frame_id = 0
        self.start_time = 0
        
        # Thread kontrol
        self.capture_thread = None
        self.running = False
        
        # İstatistikler
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_frame_time = 0
        
        logger.info(f"CameraCapture başlatıldı: {config.source_type.value}")
    
    def _open_camera(self) -> bool:
        """Kamerayı aç"""
        try:
            if self.config.source_type == CameraSource.WEBCAM:
                # Webcam
                camera_id = int(self.config.source_path)
                self.cap = cv2.VideoCapture(camera_id)
            
            elif self.config.source_type == CameraSource.VIDEO_FILE:
                # Video dosyası
                self.cap = cv2.VideoCapture(self.config.source_path)
            
            elif self.config.source_type == CameraSource.RTSP_STREAM:
                # RTSP stream
                self.cap = cv2.VideoCapture(self.config.source_path)
            
            else:
                logger.error(f"Bilinmeyen kamera tipi: {self.config.source_type}")
                return False
            
            if not self.cap.isOpened():
                logger.error("Kamera açılamadı")
                return False
            
            # Kamera parametrelerini ayarla
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Kamera özellikleri
            if self.config.brightness != 0:
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
            if self.config.contrast != 0:
                self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            if self.config.saturation != 0:
                self.cap.set(cv2.CAP_PROP_SATURATION, self.config.saturation)
            if self.config.exposure != 0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
            
            logger.info("Kamera açıldı")
            return True
            
        except Exception as e:
            logger.error(f"Kamera açma hatası: {e}")
            return False
    
    def start(self) -> bool:
        """Kamera yakalamayı başlat"""
        if not self._open_camera():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self.capture_thread.start()
        
        logger.info("Kamera yakalama başlatıldı")
        return True
    
    def stop(self):
        """Kamera yakalamayı durdur"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        logger.info("Kamera yakalama durduruldu")
    
    def _capture_loop(self):
        """Yakalama döngüsü"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.warning("Frame okunamadı")
                    self.frames_dropped += 1
                    continue
                
                # Frame işleme
                frame = self._process_frame(frame)
                
                # Timestamp
                timestamp = time.time() - self.start_time
                
                # Callback
                if self.on_frame:
                    self.on_frame(frame, self.frame_id, timestamp)
                
                self.frames_captured += 1
                self.frame_id += 1
                self.last_frame_time = time.time()
                
            except Exception as e:
                logger.error(f"Capture loop hatası: {e}")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Frame'i işle"""
        # Flip
        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.config.flip_vertical:
            frame = cv2.flip(frame, 0)
        
        # Rotate
        if self.config.rotate_90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        return frame
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri al"""
        return {
            'frames_captured': self.frames_captured,
            'frames_dropped': self.frames_dropped,
            'last_frame_time': self.last_frame_time,
            'fps': self.frames_captured / max(time.time() - self.start_time, 0.001),
        }


class FrameProcessor:
    """
    Frame İşleyici
    Kameradan gelen frame'leri işler ve AI modüllerine gönderir
    """
    
    def __init__(
        self,
        camera_config: CameraConfig,
        on_processed_frame: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Frame işleyici başlat
        
        Args:
            camera_config: Kamera konfigürasyonu
            on_processed_frame: İşlenmiş frame callback'i
        """
        self.camera_config = camera_config
        self.on_processed_frame = on_processed_frame
        
        # Kamera yakalama
        self.camera = CameraCapture(
            camera_config,
            on_frame=self._on_frame_captured,
        )
        
        # Frame buffer
        self.frame_buffer = FrameBuffer(max_size=camera_config.buffer_size)
        
        # İşleme thread'i
        self.process_thread = None
        self.running = False
        
        # İstatistikler
        self.frames_processed = 0
        self.processing_times = []
        
        logger.info("FrameProcessor başlatıldı")
    
    def _on_frame_captured(self, frame: np.ndarray, frame_id: int, timestamp: float):
        """Kameradan frame alındı"""
        self.frame_buffer.put(frame, frame_id, timestamp)
    
    def start(self) -> bool:
        """Frame işlemeyi başlat"""
        if not self.camera.start():
            return False
        
        self.running = True
        self.process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True,
        )
        self.process_thread.start()
        
        logger.info("Frame işleme başlatıldı")
        return True
    
    def stop(self):
        """Frame işlemeyi durdur"""
        self.running = False
        self.camera.stop()
        
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
        
        logger.info("Frame işleme durduruldu")
    
    def _process_loop(self):
        """İşleme döngüsü"""
        while self.running:
            try:
                # Buffer'dan frame al
                frame_data = self.frame_buffer.get(timeout=1.0)
                
                if frame_data is None:
                    continue
                
                frame = frame_data['frame']
                frame_id = frame_data['frame_id']
                timestamp = frame_data['timestamp']
                
                # İşleme zamanını ölç
                start_time = time.time()
                
                # Frame'i işle
                processed_data = {
                    'frame': frame,
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'height': frame.shape[0],
                    'width': frame.shape[1],
                }
                
                # Callback
                if self.on_processed_frame:
                    self.on_processed_frame(processed_data)
                
                # İstatistikler
                processing_time = (time.time() - start_time) * 1000  # ms
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                self.frames_processed += 1
                
            except Exception as e:
                logger.error(f"Process loop hatası: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri al"""
        camera_stats = self.camera.get_stats()
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        return {
            **camera_stats,
            'frames_processed': self.frames_processed,
            'avg_processing_time_ms': avg_processing_time,
            'buffer_size': self.frame_buffer.size(),
        }


class LiveProcessor:
    """
    Canlı İşleyici
    Kameradan gelen frame'leri gerçek zamanlı olarak işler
    """
    
    def __init__(
        self,
        camera_config: CameraConfig,
        detector: Optional[Any] = None,
        position_estimator: Optional[Any] = None,
        matcher: Optional[Any] = None,
        connection_manager: Optional[Any] = None,
        performance_monitor: Optional[Any] = None,
    ):
        """
        Canlı işleyici başlat
        
        Args:
            camera_config: Kamera konfigürasyonu
            detector: Nesne tespiti modülü
            position_estimator: Pozisyon kestirimi modülü
            matcher: Görüntü eşleme modülü
            connection_manager: Sunucu bağlantı yöneticisi
            performance_monitor: Performans izleyici
        """
        self.camera_config = camera_config
        self.detector = detector
        self.position_estimator = position_estimator
        self.matcher = matcher
        self.connection_manager = connection_manager
        self.performance_monitor = performance_monitor
        
        # Frame işleyici
        self.frame_processor = FrameProcessor(
            camera_config,
            on_processed_frame=self._process_frame,
        )
        
        # Kontrol
        self.running = False
        
        logger.info("LiveProcessor başlatıldı")
    
    def _process_frame(self, frame_data: Dict[str, Any]):
        """Frame'i işle"""
        frame = frame_data['frame']
        frame_id = frame_data['frame_id']
        timestamp = frame_data['timestamp']
        
        # Timing başlat
        frame_start_time = time.time()
        
        try:
            # Sunucuya frame başlangıcını bildir
            if self.connection_manager:
                self.connection_manager.send_frame_start(frame_id)
            
            # Nesne tespiti
            detections = []
            if self.detector:
                inference_start = time.time()
                detections = self.detector.process_frame(frame)
                inference_time = (time.time() - inference_start) * 1000
                
                if self.performance_monitor:
                    self.performance_monitor.record_inference(inference_time)
                    self.performance_monitor.record_detection(len(detections))
                
                # Sunucuya tespitleri gönder
                if self.connection_manager and detections:
                    from teknofest_ai_system.server import DetectionResult
                    detection_results = []
                    for det in detections:
                        h, w = frame.shape[:2]
                        # Normalize bbox
                        x1, y1, x2, y2 = det.bbox
                        x_center = (x1 + x2) / (2 * w)
                        y_center = (y1 + y2) / (2 * h)
                        width = (x2 - x1) / w
                        height = (y2 - y1) / h
                        
                        detection_results.append(DetectionResult(
                            class_id=det.class_id,
                            class_name=det.class_name,
                            confidence=det.confidence,
                            bbox=[x_center, y_center, width, height],
                            track_id=det.track_id,
                        ))
                    
                    self.connection_manager.send_detection(frame_id, detection_results)
            
            # Pozisyon kestirimi
            if self.position_estimator:
                position = self.position_estimator.process_frame(frame, dt=1.0/self.camera_config.fps)
                
                if self.performance_monitor and position:
                    self.performance_monitor.record_position_confidence(position.confidence)
                
                # Sunucuya pozisyonu gönder
                if self.connection_manager and position:
                    from teknofest_ai_system.server import PositionResult
                    position_result = PositionResult(
                        x=position.x,
                        y=position.y,
                        theta=position.theta,
                        confidence=position.confidence,
                        timestamp=timestamp,
                    )
                    self.connection_manager.send_position(frame_id, position_result)
            
            # Görüntü eşleme
            if self.matcher:
                matching_result = self.matcher.process_frame(frame)
            
            # Frame bitiş zamanını hesapla
            frame_elapsed = (time.time() - frame_start_time) * 1000
            
            # Sunucuya frame bitiş mesajı gönder
            if self.connection_manager:
                self.connection_manager.send_frame_end(frame_id, frame_elapsed)
            
            # Performans metrikleri
            if self.performance_monitor:
                self.performance_monitor.record_frame(frame_elapsed)
            
        except Exception as e:
            logger.error(f"Frame işleme hatası: {e}")
    
    def start(self) -> bool:
        """Canlı işlemeyi başlat"""
        if not self.frame_processor.start():
            return False
        
        self.running = True
        logger.info("Canlı işleme başlatıldı")
        return True
    
    def stop(self):
        """Canlı işlemeyi durdur"""
        self.running = False
        self.frame_processor.stop()
        logger.info("Canlı işleme durduruldu")
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri al"""
        return self.frame_processor.get_stats()


class CameraManager:
    """
    Kamera Yöneticisi
    Kamera ve canlı işleme yönetimi
    """
    
    def __init__(self):
        """Kamera yöneticisi başlat"""
        self.live_processor: Optional[LiveProcessor] = None
        self.running = False
        
        logger.info("CameraManager başlatıldı")
    
    def initialize(
        self,
        camera_config: CameraConfig,
        detector: Optional[Any] = None,
        position_estimator: Optional[Any] = None,
        matcher: Optional[Any] = None,
        connection_manager: Optional[Any] = None,
        performance_monitor: Optional[Any] = None,
    ) -> bool:
        """
        Kamera yöneticisini başlat
        
        Args:
            camera_config: Kamera konfigürasyonu
            detector: Nesne tespiti modülü
            position_estimator: Pozisyon kestirimi modülü
            matcher: Görüntü eşleme modülü
            connection_manager: Sunucu bağlantı yöneticisi
            performance_monitor: Performans izleyici
            
        Returns:
            Başarılı mı?
        """
        try:
            self.live_processor = LiveProcessor(
                camera_config=camera_config,
                detector=detector,
                position_estimator=position_estimator,
                matcher=matcher,
                connection_manager=connection_manager,
                performance_monitor=performance_monitor,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Kamera yöneticisi başlatma hatası: {e}")
            return False
    
    def start(self) -> bool:
        """Kamera işlemeyi başlat"""
        if not self.live_processor:
            logger.error("Kamera yöneticisi başlatılmamış")
            return False
        
        if not self.live_processor.start():
            return False
        
        self.running = True
        logger.info("Kamera işleme başlatıldı")
        return True
    
    def stop(self):
        """Kamera işlemeyi durdur"""
        if self.live_processor:
            self.live_processor.stop()
        
        self.running = False
        logger.info("Kamera işleme durduruldu")
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri al"""
        if self.live_processor:
            return self.live_processor.get_stats()
        return {}
    
    @property
    def is_running(self) -> bool:
        """Çalışıyor mu?"""
        return self.running
