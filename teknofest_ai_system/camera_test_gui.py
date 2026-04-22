"""
PyQt6 Kamera Test Uygulaması
Gerçek zamanlı nesne tespiti ile kamera görüntüsü
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraWorker(QThread):
    """Kamera işleme thread'i"""
    frame_ready = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)
    
    def __init__(self, camera_id=0, model=None):
        super().__init__()
        self.camera_id = camera_id
        self.model = model
        self.running = True
        self.cap = None
    
    def run(self):
        """Thread çalıştır"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.error_signal.emit(f"Kamera {self.camera_id} açılamadı")
                return
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Nesne tespiti yap
                if self.model is not None:
                    results = self.model(frame, verbose=False)
                    frame = results[0].plot()
                
                self.frame_ready.emit(frame)
        
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if self.cap:
                self.cap.release()
    
    def stop(self):
        """Thread'i durdur"""
        self.running = False
        self.wait()

class CameraTestApp(QMainWindow):
    """Kamera test uygulaması"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.worker = None
        self.init_ui()
        self.load_model()
    
    def init_ui(self):
        """UI oluştur"""
        self.setWindowTitle("Teknofest 2026 - Kamera Testi")
        self.setGeometry(100, 100, 1200, 800)
        
        # Ana widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout
        layout = QVBoxLayout()
        
        # Kontrol paneli
        control_layout = QHBoxLayout()
        
        # Kamera seçimi
        control_layout.addWidget(QLabel("Kamera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["0 (Varsayılan)", "1", "2", "3"])
        control_layout.addWidget(self.camera_combo)
        
        # Model seçimi
        control_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n", "yolov8s"])
        control_layout.addWidget(self.model_combo)
        
        # Başlat/Durdur butonları
        self.start_btn = QPushButton("Başlat")
        self.start_btn.clicked.connect(self.start_camera)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Durdur")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        # Ekran görüntüsü
        self.screenshot_btn = QPushButton("Ekran Görüntüsü")
        self.screenshot_btn.clicked.connect(self.take_screenshot)
        self.screenshot_btn.setEnabled(False)
        control_layout.addWidget(self.screenshot_btn)
        
        layout.addLayout(control_layout)
        
        # Video gösterimi
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("border: 2px solid black; background-color: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Durum
        self.status_label = QLabel("Hazır")
        font = QFont()
        font.setPointSize(10)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)
        
        main_widget.setLayout(layout)
        
        self.current_frame = None
    
    def load_model(self):
        """Modeli yükle"""
        try:
            from ultralytics import YOLO
            
            model_path = Path("models/yolov8n.pt")
            if model_path.exists():
                logger.info(f"Model yükleniyor: {model_path}")
                self.model = YOLO(str(model_path))
                self.status_label.setText("✓ Model yüklendi")
                logger.info("✓ Model yüklendi")
            else:
                self.status_label.setText("✗ Model bulunamadı")
                logger.error(f"Model bulunamadı: {model_path}")
        
        except ImportError:
            self.status_label.setText("✗ ultralytics yüklü değil")
            logger.error("ultralytics kütüphanesi yüklü değil")
    
    def start_camera(self):
        """Kamerayı başlat"""
        camera_id = int(self.camera_combo.currentText()[0])
        
        self.worker = CameraWorker(camera_id, self.model)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.screenshot_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.status_label.setText(f"✓ Kamera {camera_id} çalışıyor...")
    
    def stop_camera(self):
        """Kamerayı durdur"""
        if self.worker:
            self.worker.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.screenshot_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.status_label.setText("Durduruldu")
    
    def on_frame_ready(self, frame):
        """Frame hazır"""
        self.current_frame = frame
        
        # Frame'i göster
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Resize et
        if w > 800:
            scale = 800 / w
            rgb_frame = cv2.resize(rgb_frame, (800, int(h * scale)))
        
        bytes_per_line = 3 * rgb_frame.shape[1]
        qt_image = QImage(rgb_frame.data, rgb_frame.shape[1], rgb_frame.shape[0],
                         bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)
    
    def on_error(self, error_msg):
        """Hata"""
        self.status_label.setText(f"✗ Hata: {error_msg}")
        self.stop_camera()
    
    def take_screenshot(self):
        """Ekran görüntüsü al"""
        if self.current_frame is not None:
            filename = f"screenshot_{len(list(Path('.').glob('screenshot_*.jpg')))}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.status_label.setText(f"✓ Kaydedildi: {filename}")
            logger.info(f"✓ Ekran görüntüsü kaydedildi: {filename}")

def main():
    """Ana fonksiyon"""
    app = QApplication(sys.argv)
    window = CameraTestApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
