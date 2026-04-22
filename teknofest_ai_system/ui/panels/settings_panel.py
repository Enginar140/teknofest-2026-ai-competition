"""
Ayarlar Paneli - Sistem konfigürasyonu ve ayarlar
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QGridLayout, QPushButton,
    QComboBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging

from core.config_manager import get_config_manager

logger = logging.getLogger(__name__)


class SettingsPanel(QWidget):
    """Ayarlar paneli"""
    
    settings_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.config = get_config_manager()
        self.init_ui()
    
    def init_ui(self) -> None:
        """UI'yi başlat"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Başlık
        title = QLabel("⚙️ Sistem Ayarları")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # İşleme Ayarları
        processing_group = QGroupBox("İşleme Ayarları")
        processing_layout = QGridLayout(processing_group)
        
        processing_layout.addWidget(QLabel("Process Sayısı:"), 0, 0)
        self.num_processes_spin = QSpinBox()
        self.num_processes_spin.setRange(1, 16)
        self.num_processes_spin.setValue(self.config.get("processing.num_processes", 4))
        self.num_processes_spin.valueChanged.connect(self.on_settings_changed)
        processing_layout.addWidget(self.num_processes_spin, 0, 1)
        
        processing_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(self.config.get("processing.batch_size", 8))
        self.batch_size_spin.valueChanged.connect(self.on_settings_changed)
        processing_layout.addWidget(self.batch_size_spin, 1, 1)
        
        processing_layout.addWidget(QLabel("GPU Kullan:"), 2, 0)
        self.use_gpu_check = QCheckBox("GPU Kullan")
        self.use_gpu_check.setChecked(self.config.get("processing.use_gpu", True))
        self.use_gpu_check.stateChanged.connect(self.on_settings_changed)
        processing_layout.addWidget(self.use_gpu_check, 2, 1)
        
        main_layout.addWidget(processing_group)
        
        # Kayıt Ayarları
        recording_group = QGroupBox("Kayıt Ayarları")
        recording_layout = QGridLayout(recording_group)
        
        self.record_enabled_check = QCheckBox("Metrikleri Kaydet")
        self.record_enabled_check.setChecked(self.config.get("recording.enabled", True))
        self.record_enabled_check.stateChanged.connect(self.on_settings_changed)
        recording_layout.addWidget(self.record_enabled_check, 0, 0)
        
        self.save_frames_check = QCheckBox("Kareleri Kaydet")
        self.save_frames_check.setChecked(self.config.get("recording.save_frames", False))
        self.save_frames_check.stateChanged.connect(self.on_settings_changed)
        recording_layout.addWidget(self.save_frames_check, 0, 1)
        
        self.save_video_check = QCheckBox("Video Kaydet")
        self.save_video_check.setChecked(self.config.get("recording.save_video", True))
        self.save_video_check.stateChanged.connect(self.on_settings_changed)
        recording_layout.addWidget(self.save_video_check, 1, 0)
        
        recording_layout.addWidget(QLabel("Çıkış Dizini:"), 2, 0)
        self.output_dir_label = QLabel(self.config.get("recording.output_dir", "./results/"))
        recording_layout.addWidget(self.output_dir_label, 2, 1)
        
        browse_btn = QPushButton("📂 Dizin Seç")
        browse_btn.clicked.connect(self.browse_output_dir)
        recording_layout.addWidget(browse_btn, 2, 2)
        
        main_layout.addWidget(recording_group)
        
        # UI Ayarları
        ui_group = QGroupBox("Arayüz Ayarları")
        ui_layout = QGridLayout(ui_group)
        
        ui_layout.addWidget(QLabel("Güncelleme Aralığı (ms):"), 0, 0)
        self.update_interval_spin = QSpinBox()
        self.update_interval_spin.setRange(10, 1000)
        self.update_interval_spin.setValue(self.config.get("ui.update_interval_ms", 100))
        self.update_interval_spin.valueChanged.connect(self.on_settings_changed)
        ui_layout.addWidget(self.update_interval_spin, 0, 1)
        
        self.show_fps_check = QCheckBox("FPS Göster")
        self.show_fps_check.setChecked(self.config.get("ui.show_fps", True))
        self.show_fps_check.stateChanged.connect(self.on_settings_changed)
        ui_layout.addWidget(self.show_fps_check, 1, 0)
        
        self.show_latency_check = QCheckBox("Latency Göster")
        self.show_latency_check.setChecked(self.config.get("ui.show_latency", True))
        self.show_latency_check.stateChanged.connect(self.on_settings_changed)
        ui_layout.addWidget(self.show_latency_check, 1, 1)
        
        main_layout.addWidget(ui_group)
        
        # Butonlar
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("🔄 Varsayılana Döndür")
        reset_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(reset_btn)
        
        apply_btn = QPushButton("✓ Uygula")
        apply_btn.clicked.connect(self.apply_settings)
        button_layout.addWidget(apply_btn)
        
        save_btn = QPushButton("💾 Kaydet")
        save_btn.clicked.connect(self.save_settings)
        button_layout.addWidget(save_btn)
        
        main_layout.addLayout(button_layout)
        
        main_layout.addStretch()
    
    def on_settings_changed(self) -> None:
        """Ayarlar değiştiğinde"""
        settings = self.get_current_settings()
        self.settings_changed.emit(settings)
    
    def get_current_settings(self) -> dict:
        """Mevcut ayarları al"""
        return {
            "processing": {
                "num_processes": self.num_processes_spin.value(),
                "batch_size": self.batch_size_spin.value(),
                "use_gpu": self.use_gpu_check.isChecked()
            },
            "recording": {
                "enabled": self.record_enabled_check.isChecked(),
                "save_frames": self.save_frames_check.isChecked(),
                "save_video": self.save_video_check.isChecked(),
                "output_dir": self.output_dir_label.text()
            },
            "ui": {
                "update_interval_ms": self.update_interval_spin.value(),
                "show_fps": self.show_fps_check.isChecked(),
                "show_latency": self.show_latency_check.isChecked()
            }
        }
    
    def apply_settings(self) -> None:
        """Ayarları uygula"""
        settings = self.get_current_settings()
        
        # Konfigürasyonu güncelle
        for key, value in settings["processing"].items():
            self.config.set(f"processing.{key}", value)
        
        for key, value in settings["recording"].items():
            self.config.set(f"recording.{key}", value)
        
        for key, value in settings["ui"].items():
            self.config.set(f"ui.{key}", value)
        
        logger.info("Ayarlar uygulandı")
    
    def save_settings(self) -> None:
        """Ayarları kaydet"""
        self.apply_settings()
        self.config.save_config()
        logger.info("Ayarlar kaydedildi")
    
    def reset_to_defaults(self) -> None:
        """Varsayılanlara döndür"""
        self.num_processes_spin.setValue(4)
        self.batch_size_spin.setValue(8)
        self.use_gpu_check.setChecked(True)
        self.record_enabled_check.setChecked(True)
        self.save_frames_check.setChecked(False)
        self.save_video_check.setChecked(True)
        self.output_dir_label.setText("./results/")
        self.update_interval_spin.setValue(100)
        self.show_fps_check.setChecked(True)
        self.show_latency_check.setChecked(True)
        
        logger.info("Ayarlar varsayılana döndürüldü")
    
    def browse_output_dir(self) -> None:
        """Çıkış dizinini seç"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Çıkış Dizinini Seç",
            self.output_dir_label.text()
        )
        
        if directory:
            self.output_dir_label.setText(directory)
            self.on_settings_changed()
