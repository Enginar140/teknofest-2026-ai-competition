"""
Dashboard Paneli - Real-time metrik gösterimi
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor
import logging

logger = logging.getLogger(__name__)


class MetricCard(QFrame):
    """Metrik kartı widget'ı"""
    
    def __init__(self, title: str, unit: str = ""):
        super().__init__()
        self.title = title
        self.unit = unit
        self.value = 0.0
        self.init_ui()
    
    def init_ui(self) -> None:
        """UI'yi başlat"""
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Başlık
        title_label = QLabel(self.title)
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Değer
        self.value_label = QLabel("0.00")
        value_font = QFont()
        value_font.setPointSize(16)
        value_font.setBold(True)
        self.value_label.setFont(value_font)
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.value_label)
        
        # Birim
        if self.unit:
            unit_label = QLabel(self.unit)
            unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(unit_label)
    
    def set_value(self, value: float) -> None:
        """Değeri ayarla"""
        self.value = value
        self.value_label.setText(f"{value:.2f}")


class DashboardPanel(QWidget):
    """Dashboard paneli"""
    
    metrics_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_timers()
    
    def init_ui(self) -> None:
        """UI'yi başlat"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Başlık
        title = QLabel("📊 Real-time Metrikler")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Metrik kartları
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(10)
        
        # Görev 1 Metrikleri
        self.map_card = MetricCard("mAP@0.5", "")
        self.task1_latency_card = MetricCard("Latency (T1)", "ms")
        metrics_layout.addWidget(self.map_card, 0, 0)
        metrics_layout.addWidget(self.task1_latency_card, 0, 1)
        
        # Görev 2 Metrikleri
        self.rmse_card = MetricCard("RMSE", "m")
        self.task2_latency_card = MetricCard("Latency (T2)", "ms")
        metrics_layout.addWidget(self.rmse_card, 1, 0)
        metrics_layout.addWidget(self.task2_latency_card, 1, 1)
        
        # Görev 3 Metrikleri
        self.matching_score_card = MetricCard("Eşleşme Skoru", "")
        self.task3_latency_card = MetricCard("Latency (T3)", "ms")
        metrics_layout.addWidget(self.matching_score_card, 2, 0)
        metrics_layout.addWidget(self.task3_latency_card, 2, 1)
        
        # Sistem Metrikleri
        self.fps_card = MetricCard("FPS", "fps")
        self.gpu_card = MetricCard("GPU", "%")
        metrics_layout.addWidget(self.fps_card, 3, 0)
        metrics_layout.addWidget(self.gpu_card, 3, 1)
        
        main_layout.addLayout(metrics_layout)
        
        # İlerleme çubukları
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(5)
        
        # Görev 1 İlerleme
        progress_layout.addWidget(QLabel("Görev 1 İlerleme:"))
        self.task1_progress = QProgressBar()
        self.task1_progress.setValue(0)
        progress_layout.addWidget(self.task1_progress)
        
        # Görev 2 İlerleme
        progress_layout.addWidget(QLabel("Görev 2 İlerleme:"))
        self.task2_progress = QProgressBar()
        self.task2_progress.setValue(0)
        progress_layout.addWidget(self.task2_progress)
        
        # Görev 3 İlerleme
        progress_layout.addWidget(QLabel("Görev 3 İlerleme:"))
        self.task3_progress = QProgressBar()
        self.task3_progress.setValue(0)
        progress_layout.addWidget(self.task3_progress)
        
        main_layout.addLayout(progress_layout)
        
        # Durum bilgisi
        self.status_label = QLabel("Durum: Hazır")
        main_layout.addWidget(self.status_label)
        
        main_layout.addStretch()
    
    def setup_timers(self) -> None:
        """Zamanlayıcıları oluştur"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(100)  # 100ms
    
    def update_metrics(self) -> None:
        """Metrikleri güncelle (simülasyon)"""
        # Burada gerçek metrikler güncellenecek
        pass
    
    def set_metrics(self, metrics: dict) -> None:
        """Metrikleri ayarla"""
        if "task1_map" in metrics:
            self.map_card.set_value(metrics["task1_map"])
        if "task1_latency" in metrics:
            self.task1_latency_card.set_value(metrics["task1_latency"])
        if "task2_rmse" in metrics:
            self.rmse_card.set_value(metrics["task2_rmse"])
        if "task2_latency" in metrics:
            self.task2_latency_card.set_value(metrics["task2_latency"])
        if "task3_score" in metrics:
            self.matching_score_card.set_value(metrics["task3_score"])
        if "task3_latency" in metrics:
            self.task3_latency_card.set_value(metrics["task3_latency"])
        if "fps" in metrics:
            self.fps_card.set_value(metrics["fps"])
        if "gpu_usage" in metrics:
            self.gpu_card.set_value(metrics["gpu_usage"])
        
        # İlerleme çubukları
        if "task1_progress" in metrics:
            self.task1_progress.setValue(int(metrics["task1_progress"]))
        if "task2_progress" in metrics:
            self.task2_progress.setValue(int(metrics["task2_progress"]))
        if "task3_progress" in metrics:
            self.task3_progress.setValue(int(metrics["task3_progress"]))
        
        # Durum
        if "status" in metrics:
            self.status_label.setText(f"Durum: {metrics['status']}")
