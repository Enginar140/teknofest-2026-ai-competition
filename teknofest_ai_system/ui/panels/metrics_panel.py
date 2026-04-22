"""
Metrikleme Paneli - Real-time metrik takibi ve kayıt
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QCheckBox, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import logging
import csv
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsPanel(QWidget):
    """Metrikleme paneli"""
    
    metrics_saved = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.metrics_history = []
        self.init_ui()
    
    def init_ui(self) -> None:
        """UI'yi başlat"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Başlık
        title = QLabel("📈 Metrikleme")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Kontrol Butonları
        control_layout = QHBoxLayout()
        
        self.record_check = QCheckBox("Metrikleri Kaydet")
        self.record_check.setChecked(True)
        control_layout.addWidget(self.record_check)
        
        export_btn = QPushButton("📥 CSV'ye Aktar")
        export_btn.clicked.connect(self.export_to_csv)
        control_layout.addWidget(export_btn)
        
        clear_btn = QPushButton("🗑️ Temizle")
        clear_btn.clicked.connect(self.clear_metrics)
        control_layout.addWidget(clear_btn)
        
        main_layout.addLayout(control_layout)
        
        # Metrikleme Tablosu
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(10)
        self.metrics_table.setHorizontalHeaderLabels([
            "Kare", "Zaman", "mAP", "RMSE", "Eşleşme", 
            "FPS", "Latency", "GPU", "RAM", "Durum"
        ])
        main_layout.addWidget(self.metrics_table)
        
        main_layout.addStretch()
    
    def add_metric(self, metric_data: dict) -> None:
        """Metrik ekle"""
        if not self.record_check.isChecked():
            return
        
        self.metrics_history.append(metric_data)
        
        row = self.metrics_table.rowCount()
        self.metrics_table.insertRow(row)
        
        # Kare
        self.metrics_table.setItem(row, 0, QTableWidgetItem(str(metric_data.get("frame", 0))))
        
        # Zaman
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.metrics_table.setItem(row, 1, QTableWidgetItem(timestamp))
        
        # mAP
        self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{metric_data.get('map', 0):.3f}"))
        
        # RMSE
        self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{metric_data.get('rmse', 0):.3f}"))
        
        # Eşleşme
        self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{metric_data.get('matching', 0):.3f}"))
        
        # FPS
        self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{metric_data.get('fps', 0):.1f}"))
        
        # Latency
        self.metrics_table.setItem(row, 6, QTableWidgetItem(f"{metric_data.get('latency', 0):.1f}"))
        
        # GPU
        self.metrics_table.setItem(row, 7, QTableWidgetItem(f"{metric_data.get('gpu', 0):.1f}%"))
        
        # RAM
        self.metrics_table.setItem(row, 8, QTableWidgetItem(f"{metric_data.get('ram', 0):.1f}GB"))
        
        # Durum
        self.metrics_table.setItem(row, 9, QTableWidgetItem(metric_data.get("status", "OK")))
    
    def export_to_csv(self) -> None:
        """CSV'ye aktar"""
        if not self.metrics_history:
            logger.warning("Dışa aktarılacak metrik yok")
            return
        
        try:
            output_dir = Path("./results")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.metrics_history[0].keys())
                writer.writeheader()
                writer.writerows(self.metrics_history)
            
            logger.info(f"Metrikler dışa aktarıldı: {filename}")
            self.metrics_saved.emit(str(filename))
        except Exception as e:
            logger.error(f"CSV dışa aktarma hatası: {e}")
    
    def clear_metrics(self) -> None:
        """Metrikleri temizle"""
        self.metrics_history.clear()
        self.metrics_table.setRowCount(0)
        logger.info("Metrikler temizlendi")
