"""
Kıyaslama Paneli - Model ve algoritma karşılaştırması
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging

logger = logging.getLogger(__name__)


class ComparisonPanel(QWidget):
    """Kıyaslama paneli"""
    
    comparison_requested = pyqtSignal(str, str)  # task, model1, model2
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self) -> None:
        """UI'yi başlat"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Başlık
        title = QLabel("🔄 Model/Algoritma Kıyaslaması")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Seçim Alanı
        selection_layout = QHBoxLayout()
        
        selection_layout.addWidget(QLabel("Görev:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems(["Görev 1", "Görev 2", "Görev 3"])
        selection_layout.addWidget(self.task_combo)
        
        selection_layout.addWidget(QLabel("Model 1:"))
        self.model1_combo = QComboBox()
        selection_layout.addWidget(self.model1_combo)
        
        selection_layout.addWidget(QLabel("Model 2:"))
        self.model2_combo = QComboBox()
        selection_layout.addWidget(self.model2_combo)
        
        compare_btn = QPushButton("📊 Karşılaştır")
        compare_btn.clicked.connect(self.compare_models)
        selection_layout.addWidget(compare_btn)
        
        main_layout.addLayout(selection_layout)
        
        # Karşılaştırma Tablosu
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setHorizontalHeaderLabels(["Metrik", "Model 1", "Model 2"])
        main_layout.addWidget(self.comparison_table)
        
        main_layout.addStretch()
    
    def compare_models(self) -> None:
        """Modelleri karşılaştır"""
        task = self.task_combo.currentText()
        model1 = self.model1_combo.currentText()
        model2 = self.model2_combo.currentText()
        
        if model1 == model2:
            logger.warning("Aynı modeller seçildi")
            return
        
        self.comparison_requested.emit(task, model1, model2)
        logger.info(f"Karşılaştırma: {task} - {model1} vs {model2}")
