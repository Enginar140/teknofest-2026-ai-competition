"""
Model Seçimi Paneli - Görev başına model seçimi ve parametreleri
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QGridLayout,
    QPushButton, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging

from core.config_manager import get_config_manager
from core.constants import YOLOV8_MODELS

logger = logging.getLogger(__name__)


class ModelSelectionPanel(QWidget):
    """Model seçimi paneli"""
    
    model_changed = pyqtSignal(str, str)  # task, model_name
    parameters_changed = pyqtSignal(str, dict)  # task, parameters
    
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
        title = QLabel("🤖 Model Seçimi")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Görev 1: Nesne Tespiti
        task1_group = self._create_task1_group()
        main_layout.addWidget(task1_group)
        
        # Görev 2: Pozisyon Kestirimi
        task2_group = self._create_task2_group()
        main_layout.addWidget(task2_group)
        
        # Görev 3: Görüntü Eşleme
        task3_group = self._create_task3_group()
        main_layout.addWidget(task3_group)
        
        # Model Karşılaştırma Tablosu
        comparison_label = QLabel("Model Performans Karşılaştırması:")
        main_layout.addWidget(comparison_label)
        
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(5)
        self.comparison_table.setHorizontalHeaderLabels(
            ["Model", "Parametreler", "mAP@0.5", "FPS", "VRAM"]
        )
        self.comparison_table.setMaximumHeight(200)
        self._populate_comparison_table()
        main_layout.addWidget(self.comparison_table)
        
        main_layout.addStretch()
    
    def _create_task1_group(self) -> QGroupBox:
        """Görev 1 grubunu oluştur"""
        group = QGroupBox("Görev 1: Nesne Tespiti (mAP@0.5)")
        layout = QGridLayout(group)
        
        # Model seçimi
        layout.addWidget(QLabel("Model:"), 0, 0)
        self.task1_model_combo = QComboBox()
        self.task1_model_combo.addItems(self.config.get_all_models("task1"))
        self.task1_model_combo.currentTextChanged.connect(
            lambda m: self.on_model_changed("task1", m)
        )
        layout.addWidget(self.task1_model_combo, 0, 1)
        
        # Confidence Threshold
        layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.task1_conf_spin = QDoubleSpinBox()
        self.task1_conf_spin.setRange(0.0, 1.0)
        self.task1_conf_spin.setValue(0.5)
        self.task1_conf_spin.setSingleStep(0.05)
        self.task1_conf_spin.valueChanged.connect(self.on_task1_params_changed)
        layout.addWidget(self.task1_conf_spin, 1, 1)
        
        # IoU Threshold
        layout.addWidget(QLabel("IoU Threshold:"), 2, 0)
        self.task1_iou_spin = QDoubleSpinBox()
        self.task1_iou_spin.setRange(0.0, 1.0)
        self.task1_iou_spin.setValue(0.5)
        self.task1_iou_spin.setSingleStep(0.05)
        self.task1_iou_spin.valueChanged.connect(self.on_task1_params_changed)
        layout.addWidget(self.task1_iou_spin, 2, 1)
        
        # ByteTrack
        self.task1_bytetrack_check = QCheckBox("ByteTrack MOT Kullan")
        self.task1_bytetrack_check.setChecked(True)
        self.task1_bytetrack_check.stateChanged.connect(self.on_task1_params_changed)
        layout.addWidget(self.task1_bytetrack_check, 3, 0)
        
        # SAHI
        self.task1_sahi_check = QCheckBox("SAHI Kullan")
        self.task1_sahi_check.setChecked(True)
        self.task1_sahi_check.stateChanged.connect(self.on_task1_params_changed)
        layout.addWidget(self.task1_sahi_check, 3, 1)
        
        return group
    
    def _create_task2_group(self) -> QGroupBox:
        """Görev 2 grubunu oluştur"""
        group = QGroupBox("Görev 2: Pozisyon Kestirimi (RMSE)")
        layout = QGridLayout(group)
        
        # Algoritma seçimi
        layout.addWidget(QLabel("Algoritma:"), 0, 0)
        self.task2_algo_combo = QComboBox()
        self.task2_algo_combo.addItems(self.config.get_all_models("task2"))
        self.task2_algo_combo.currentTextChanged.connect(
            lambda a: self.on_model_changed("task2", a)
        )
        layout.addWidget(self.task2_algo_combo, 0, 1)
        
        # SSR Kullan
        self.task2_ssr_check = QCheckBox("Semantik Ölçek Kurtarma (SSR)")
        self.task2_ssr_check.setChecked(True)
        self.task2_ssr_check.stateChanged.connect(self.on_task2_params_changed)
        layout.addWidget(self.task2_ssr_check, 1, 0)
        
        # ORB-SLAM2 Kullan
        self.task2_orb_check = QCheckBox("ORB-SLAM2 Kullan")
        self.task2_orb_check.setChecked(True)
        self.task2_orb_check.stateChanged.connect(self.on_task2_params_changed)
        layout.addWidget(self.task2_orb_check, 1, 1)
        
        # Kalman Process Noise
        layout.addWidget(QLabel("Kalman Process Noise:"), 2, 0)
        self.task2_kf_pn_spin = QDoubleSpinBox()
        self.task2_kf_pn_spin.setRange(0.0, 1.0)
        self.task2_kf_pn_spin.setValue(0.01)
        self.task2_kf_pn_spin.setSingleStep(0.001)
        self.task2_kf_pn_spin.valueChanged.connect(self.on_task2_params_changed)
        layout.addWidget(self.task2_kf_pn_spin, 2, 1)
        
        # Kalman Measurement Noise
        layout.addWidget(QLabel("Kalman Measurement Noise:"), 3, 0)
        self.task2_kf_mn_spin = QDoubleSpinBox()
        self.task2_kf_mn_spin.setRange(0.0, 1.0)
        self.task2_kf_mn_spin.setValue(0.1)
        self.task2_kf_mn_spin.setSingleStep(0.01)
        self.task2_kf_mn_spin.valueChanged.connect(self.on_task2_params_changed)
        layout.addWidget(self.task2_kf_mn_spin, 3, 1)
        
        return group
    
    def _create_task3_group(self) -> QGroupBox:
        """Görev 3 grubunu oluştur"""
        group = QGroupBox("Görev 3: Görüntü Eşleme (mAP@0.5)")
        layout = QGridLayout(group)
        
        # Algoritma seçimi
        layout.addWidget(QLabel("Algoritma:"), 0, 0)
        self.task3_algo_combo = QComboBox()
        self.task3_algo_combo.addItems(self.config.get_all_models("task3"))
        self.task3_algo_combo.currentTextChanged.connect(
            lambda a: self.on_model_changed("task3", a)
        )
        layout.addWidget(self.task3_algo_combo, 0, 1)
        
        # Confidence Threshold
        layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        self.task3_conf_spin = QDoubleSpinBox()
        self.task3_conf_spin.setRange(0.0, 1.0)
        self.task3_conf_spin.setValue(0.7)
        self.task3_conf_spin.setSingleStep(0.05)
        self.task3_conf_spin.valueChanged.connect(self.on_task3_params_changed)
        layout.addWidget(self.task3_conf_spin, 1, 1)
        
        # Ensemble Kullan
        self.task3_ensemble_check = QCheckBox("Ensemble Yöntemi Kullan")
        self.task3_ensemble_check.setChecked(True)
        self.task3_ensemble_check.stateChanged.connect(self.on_task3_params_changed)
        layout.addWidget(self.task3_ensemble_check, 2, 0)
        
        return group
    
    def _populate_comparison_table(self) -> None:
        """Karşılaştırma tablosunu doldur"""
        row = 0
        for model_name, specs in YOLOV8_MODELS.items():
            self.comparison_table.insertRow(row)
            
            self.comparison_table.setItem(row, 0, QTableWidgetItem(model_name))
            self.comparison_table.setItem(row, 1, QTableWidgetItem(specs["params"]))
            self.comparison_table.setItem(row, 2, QTableWidgetItem(str(specs["map50"])))
            self.comparison_table.setItem(row, 3, QTableWidgetItem(f"{specs['fps']} fps"))
            self.comparison_table.setItem(row, 4, QTableWidgetItem(specs["vram"]))
            
            row += 1
    
    def on_model_changed(self, task: str, model_name: str) -> None:
        """Model değiştiğinde"""
        self.config.set_selected_model(task, model_name)
        self.model_changed.emit(task, model_name)
        logger.info(f"{task} için model değiştirildi: {model_name}")
    
    def on_task1_params_changed(self) -> None:
        """Görev 1 parametreleri değiştiğinde"""
        params = {
            "confidence_threshold": self.task1_conf_spin.value(),
            "iou_threshold": self.task1_iou_spin.value(),
            "use_bytetrack": self.task1_bytetrack_check.isChecked(),
            "use_sahi": self.task1_sahi_check.isChecked()
        }
        self.parameters_changed.emit("task1", params)
    
    def on_task2_params_changed(self) -> None:
        """Görev 2 parametreleri değiştiğinde"""
        params = {
            "use_ssr": self.task2_ssr_check.isChecked(),
            "use_orb_slam2": self.task2_orb_check.isChecked(),
            "kalman_process_noise": self.task2_kf_pn_spin.value(),
            "kalman_measurement_noise": self.task2_kf_mn_spin.value()
        }
        self.parameters_changed.emit("task2", params)
    
    def on_task3_params_changed(self) -> None:
        """Görev 3 parametreleri değiştiğinde"""
        params = {
            "confidence_threshold": self.task3_conf_spin.value(),
            "use_ensemble": self.task3_ensemble_check.isChecked()
        }
        self.parameters_changed.emit("task3", params)
