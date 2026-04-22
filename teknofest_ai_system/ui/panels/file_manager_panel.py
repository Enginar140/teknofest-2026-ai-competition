"""
Dosya Yönetimi Paneli - Video/Görüntü yükleme ve yönetimi
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QFont, QDrag, QPixmap
from PyQt6.QtCore import QUrl
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DragDropArea(QFrame):
    """Sürükle-bırak alanı"""
    
    files_dropped = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        self.setLineWidth(2)
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        label = QLabel("🎬 Video/Görüntü Sürükle-Bırak\n\nveya Dosya Seç")
        label_font = QFont()
        label_font.setPointSize(12)
        label.setFont(label_font)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
    
    def dragEnterEvent(self, event):
        """Sürükleme girişi"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Bırakma olayı"""
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        self.files_dropped.emit(files)


class FileManagerPanel(QWidget):
    """Dosya yönetimi paneli"""
    
    files_selected = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.selected_files = []
        self.init_ui()
    
    def init_ui(self) -> None:
        """UI'yi başlat"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Başlık
        title = QLabel("📁 Dosya Yönetimi")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Sürükle-bırak alanı
        self.drag_drop_area = DragDropArea()
        self.drag_drop_area.files_dropped.connect(self.on_files_dropped)
        main_layout.addWidget(self.drag_drop_area)
        
        # Butonlar
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        browse_btn = QPushButton("📂 Dosya Seç")
        browse_btn.clicked.connect(self.browse_files)
        button_layout.addWidget(browse_btn)
        
        clear_btn = QPushButton("🗑️ Temizle")
        clear_btn.clicked.connect(self.clear_files)
        button_layout.addWidget(clear_btn)
        
        main_layout.addLayout(button_layout)
        
        # Dosya listesi
        list_label = QLabel("Seçili Dosyalar:")
        main_layout.addWidget(list_label)
        
        self.file_list = QListWidget()
        main_layout.addWidget(self.file_list)
        
        # Bilgi
        info_label = QLabel(
            "Desteklenen Formatlar: MP4, AVI, MOV, PNG, JPG\n"
            "Maksimum Dosya Boyutu: 5GB"
        )
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        main_layout.addWidget(info_label)
    
    def browse_files(self) -> None:
        """Dosya tarayıcısını aç"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Dosya Seç",
            "",
            "Video Dosyaları (*.mp4 *.avi *.mov);;Görüntü Dosyaları (*.png *.jpg *.jpeg);;Tüm Dosyalar (*.*)"
        )
        
        if files:
            self.on_files_dropped(files)
    
    def on_files_dropped(self, files: list) -> None:
        """Dosyalar bırakıldığında"""
        self.selected_files.extend(files)
        self.update_file_list()
        self.files_selected.emit(self.selected_files)
        logger.info(f"{len(files)} dosya seçildi")
    
    def update_file_list(self) -> None:
        """Dosya listesini güncelle"""
        self.file_list.clear()
        
        for file_path in self.selected_files:
            file_name = Path(file_path).name
            file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            
            item_text = f"{file_name} ({file_size:.2f} MB)"
            item = QListWidgetItem(item_text)
            self.file_list.addItem(item)
    
    def clear_files(self) -> None:
        """Dosyaları temizle"""
        self.selected_files.clear()
        self.file_list.clear()
        logger.info("Dosya listesi temizlendi")
    
    def get_selected_files(self) -> list:
        """Seçili dosyaları al"""
        return self.selected_files
