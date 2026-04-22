"""
Sunucu Bağlantısı Paneli - TEKNOFEST sunucusu ile iletişim
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QGridLayout, QTextEdit, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor
import logging

logger = logging.getLogger(__name__)


class ServerPanel(QWidget):
    """Sunucu bağlantısı paneli"""
    
    connection_requested = pyqtSignal(str, str, str)  # url, username, password
    connection_result = pyqtSignal(bool, str)  # success, message
    disconnect_requested = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.is_connected = False
        self.init_ui()
    
    def init_ui(self) -> None:
        """UI'yi başlat"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Başlık
        title = QLabel("🌐 Sunucu Bağlantısı")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        main_layout.addWidget(title)
        
        # Bağlantı Ayarları
        connection_group = QGroupBox("Bağlantı Ayarları")
        connection_layout = QGridLayout(connection_group)
        
        connection_layout.addWidget(QLabel("Sunucu URL:"), 0, 0)
        self.url_input = QLineEdit()
        self.url_input.setText("http://127.0.0.1:5000/")
        connection_layout.addWidget(self.url_input, 0, 1)
        
        connection_layout.addWidget(QLabel("Takım Adı:"), 1, 0)
        self.team_input = QLineEdit()
        self.team_input.setText("demo")
        connection_layout.addWidget(self.team_input, 1, 1)
        
        connection_layout.addWidget(QLabel("Şifre:"), 2, 0)
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        connection_layout.addWidget(self.password_input, 2, 1)
        
        main_layout.addWidget(connection_group)
        
        # Bağlantı Butonları
        button_layout = QHBoxLayout()
        
        self.connect_btn = QPushButton("🔗 Bağlan")
        self.connect_btn.clicked.connect(self.connect_to_server)
        button_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("🔌 Bağlantıyı Kes")
        self.disconnect_btn.clicked.connect(self.disconnect_from_server)
        self.disconnect_btn.setEnabled(False)
        button_layout.addWidget(self.disconnect_btn)
        
        main_layout.addLayout(button_layout)
        
        # Durum Bilgisi
        status_group = QGroupBox("Durum Bilgisi")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Durum: Bağlı Değil")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addWidget(QLabel("Oturum Bilgileri:"))
        self.session_info = QTextEdit()
        self.session_info.setReadOnly(True)
        self.session_info.setMaximumHeight(150)
        status_layout.addWidget(self.session_info)
        
        main_layout.addWidget(status_group)
        
        # İstatistikler
        stats_group = QGroupBox("İstatistikler")
        stats_layout = QGridLayout(stats_group)
        
        stats_layout.addWidget(QLabel("Gönderilen Kare:"), 0, 0)
        self.sent_frames_label = QLabel("0")
        stats_layout.addWidget(self.sent_frames_label, 0, 1)
        
        stats_layout.addWidget(QLabel("Alınan Kare:"), 1, 0)
        self.received_frames_label = QLabel("0")
        stats_layout.addWidget(self.received_frames_label, 1, 1)
        
        stats_layout.addWidget(QLabel("Hata Sayısı:"), 2, 0)
        self.error_count_label = QLabel("0")
        stats_layout.addWidget(self.error_count_label, 2, 1)
        
        main_layout.addWidget(stats_group)
        
        main_layout.addStretch()
    
    def connect_to_server(self) -> None:
        """Sunucuya bağlan (kimlik doğrulama MainWindow'da yapılır)."""
        url = self.url_input.text().strip()
        team = self.team_input.text().strip()
        password = self.password_input.text()
        
        if not all([url, team, password]):
            logger.warning("Eksik bağlantı bilgileri")
            return
        
        self.connection_requested.emit(url, team, password)
    
    def disconnect_from_server(self) -> None:
        """Sunucudan bağlantıyı kes"""
        self.is_connected = False
        self.update_connection_status(False)
        self.disconnect_requested.emit()
        logger.info("Sunucu bağlantısı kesildi")
    
    def apply_connection_success(self, message: str = "") -> None:
        """MainWindow başarılı login sonrası çağırır."""
        self.is_connected = True
        self.update_connection_status(True)
        if message:
            self.session_info.append("\n" + message)
        self.connection_result.emit(True, message or "Bağlandı")

    def apply_connection_failure(self, message: str) -> None:
        """MainWindow başarısız login sonrası çağırır."""
        self.is_connected = False
        self.update_connection_status(False)
        self.status_label.setText("Durum: Bağlantı başarısız")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.connection_result.emit(False, message)

    def update_connection_status(self, connected: bool) -> None:
        """Bağlantı durumunu güncelle"""
        if connected:
            self.status_label.setText("Durum: Bağlı ✓")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            
            info_text = (
                f"Sunucu: {self.url_input.text()}\n"
                f"Takım: {self.team_input.text()}\n"
                f"Bağlantı Zamanı: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}"
            )
            self.session_info.setText(info_text)
        else:
            self.status_label.setText("Durum: Bağlı Değil")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.session_info.clear()
    
    def update_statistics(self, sent: int, received: int, errors: int) -> None:
        """İstatistikleri güncelle"""
        self.sent_frames_label.setText(str(sent))
        self.received_frames_label.setText(str(received))
        self.error_count_label.setText(str(errors))
