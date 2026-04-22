"""
PyQt6 Ana Pencere - TEKNOFEST 2026 Yapay Zeka Sistemi
"""
import sys
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QStatusBar, QMenuBar, QMenu, QMessageBox, QPushButton,
    QLabel, QFrame, QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon, QFont, QPalette

from core.config_manager import get_config_manager
from core.constants import COLORS, GRAPH_COLORS

# UI Panelleri
from ui.panels.dashboard_panel import DashboardPanel
from ui.panels.file_manager_panel import FileManagerPanel
from ui.panels.model_selection_panel import ModelSelectionPanel
from ui.panels.camera_panel import CameraPanel
from ui.panels.metrics_panel import MetricsPanel
from ui.panels.comparison_panel import ComparisonPanel
from ui.panels.server_panel import ServerPanel
from ui.panels.settings_panel import SettingsPanel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Ana pencere sınıfı"""
    
    # Sinyaller
    config_changed = pyqtSignal(str, object)
    metrics_updated = pyqtSignal(dict)
    processing_started = pyqtSignal()
    processing_stopped = pyqtSignal()
    
    # Görev sinyalleri
    task_selected = pyqtSignal(str)  # 'task1', 'task2', 'task3'
    
    def __init__(self):
        super().__init__()
        self.config = get_config_manager()
        self.current_task = 'task1'  # Varsayılan görev
        self.real_time_stats = {
            'fps': 0.0,
            'latency': 0.0,
            'gpu_usage': 0.0,
            'detections': 0,
            'tracked_objects': 0,
            'server_status': 'Bağlı Değil'
        }
        self.init_ui()
        self.setup_menu()
        self.setup_status_bar()
        self.setup_timers()
        self._connect_signals()
        
    def init_ui(self) -> None:
        """UI'yi başlat"""
        # Pencere ayarları
        self.setWindowTitle("TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması")
        self.setGeometry(100, 100, 1600, 900)
        
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Ana layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sol panel (Kontrol)
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Sağ panel (Sekme widget)
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._get_tab_stylesheet())
        self._create_tabs()
        main_layout.addWidget(self.tab_widget, 3)
        
        # Stil uygula
        self.setStyleSheet(self._get_stylesheet())
    
    def _create_left_panel(self) -> QWidget:
        """Sol kontrol panelini oluştur - Görev seçimi ve hızlı istatistikler"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(12)
        
        # === Başlık ===
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        
        title_label = QLabel("TEKNOFEST 2026")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {COLORS['accent']}; padding: 8px;")
        left_layout.addWidget(title_label)
        
        subtitle_label = QLabel("Havacılıkta Yapay Zeka")
        _sf = QFont()
        _sf.setPointSize(10)
        subtitle_label.setFont(_sf)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setStyleSheet("color: #888;")
        left_layout.addWidget(subtitle_label)
        
        left_layout.addSpacing(15)
        
        # === Görev Seçimi Grubu ===
        task_group = QGroupBox("📋 Görev Seçimi")
        task_group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {COLORS['accent']};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: {COLORS['foreground']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        task_layout = QVBoxLayout()
        task_layout.setSpacing(8)
        
        # Görev 1: Nesne Tespiti
        self.task1_btn = QPushButton("1️⃣ Nesne Tespiti")
        self._setup_task_button(self.task1_btn, 'task1',
                               "4 sınıf tespiti: Taşıt, İnsan, UAP, UAİ")
        task_layout.addWidget(self.task1_btn)
        
        # Görev 2: Pozisyon Kestirimi
        self.task2_btn = QPushButton("2️⃣ Pozisyon Kestirimi")
        self._setup_task_button(self.task2_btn, 'task2',
                               "GPS + VO + EKF ile konum belirleme")
        task_layout.addWidget(self.task2_btn)
        
        # Görev 3: Görüntü Eşleme
        self.task3_btn = QPushButton("3️⃣ Görüntü Eşleme")
        self._setup_task_button(self.task3_btn, 'task3',
                               "XoFTR + LightGlue ile feature matching")
        task_layout.addWidget(self.task3_btn)
        
        task_group.setLayout(task_layout)
        left_layout.addWidget(task_group)
        
        left_layout.addSpacing(10)
        
        # === Hızlı İstatistikler Grubu ===
        stats_group = QGroupBox("📊 Canlı İstatistikler")
        stats_group.setStyleSheet(task_group.styleSheet())
        stats_layout = QGridLayout()
        stats_layout.setSpacing(8)
        
        # FPS
        self.fps_label = self._create_stat_label("FPS", "0.0", "#4CAF50")
        stats_layout.addWidget(self.fps_label['title'], 0, 0)
        stats_layout.addWidget(self.fps_label['value'], 1, 0)
        
        # Latency
        self.latency_label = self._create_stat_label("Gecikme", "0ms", "#FF9800")
        stats_layout.addWidget(self.latency_label['title'], 0, 1)
        stats_layout.addWidget(self.latency_label['value'], 1, 1)
        
        # Tespit Sayısı
        self.detections_label = self._create_stat_label("Tespit", "0", "#2196F3")
        stats_layout.addWidget(self.detections_label['title'], 2, 0)
        stats_layout.addWidget(self.detections_label['value'], 3, 0)
        
        # Takip Edilen Nesneler
        self.tracked_label = self._create_stat_label("Takip", "0", "#9C27B0")
        stats_layout.addWidget(self.tracked_label['title'], 2, 1)
        stats_layout.addWidget(self.tracked_label['value'], 3, 1)
        
        # GPU Kullanımı
        self.gpu_label = self._create_stat_label("GPU", "0%", "#F44336")
        stats_layout.addWidget(self.gpu_label['title'], 4, 0)
        stats_layout.addWidget(self.gpu_label['value'], 5, 0)
        
        # Sunucu Durumu
        self.server_label = self._create_stat_label("Sunucu", "❌", "#607D8B")
        stats_layout.addWidget(self.server_label['title'], 4, 1)
        stats_layout.addWidget(self.server_label['value'], 5, 1)
        
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)
        
        left_layout.addSpacing(10)
        
        # === Mevcut Görev Bilgisi ===
        current_task_group = QGroupBox("🎯 Mevcut Görev")
        current_task_group.setStyleSheet(task_group.styleSheet())
        current_task_layout = QVBoxLayout()
        
        self.current_task_label = QLabel("Nesne Tespiti")
        _ctf = QFont()
        _ctf.setPointSize(11)
        _ctf.setBold(True)
        self.current_task_label.setFont(_ctf)
        self.current_task_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_task_label.setStyleSheet(f"color: {COLORS['accent']};")
        current_task_layout.addWidget(self.current_task_label)
        
        self.current_task_desc = QLabel("4 sınıf tespiti")
        self.current_task_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_task_desc.setStyleSheet("color: #888; font-size: 9px;")
        current_task_layout.addWidget(self.current_task_desc)
        
        current_task_group.setLayout(current_task_layout)
        left_layout.addWidget(current_task_group)
        
        # === Hızlı Eylem Butonları ===
        left_layout.addSpacing(10)
        
        actions_layout = QGridLayout()
        actions_layout.setSpacing(8)
        
        self.start_btn = QPushButton("▶️ Başlat")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.start_btn.clicked.connect(self.on_start_clicked)
        actions_layout.addWidget(self.start_btn, 0, 0)
        
        self.stop_btn = QPushButton("⏹️ Durdur")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #b9150a;
            }
        """)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.stop_btn.setEnabled(False)
        actions_layout.addWidget(self.stop_btn, 0, 1)
        
        left_layout.addLayout(actions_layout)
        
        left_layout.addStretch()
        
        # İlk görevi seçili olarak işaretle
        self._highlight_selected_task('task1')
        
        return left_widget
    
    def _setup_task_button(self, button: QPushButton, task_id: str, tooltip: str):
        """Görev butonunu yapılandır"""
        button.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: white;
                border: 2px solid #444;
                border-radius: 6px;
                padding: 12px;
                text-align: left;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #555;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
        """)
        button.setToolTip(tooltip)
        button.clicked.connect(lambda: self.on_task_selected(task_id))
    
    def _create_stat_label(self, title: str, initial_value: str, color: str) -> dict:
        """İstatistik label'ı oluştur"""
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: #888; font-size: 9px;")
        
        value_label = QLabel(initial_value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
                font-size: 12px;
            }}
        """)
        
        return {'title': title_label, 'value': value_label}
    
    def _highlight_selected_task(self, task_id: str):
        """Seçili görevi vurgula"""
        buttons = {
            'task1': self.task1_btn,
            'task2': self.task2_btn,
            'task3': self.task3_btn
        }
        
        task_info = {
            'task1': ('Nesne Tespiti', '4 sınıf tespiti: Taşıt, İnsan, UAP, UAİ'),
            'task2': ('Pozisyon Kestirimi', 'GPS + VO + EKF ile konum belirleme'),
            'task3': ('Görüntü Eşleme', 'XoFTR + LightGlue ile feature matching')
        }
        
        for tid, btn in buttons.items():
            if tid == task_id:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #1a237e;
                        color: white;
                        border: 2px solid #3949ab;
                        border-radius: 6px;
                        padding: 12px;
                        text-align: left;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        background-color: #283593;
                    }
                """)
            else:
                self._setup_task_button(btn, tid, "")
        
        # Mevcut görev bilgisini güncelle
        title, desc = task_info[task_id]
        self.current_task_label.setText(title)
        self.current_task_desc.setText(desc)
    
    def on_task_selected(self, task_id: str):
        """Görev seçildiğinde"""
        self.current_task = task_id
        self._highlight_selected_task(task_id)
        
        # Diğer panellere bildir
        self.task_selected.emit(task_id)
        
        # Camera panelini güncelle
        if hasattr(self, 'camera_panel'):
            self.camera_panel.set_task(task_id)
        
        logger.info(f"Görev seçildi: {task_id}")
    
    def on_start_clicked(self):
        """Sol panel Başlat — Kamera sekmesinde işlemi başlatır."""
        self.tab_widget.setCurrentWidget(self.camera_panel)
        self.camera_panel.start_processing()
        logger.info("Kamera işlemi başlatıldı (sol panel)")

    def on_stop_clicked(self):
        """Sol panel Durdur — kamera iş parçacığını durdurur."""
        self.camera_panel.stop_processing()
        logger.info("Kamera işlemi durduruldu (sol panel)")

    def _on_camera_processing_state(self, running: bool) -> None:
        """Kamera paneli çalışma durumuna göre sol Başlat/Durdur ve sinyaller."""
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        if running:
            self.processing_started.emit()
        else:
            self.processing_stopped.emit()

    def _sync_left_task_from_camera_panel(self, task_id: str) -> None:
        """Kamera sekmesinde seçilen görevi sol panelde göster (yeniden kamera çağrısı yok)."""
        self.current_task = task_id
        self._highlight_selected_task(task_id)
        self.task_selected.emit(task_id)
    
    def _connect_signals(self):
        """Panel arası sinyal bağlantılarını kur"""
        srv_url = self.config.get("server.url")
        if srv_url and hasattr(self, "server_panel"):
            self.server_panel.url_input.setText(str(srv_url).strip())

        # Dashboard paneline metrik güncellemesi
        self.metrics_updated.connect(self.dashboard_panel.set_metrics)
        
        # Camera panelinden sinyaller
        if hasattr(self, 'camera_panel'):
            # Camera panel frame hazır olduğunda
            self.camera_panel.frame_ready.connect(self.on_camera_frame_ready)
            # Camera panel istatistik güncellediğinde
            self.camera_panel.stats_updated.connect(self.on_camera_stats_updated)
            self.camera_panel.processing_state_changed.connect(
                self._on_camera_processing_state
            )
            self.camera_panel.camera_task_sync.connect(
                self._sync_left_task_from_camera_panel
            )
            self.camera_panel.set_task("task1")

        if hasattr(self, 'server_panel'):
            self.server_panel.connection_requested.connect(self._on_teknofest_server_connect)
            self.server_panel.disconnect_requested.connect(self._on_teknofest_server_disconnect)

    def _on_teknofest_server_connect(self, url: str, team: str, password: str) -> None:
        """Resmi Teknofest HTTP API ile giriş ve kamera paneline aktarım."""
        from server.teknofest_connection import TeknofestConnectionHandler

        base = url.strip().rstrip("/") + "/"
        try:
            handler = TeknofestConnectionHandler(base, username=team, password=password)
        except Exception as exc:
            logger.exception("Sunucu bağlantı hatası")
            self.server_panel.apply_connection_failure(str(exc))
            QMessageBox.warning(self, "Bağlantı hatası", str(exc))
            return

        if handler.is_authenticated:
            self.camera_panel.set_teknofest_connection(handler)
            self.server_panel.apply_connection_success()
            self.real_time_stats["server_status"] = "Bağlı (Teknofest API)"
            self._update_left_panel_stats()
        else:
            self.server_panel.apply_connection_failure("Kimlik doğrulama başarısız")
            QMessageBox.warning(self, "Bağlantı", "Kullanıcı adı veya şifre kabul edilmedi.")

    def _on_teknofest_server_disconnect(self) -> None:
        self.camera_panel.clear_teknofest_connection()
        self.real_time_stats["server_status"] = "Bağlı Değil"
        self._update_left_panel_stats()
    
    def on_camera_frame_ready(self, frame, metadata):
        """Kamera frame'i hazır olduğunda"""
        # Dashboard'a metrik gönder
        if 'stats' in metadata:
            self.metrics_updated.emit(metadata['stats'])
    
    def on_camera_stats_updated(self, stats):
        """Kamera istatistikleri güncellendiğinde"""
        self.real_time_stats.update(stats)
        self._update_left_panel_stats()
        
        # Dashboard'a da metrikleri gönder
        dashboard_metrics = {
            'fps': stats.get('fps', 0.0),
            'gpu_usage': stats.get('gpu_usage', 0.0),
            'status': 'Çalışıyor' if stats.get('fps', 0) > 0 else 'Hazır'
        }
        self.metrics_updated.emit(dashboard_metrics)
    
    def _update_left_panel_stats(self):
        """Sol panel istatistiklerini güncelle"""
        stats = self.real_time_stats
        
        # FPS
        self.fps_label['value'].setText(f"{stats.get('fps', 0.0):.1f}")
        
        # Latency
        self.latency_label['value'].setText(f"{stats.get('latency', 0.0):.0f}ms")
        
        # Tespit
        self.detections_label['value'].setText(str(stats.get('detections', 0)))
        
        # Takip
        self.tracked_label['value'].setText(str(stats.get('tracked_objects', 0)))
        
        # GPU
        self.gpu_label['value'].setText(f"{stats.get('gpu_usage', 0.0):.0f}%")
        
        # Sunucu
        server_status = stats.get('server_status', 'Bağlı Değil')
        if 'Bağlı' in server_status:
            self.server_label['value'].setText("✅")
            self.server_label['value'].setStyleSheet("""
                QLabel {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 6px;
                    padding: 8px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """)
        else:
            self.server_label['value'].setText("❌")
            self.server_label['value'].setStyleSheet("""
                QLabel {
                    background-color: #607D8B;
                    color: white;
                    border-radius: 6px;
                    padding: 8px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """)
    
    def _create_tabs(self) -> None:
        """Sekmeleri oluştur"""
        # 1. Dashboard
        self.dashboard_panel = DashboardPanel()
        self.tab_widget.addTab(self.dashboard_panel, "📊 Dashboard")
        
        # 2. Dosya Yönetimi
        self.file_manager_panel = FileManagerPanel()
        self.tab_widget.addTab(self.file_manager_panel, "📁 Dosya Yönetimi")
        
        # 3. Model Seçimi
        self.model_selection_panel = ModelSelectionPanel()
        self.tab_widget.addTab(self.model_selection_panel, "🤖 Model Seçimi")
        
        # 4. Kamera
        self.camera_panel = CameraPanel()
        self.tab_widget.addTab(self.camera_panel, "🎥 Kamera")
        
        # 5. Metrikleme
        self.metrics_panel = MetricsPanel()
        self.tab_widget.addTab(self.metrics_panel, "📈 Metrikleme")
        
        # 6. Kıyaslama
        self.comparison_panel = ComparisonPanel()
        self.tab_widget.addTab(self.comparison_panel, "🔄 Kıyaslama")
        
        # 7. Sunucu Bağlantısı
        self.server_panel = ServerPanel()
        self.tab_widget.addTab(self.server_panel, "🌐 Sunucu")
        
        # 8. Ayarlar
        self.settings_panel = SettingsPanel()
        self.tab_widget.addTab(self.settings_panel, "⚙️ Ayarlar")
    
    def setup_menu(self) -> None:
        """Menü çubuğunu oluştur"""
        menubar = self.menuBar()
        
        # Dosya Menüsü
        file_menu = menubar.addMenu("Dosya")
        file_menu.addAction("Aç", self.open_file)
        file_menu.addAction("Kaydet", self.save_config)
        file_menu.addSeparator()
        file_menu.addAction("Çıkış", self.close)
        
        # Görüntüle Menüsü
        view_menu = menubar.addMenu("Görüntüle")
        view_menu.addAction("Tema Değiştir", self.change_theme)
        view_menu.addAction("Tam Ekran", self.toggle_fullscreen)
        
        # Araçlar Menüsü
        tools_menu = menubar.addMenu("Araçlar")
        tools_menu.addAction("Kalibrasyonu Yükle", self.load_calibration)
        tools_menu.addAction("Modelleri İndir", self.download_models)
        
        # Hakkında Menüsü
        help_menu = menubar.addMenu("Hakkında")
        help_menu.addAction("Hakkında", self.show_about)
        help_menu.addAction("Yardım", self.show_help)
    
    def setup_status_bar(self) -> None:
        """Durum çubuğunu oluştur"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hazır | FPS: 0.0 | Latency: 0ms | GPU: 0%")
    
    def setup_timers(self) -> None:
        """Zamanlayıcıları oluştur"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(100)  # 100ms
    
    def update_status(self) -> None:
        """Durum çubuğunu güncelle - gerçek zamanlı metrikler"""
        stats = self.real_time_stats
        status_text = (
            f"Hazır | FPS: {stats['fps']:.1f} | "
            f"Gecikme: {stats['latency']:.0f}ms | "
            f"GPU: {stats['gpu_usage']:.0f}% | "
            f"Tespit: {stats['detections']} | "
            f"Sunucu: {stats['server_status']}"
        )
        self.status_bar.showMessage(status_text)
    
    def open_file(self) -> None:
        """Dosya aç"""
        logger.info("Dosya açma işlemi başlatıldı")
    
    def save_config(self) -> None:
        """Konfigürasyonu kaydet"""
        self.config.save_config()
        QMessageBox.information(self, "Başarılı", "Konfigürasyon kaydedildi")
    
    def change_theme(self) -> None:
        """Temayı değiştir"""
        logger.info("Tema değiştirme işlemi başlatıldı")
    
    def toggle_fullscreen(self) -> None:
        """Tam ekran modunu aç/kapat"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    
    def load_calibration(self) -> None:
        """Kalibrasyonu yükle"""
        logger.info("Kalibrasyonu yükleme işlemi başlatıldı")
    
    def download_models(self) -> None:
        """Modelleri indir"""
        logger.info("Model indirme işlemi başlatıldı")
    
    def show_about(self) -> None:
        """Hakkında penceresini göster"""
        QMessageBox.about(
            self,
            "Hakkında",
            "TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması\n"
            "Versiyon: 1.0.0\n"
            "Sistem: Modüler Yapay Zeka Platformu"
        )
    
    def show_help(self) -> None:
        """Yardım penceresini göster"""
        QMessageBox.information(
            self,
            "Yardım",
            "Yardım belgesi burada gösterilecektir"
        )
    
    def _get_stylesheet(self) -> str:
        """Genel stil sayfasını döndür"""
        return f"""
        QMainWindow {{
            background-color: {COLORS['background']};
            color: {COLORS['foreground']};
        }}
        QMenuBar {{
            background-color: {COLORS['background']};
            color: {COLORS['foreground']};
            border-bottom: 1px solid {COLORS['accent']};
        }}
        QMenuBar::item:selected {{
            background-color: {COLORS['accent']};
        }}
        QMenu {{
            background-color: {COLORS['background']};
            color: {COLORS['foreground']};
        }}
        QMenu::item:selected {{
            background-color: {COLORS['accent']};
        }}
        QStatusBar {{
            background-color: {COLORS['background']};
            color: {COLORS['foreground']};
            border-top: 1px solid {COLORS['accent']};
        }}
        """
    
    def _get_tab_stylesheet(self) -> str:
        """Sekme stil sayfasını döndür"""
        return f"""
        QTabWidget::pane {{
            border: 1px solid {COLORS['accent']};
        }}
        QTabBar::tab {{
            background-color: {COLORS['background']};
            color: {COLORS['foreground']};
            padding: 8px 20px;
            border: 1px solid {COLORS['accent']};
        }}
        QTabBar::tab:selected {{
            background-color: {COLORS['accent']};
            color: white;
        }}
        """
    
    def closeEvent(self, event) -> None:
        """Pencere kapatılırken"""
        reply = QMessageBox.question(
            self,
            "Çıkış",
            "Çıkmak istediğinizden emin misiniz?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if hasattr(self, "camera_panel"):
                self.camera_panel.stop_processing()
            self.config.save_config()
            event.accept()
        else:
            event.ignore()


def main():
    """Ana fonksiyon"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    main()
