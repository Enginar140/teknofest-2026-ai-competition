"""
TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması
Ana Giriş Noktası
"""
import sys
import io
import logging
from pathlib import Path

# Logging konfigürasyonu - UTF-8 encoding için
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('teknofest_ai.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_directories():
    """Gerekli dizinleri oluştur"""
    directories = [
        Path("./config"),
        Path("./logs"),
        Path("./results"),
        Path("./models"),
        Path("./data"),
        Path("./temp")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Dizin oluşturuldu/kontrol edildi: {directory}")


def main():
    """Ana fonksiyon"""
    try:
        logger.info("=" * 60)
        logger.info("TEKNOFEST 2026 - Havacılıkta Yapay Zeka Yarışması")
        logger.info("Sistem başlatılıyor...")
        logger.info("=" * 60)
        
        # Dizinleri oluştur
        setup_directories()
        logger.info("Dizinler hazırlandı")
        
        # PyQt6 uygulamasını başlat
        from PyQt6.QtWidgets import QApplication
        from ui.main_window import MainWindow
        from core.config_manager import init_config
        
        # Konfigürasyonu başlat
        config = init_config("./config/default_config.json")
        logger.info("Konfigürasyon yüklendi")
        
        # PyQt6 uygulamasını oluştur
        app = QApplication(sys.argv)
        logger.info("PyQt6 uygulaması oluşturuldu")
        
        # Ana pencereyi oluştur ve göster
        window = MainWindow()
        window.show()
        logger.info("Ana pencere gösterildi")
        
        logger.info("Sistem başarıyla başlatıldı")
        
        # Uygulamayı çalıştır
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical(f"Sistem başlatma hatası: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
