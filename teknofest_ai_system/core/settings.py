"""
Konfigürasyon ve Ayarlar Yönetimi
Sistem ayarlarının merkezi yönetimi
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfigSection(Enum):
    """Konfigürasyon bölümleri"""
    CAMERA = "camera"
    DETECTION = "detection"
    POSITION = "position"
    MATCHING = "matching"
    SERVER = "server"
    PERFORMANCE = "performance"
    UI = "ui"
    LOGGING = "logging"


@dataclass
class CameraSettings:
    """Kamera ayarları"""
    source_type: str = "webcam"
    source_path: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    exposure: int = -5
    flip_horizontal: bool = False
    flip_vertical: bool = False
    buffer_size: int = 1


@dataclass
class DetectionSettings:
    """Tespit ayarları"""
    model_path: str = "yolov8l.pt"
    model_size: str = "l"
    device: str = "cuda"
    conf_threshold: float = 0.45
    iou_threshold: float = 0.5
    use_sahi: bool = False
    use_tracking: bool = True
    sahi_slice_height: int = 640
    sahi_slice_width: int = 640
    sahi_overlap_ratio: float = 0.1


@dataclass
class PositionSettings:
    """Pozisyon ayarları"""
    use_visual_odometry: bool = True
    use_ekf: bool = True
    use_ssr: bool = True
    vo_feature_type: str = "orb"
    ekf_process_noise: float = 0.1
    ekf_measurement_noise: float = 0.5
    ssr_window_size: int = 10


@dataclass
class MatchingSettings:
    """Eşleme ayarları"""
    matcher_type: str = "orb"  # orb, sift, xoftr, lightglue
    use_kalman_tracking: bool = True
    min_matches: int = 10
    ransac_threshold: float = 5.0


@dataclass
class ServerSettings:
    """Sunucu ayarları"""
    host: str = "localhost"
    port: int = 10000
    team_id: str = ""
    team_password: str = ""
    reconnect_interval: float = 5.0
    timeout: float = 10.0
    max_retries: int = 3
    enabled: bool = False


@dataclass
class PerformanceSettings:
    """Performans ayarları"""
    target_fps: int = 30
    max_gpu_memory_mb: int = 4000
    enable_dynamic_model_selection: bool = True
    enable_quantization: bool = False
    enable_tensorrt: bool = False
    batch_size: int = 1


@dataclass
class UISettings:
    """UI ayarları"""
    theme: str = "dark"  # dark, light
    window_width: int = 1920
    window_height: int = 1080
    show_fps: bool = True
    show_detections: bool = True
    show_position: bool = True
    show_metrics: bool = True
    auto_save_logs: bool = True
    log_directory: str = "logs"


@dataclass
class LoggingSettings:
    """Logging ayarları"""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/teknofest.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    console_enabled: bool = True


@dataclass
class SystemConfig:
    """Sistem konfigürasyonu"""
    version: str = "1.0.0"
    
    camera: CameraSettings = field(default_factory=CameraSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    position: PositionSettings = field(default_factory=PositionSettings)
    matching: MatchingSettings = field(default_factory=MatchingSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    ui: UISettings = field(default_factory=UISettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    
    # Metadata
    created_at: float = field(default_factory=lambda: __import__('time').time())
    modified_at: float = field(default_factory=lambda: __import__('time').time())
    
    def to_dict(self) -> Dict[str, Any]:
        """Dict'e çevir"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Dict'ten oluştur"""
        config = cls()
        
        if 'camera' in data:
            config.camera = CameraSettings(**data['camera'])
        if 'detection' in data:
            config.detection = DetectionSettings(**data['detection'])
        if 'position' in data:
            config.position = PositionSettings(**data['position'])
        if 'matching' in data:
            config.matching = MatchingSettings(**data['matching'])
        if 'server' in data:
            config.server = ServerSettings(**data['server'])
        if 'performance' in data:
            config.performance = PerformanceSettings(**data['performance'])
        if 'ui' in data:
            config.ui = UISettings(**data['ui'])
        if 'logging' in data:
            config.logging = LoggingSettings(**data['logging'])
        
        return config


class ConfigManager:
    """
    Konfigürasyon Yöneticisi
    Sistem ayarlarını yönetir ve kalıcı hale getirir
    """
    
    def __init__(self, config_path: str = "config/system_config.json"):
        """
        Konfigürasyon yöneticisi başlat
        
        Args:
            config_path: Konfigürasyon dosya yolu
        """
        self.config_path = Path(config_path)
        self.config = SystemConfig()
        
        # Konfigürasyonu yükle
        self.load()
        
        logger.info(f"ConfigManager başlatıldı: {config_path}")
    
    def load(self) -> bool:
        """
        Konfigürasyonu dosyadan yükle
        
        Returns:
            Başarılı mı?
        """
        if not self.config_path.exists():
            logger.info("Konfigürasyon dosyası bulunamadı, varsayılan ayarlar kullanılıyor")
            self.save()
            return True
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.config = SystemConfig.from_dict(data)
            logger.info("Konfigürasyon yüklendi")
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Konfigürasyon yükleme hatası: {e}")
            return False
    
    def save(self) -> bool:
        """
        Konfigürasyonu dosyaya kaydet
        
        Returns:
            Başarılı mı?
        """
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            import time
            self.config.modified_at = time.time()
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info("Konfigürasyon kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"Konfigürasyon kaydetme hatası: {e}")
            return False
    
    def get_section(self, section: ConfigSection) -> Any:
        """
        Konfigürasyon bölümünü al
        
        Args:
            section: Bölüm
            
        Returns:
            Bölüm ayarları
        """
        section_map = {
            ConfigSection.CAMERA: self.config.camera,
            ConfigSection.DETECTION: self.config.detection,
            ConfigSection.POSITION: self.config.position,
            ConfigSection.MATCHING: self.config.matching,
            ConfigSection.SERVER: self.config.server,
            ConfigSection.PERFORMANCE: self.config.performance,
            ConfigSection.UI: self.config.ui,
            ConfigSection.LOGGING: self.config.logging,
        }
        
        return section_map.get(section)
    
    def set_section(self, section: ConfigSection, settings: Any) -> bool:
        """
        Konfigürasyon bölümünü ayarla
        
        Args:
            section: Bölüm
            settings: Ayarlar
            
        Returns:
            Başarılı mı?
        """
        try:
            if section == ConfigSection.CAMERA:
                self.config.camera = settings
            elif section == ConfigSection.DETECTION:
                self.config.detection = settings
            elif section == ConfigSection.POSITION:
                self.config.position = settings
            elif section == ConfigSection.MATCHING:
                self.config.matching = settings
            elif section == ConfigSection.SERVER:
                self.config.server = settings
            elif section == ConfigSection.PERFORMANCE:
                self.config.performance = settings
            elif section == ConfigSection.UI:
                self.config.ui = settings
            elif section == ConfigSection.LOGGING:
                self.config.logging = settings
            else:
                return False
            
            self.save()
            return True
            
        except Exception as e:
            logger.error(f"Bölüm ayarlama hatası: {e}")
            return False
    
    def get_value(self, section: str, key: str) -> Optional[Any]:
        """
        Belirli bir değeri al
        
        Args:
            section: Bölüm adı
            key: Anahtar
            
        Returns:
            Değer veya None
        """
        try:
            section_obj = self.get_section(ConfigSection(section))
            if section_obj:
                return getattr(section_obj, key, None)
        except:
            pass
        
        return None
    
    def set_value(self, section: str, key: str, value: Any) -> bool:
        """
        Belirli bir değeri ayarla
        
        Args:
            section: Bölüm adı
            key: Anahtar
            value: Değer
            
        Returns:
            Başarılı mı?
        """
        try:
            section_obj = self.get_section(ConfigSection(section))
            if section_obj and hasattr(section_obj, key):
                setattr(section_obj, key, value)
                self.save()
                return True
        except:
            pass
        
        return False
    
    def reset_to_defaults(self) -> bool:
        """
        Varsayılan ayarlara sıfırla
        
        Returns:
            Başarılı mı?
        """
        try:
            self.config = SystemConfig()
            self.save()
            logger.info("Ayarlar varsayılan değerlere sıfırlandı")
            return True
        except Exception as e:
            logger.error(f"Sıfırlama hatası: {e}")
            return False
    
    def export_to_file(self, filepath: str) -> bool:
        """
        Konfigürasyonu dosyaya dışa aktar
        
        Args:
            filepath: Dosya yolu
            
        Returns:
            Başarılı mı?
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Konfigürasyon dışa aktarıldı: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Dışa aktarma hatası: {e}")
            return False
    
    def import_from_file(self, filepath: str) -> bool:
        """
        Konfigürasyonu dosyadan içe aktar
        
        Args:
            filepath: Dosya yolu
            
        Returns:
            Başarılı mı?
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.config = SystemConfig.from_dict(data)
            self.save()
            
            logger.info(f"Konfigürasyon içe aktarıldı: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"İçe aktarma hatası: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Konfigürasyon özetini al
        
        Returns:
            Özet dict'i
        """
        return {
            'version': self.config.version,
            'camera': {
                'source': self.config.camera.source_type,
                'resolution': f"{self.config.camera.width}x{self.config.camera.height}",
                'fps': self.config.camera.fps,
            },
            'detection': {
                'model': self.config.detection.model_path,
                'device': self.config.detection.device,
                'conf_threshold': self.config.detection.conf_threshold,
            },
            'server': {
                'enabled': self.config.server.enabled,
                'host': self.config.server.host,
                'port': self.config.server.port,
            },
            'performance': {
                'target_fps': self.config.performance.target_fps,
                'dynamic_selection': self.config.performance.enable_dynamic_model_selection,
            },
        }


class SettingsValidator:
    """
    Ayarlar Doğrulayıcısı
    Konfigürasyon ayarlarını doğrular
    """
    
    @staticmethod
    def validate_camera_settings(settings: CameraSettings) -> List[str]:
        """Kamera ayarlarını doğrula"""
        errors = []
        
        if settings.width < 320 or settings.width > 4096:
            errors.append("Genişlik 320-4096 arasında olmalı")
        
        if settings.height < 240 or settings.height > 2160:
            errors.append("Yükseklik 240-2160 arasında olmalı")
        
        if settings.fps < 1 or settings.fps > 120:
            errors.append("FPS 1-120 arasında olmalı")
        
        if settings.buffer_size < 1 or settings.buffer_size > 10:
            errors.append("Buffer boyutu 1-10 arasında olmalı")
        
        return errors
    
    @staticmethod
    def validate_detection_settings(settings: DetectionSettings) -> List[str]:
        """Tespit ayarlarını doğrula"""
        errors = []
        
        if settings.conf_threshold < 0.0 or settings.conf_threshold > 1.0:
            errors.append("Güven eşiği 0.0-1.0 arasında olmalı")
        
        if settings.iou_threshold < 0.0 or settings.iou_threshold > 1.0:
            errors.append("IOU eşiği 0.0-1.0 arasında olmalı")
        
        if settings.device not in ['cuda', 'cpu']:
            errors.append("Device 'cuda' veya 'cpu' olmalı")
        
        return errors
    
    @staticmethod
    def validate_performance_settings(settings: PerformanceSettings) -> List[str]:
        """Performans ayarlarını doğrula"""
        errors = []
        
        if settings.target_fps < 1 or settings.target_fps > 120:
            errors.append("Hedef FPS 1-120 arasında olmalı")
        
        if settings.max_gpu_memory_mb < 512 or settings.max_gpu_memory_mb > 16000:
            errors.append("Maksimum GPU belleği 512-16000 MB arasında olmalı")
        
        if settings.batch_size < 1 or settings.batch_size > 128:
            errors.append("Batch boyutu 1-128 arasında olmalı")
        
        return errors
    
    @staticmethod
    def validate_all(config: SystemConfig) -> Dict[str, List[str]]:
        """Tüm ayarları doğrula"""
        errors = {
            'camera': SettingsValidator.validate_camera_settings(config.camera),
            'detection': SettingsValidator.validate_detection_settings(config.detection),
            'performance': SettingsValidator.validate_performance_settings(config.performance),
        }
        
        # Boş hata listelerini kaldır
        return {k: v for k, v in errors.items() if v}


class SettingsPresets:
    """
    Ayarlar Ön Ayarları
    Yaygın kullanım senaryoları için ön ayarlar
    """
    
    @staticmethod
    def get_preset(preset_name: str) -> Optional[SystemConfig]:
        """
        Ön ayarı al
        
        Args:
            preset_name: Ön ayar adı
            
        Returns:
            Sistem konfigürasyonu
        """
        presets = {
            'high_accuracy': SettingsPresets._high_accuracy_preset(),
            'balanced': SettingsPresets._balanced_preset(),
            'high_speed': SettingsPresets._high_speed_preset(),
            'low_power': SettingsPresets._low_power_preset(),
        }
        
        return presets.get(preset_name)
    
    @staticmethod
    def _high_accuracy_preset() -> SystemConfig:
        """Yüksek doğruluk ön ayarı"""
        config = SystemConfig()
        config.detection.model_size = "x"
        config.detection.conf_threshold = 0.5
        config.performance.target_fps = 15
        config.performance.enable_dynamic_model_selection = False
        return config
    
    @staticmethod
    def _balanced_preset() -> SystemConfig:
        """Dengeli ön ayarı"""
        config = SystemConfig()
        config.detection.model_size = "m"
        config.detection.conf_threshold = 0.45
        config.performance.target_fps = 30
        config.performance.enable_dynamic_model_selection = True
        return config
    
    @staticmethod
    def _high_speed_preset() -> SystemConfig:
        """Yüksek hız ön ayarı"""
        config = SystemConfig()
        config.detection.model_size = "s"
        config.detection.conf_threshold = 0.4
        config.performance.target_fps = 60
        config.performance.enable_dynamic_model_selection = True
        config.performance.enable_quantization = True
        return config
    
    @staticmethod
    def _low_power_preset() -> SystemConfig:
        """Düşük güç ön ayarı"""
        config = SystemConfig()
        config.detection.model_size = "n"
        config.detection.conf_threshold = 0.35
        config.detection.device = "cpu"
        config.performance.target_fps = 10
        config.performance.max_gpu_memory_mb = 1000
        return config
    
    @staticmethod
    def list_presets() -> List[str]:
        """Mevcut ön ayarları listele"""
        return ['high_accuracy', 'balanced', 'high_speed', 'low_power']
