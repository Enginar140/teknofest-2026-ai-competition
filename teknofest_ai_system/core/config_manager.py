"""
Konfigürasyon Yöneticisi - JSON dosyasından ayarları yükler ve yönetir
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Sistem konfigürasyonunu yönetir"""
    
    def __init__(self, config_path: str = "./config/default_config.json"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Konfigürasyon dosyasını yükle"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Konfigürasyon yüklendi: {self.config_path}")
            else:
                logger.warning(f"Konfigürasyon dosyası bulunamadı: {self.config_path}")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Konfigürasyon yükleme hatası: {e}")
            self.config = self._get_default_config()
    
    def save_config(self) -> None:
        """Konfigürasyonu dosyaya kaydet"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Konfigürasyon kaydedildi: {self.config_path}")
        except Exception as e:
            logger.error(f"Konfigürasyon kaydetme hatası: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Konfigürasyon değeri al (nokta notasyonu destekler)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Konfigürasyon değeri ayarla (nokta notasyonu destekler)"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Konfigürasyon güncellendi: {key} = {value}")
    
    def get_camera_config(self, camera_name: str) -> Optional[Dict]:
        """Kamera konfigürasyonunu al"""
        return self.get(f"camera.cameras.{camera_name}")
    
    def get_model_config(self, task: str) -> Optional[Dict]:
        """Model konfigürasyonunu al"""
        return self.get(f"models.{task}")
    
    def get_selected_camera(self) -> str:
        """Seçili kamerayı al"""
        return self.get("camera.selected", "rgb_2024")
    
    def set_selected_camera(self, camera_name: str) -> None:
        """Seçili kamerayı ayarla"""
        self.set("camera.selected", camera_name)
    
    def get_selected_model(self, task: str) -> str:
        """Seçili modeli al"""
        return self.get(f"models.{task}.selected", "yolov8l")
    
    def set_selected_model(self, task: str, model_name: str) -> None:
        """Seçili modeli ayarla"""
        self.set(f"models.{task}.selected", model_name)
    
    def get_server_config(self) -> Dict:
        """Sunucu konfigürasyonunu al"""
        return self.get("server", {})
    
    def get_recording_config(self) -> Dict:
        """Kayıt konfigürasyonunu al"""
        return self.get("recording", {})
    
    def get_ui_config(self) -> Dict:
        """UI konfigürasyonunu al"""
        return self.get("ui", {})
    
    def get_processing_config(self) -> Dict:
        """İşleme konfigürasyonunu al"""
        return self.get("processing", {})
    
    def get_all_cameras(self) -> Dict:
        """Tüm kameraları al"""
        return self.get("camera.cameras", {})
    
    def get_all_models(self, task: str) -> list:
        """Görev için tüm modelleri al"""
        return self.get(f"models.{task}.options", [])
    
    def _get_default_config(self) -> Dict:
        """Varsayılan konfigürasyonu döndür"""
        return {
            "system": {
                "name": "TEKNOFEST 2026 AI System",
                "version": "1.0.0",
                "theme": "dark"
            },
            "camera": {
                "selected": "rgb_2024",
                "cameras": {}
            },
            "models": {
                "task1": {"selected": "yolov8l", "options": []},
                "task2": {"selected": "vo_ekf", "options": []},
                "task3": {"selected": "xoftr", "options": []}
            },
            "server": {
                "url": "http://127.0.0.1:5000/",
                "timeout": 30
            },
            "recording": {
                "enabled": True,
                "output_dir": "./results/"
            }
        }


# Global konfigürasyon örneği
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Global konfigürasyon yöneticisini al"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_path: str) -> ConfigManager:
    """Konfigürasyon yöneticisini başlat"""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager
