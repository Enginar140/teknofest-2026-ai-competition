"""
Hazır Eğitilmiş YOLOv8 Modellerini İndir
Ultralytics'ten resmi YOLOv8 modellerini indir
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hazır modeller
PRETRAINED_MODELS = {
    "yolov8n": {
        "name": "YOLOv8 Nano",
        "size": "3.2 MB",
        "speed": "80 ms",
        "mAP": "37.3%",
        "description": "En hızlı, en hafif model. Edge devices için ideal."
    },
    "yolov8s": {
        "name": "YOLOv8 Small",
        "size": "22.5 MB",
        "speed": "128 ms",
        "mAP": "44.9%",
        "description": "Dengeli hız ve doğruluk. Çoğu uygulamada tercih edilen."
    },
    "yolov8m": {
        "name": "YOLOv8 Medium",
        "size": "49.7 MB",
        "speed": "234 ms",
        "mAP": "50.2%",
        "description": "Daha yüksek doğruluk, daha yavaş."
    },
    "yolov8l": {
        "name": "YOLOv8 Large",
        "size": "83.7 MB",
        "speed": "375 ms",
        "mAP": "52.9%",
        "description": "Yüksek doğruluk, daha fazla GPU gerekli."
    },
    "yolov8x": {
        "name": "YOLOv8 Extra Large",
        "size": "135.4 MB",
        "speed": "479 ms",
        "mAP": "53.9%",
        "description": "En yüksek doğruluk, en yavaş."
    }
}

def check_ultralytics():
    """ultralytics kütüphanesini kontrol et"""
    try:
        from ultralytics import YOLO
        logger.info("✓ ultralytics kütüphanesi yüklü")
        return True
    except ImportError:
        logger.error("✗ ultralytics kütüphanesi yüklü değil")
        logger.info("Kurulum: pip install ultralytics")
        return False

def download_model(model_name):
    """Modeli indir"""
    try:
        from ultralytics import YOLO
        
        logger.info(f"\n{model_name} modeli indiriliyor...")
        
        # Model otomatik olarak ~/.yolov8/ dizinine indirilir
        model = YOLO(f"{model_name}.pt")
        
        logger.info(f"✓ {model_name} başarıyla indirildi")
        logger.info(f"  Konum: {model.model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ İndirme hatası: {e}")
        return False

def list_models():
    """Mevcut modelleri listele"""
    logger.info("\n" + "="*70)
    logger.info("TEKNOFEST 2026 - Hazır Eğitilmiş YOLOv8 Modelleri")
    logger.info("="*70)
    
    for idx, (model_id, info) in enumerate(PRETRAINED_MODELS.items(), 1):
        logger.info(f"\n{idx}. {info['name']} ({model_id})")
        logger.info(f"   Boyut: {info['size']}")
        logger.info(f"   Hız: {info['speed']}")
        logger.info(f"   mAP: {info['mAP']}")
        logger.info(f"   Açıklama: {info['description']}")
    
    logger.info("\n" + "="*70)

def main():
    """Ana fonksiyon"""
    logger.info("Teknofest 2026 - Hazır Model İndirici")
    logger.info("="*70)
    
    # ultralytics kontrol et
    if not check_ultralytics():
        logger.warning("ultralytics yüklemek için: pip install ultralytics")
        sys.exit(1)
    
    # Modelleri listele
    list_models()
    
    logger.info("\nSeçenekler:")
    logger.info("1. Tüm modelleri indir")
    logger.info("2. Belirli modelleri seç")
    logger.info("3. Çıkış")
    
    choice = input("\nSeçiminiz (1-3): ").strip()
    
    if choice == "1":
        logger.info("\nTüm modeller indiriliyor...")
        success_count = 0
        for model_name in PRETRAINED_MODELS.keys():
            if download_model(model_name):
                success_count += 1
        
        logger.info(f"\n✓ {success_count}/{len(PRETRAINED_MODELS)} model başarıyla indirildi")
    
    elif choice == "2":
        logger.info("\nMevcut modeller:")
        for idx, model_name in enumerate(PRETRAINED_MODELS.keys(), 1):
            logger.info(f"{idx}. {model_name}")
        
        model_choice = input("\nModel numarası girin (1-5): ").strip()
        
        try:
            model_idx = int(model_choice) - 1
            if 0 <= model_idx < len(PRETRAINED_MODELS):
                model_name = list(PRETRAINED_MODELS.keys())[model_idx]
                if download_model(model_name):
                    logger.info("✓ Model başarıyla indirildi!")
            else:
                logger.error("Geçersiz seçim!")
        except ValueError:
            logger.error("Geçersiz giriş!")
    
    elif choice == "3":
        logger.info("Çıkılıyor...")
        sys.exit(0)
    
    else:
        logger.error("Geçersiz seçim!")
        sys.exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("Modeller ~/.yolov8/ dizinine indirildi")
    logger.info("Sistem otomatik olarak bu modelleri kullanabilir")
    logger.info("="*70)

if __name__ == "__main__":
    main()
