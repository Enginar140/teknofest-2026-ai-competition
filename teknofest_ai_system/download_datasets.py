"""
Teknofest 2026 Veri Setlerini Indir
Google Drive'dan dataset indirme scripti
"""

import os
import sys
import io
import logging
from pathlib import Path

# UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Logging ayarı
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Google Drive folder ID
DRIVE_FOLDER_ID = "18_VqLBbyTubVSWAXG_CgmuJWGCx0mcBd"

# Dataset bilgileri
DATASETS = {
    "VisDrone2019": {
        "description": "UAV platformu, 10K+ görüntü, 10 sınıf",
        "url": "https://drive.google.com/uc?id=1_dataset_id_visdrone"
    },
    "UAVDT": {
        "description": "UAV'dan araç tespiti, hareket etiketli",
        "url": "https://drive.google.com/uc?id=1_dataset_id_uavdt"
    },
    "HIT-UAV": {
        "description": "RGB+Thermal hibrit UAV veri seti",
        "url": "https://drive.google.com/uc?id=1_dataset_id_hituav"
    },
    "Sentetik": {
        "description": "AirSim simülatörü, edge case kapsamı",
        "url": "https://drive.google.com/uc?id=1_dataset_id_synthetic"
    }
}

def check_gdown():
    """gdown kütüphanesini kontrol et"""
    try:
        import gdown
        logger.info("✓ gdown kütüphanesi yüklü")
        return True
    except ImportError:
        logger.error("✗ gdown kütüphanesi yüklü değil")
        logger.info("Kurulum: pip install gdown")
        return False

def download_from_drive(folder_id, output_dir):
    """Google Drive'dan klasörü indir"""
    try:
        import gdown
        
        logger.info(f"Google Drive'dan indiriliyor: {folder_id}")
        logger.info(f"Hedef dizin: {output_dir}")
        
        # Klasörü indir
        gdown.download_folder(
            id=folder_id,
            output=output_dir,
            quiet=False,
            use_cookies=False
        )
        
        logger.info("✓ İndirme tamamlandı")
        return True
        
    except Exception as e:
        logger.error(f"İndirme hatası: {e}")
        return False

def setup_dataset_structure(data_dir):
    """Dataset klasör yapısını oluştur"""
    logger.info("Dataset klasör yapısı oluşturuluyor...")
    
    splits = ['train', 'val', 'test']
    subdirs = ['images', 'labels']
    
    for split in splits:
        for subdir in subdirs:
            path = Path(data_dir) / split / subdir
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Oluşturuldu: {path}")

def list_available_datasets():
    """Mevcut datasetleri listele"""
    logger.info("\n" + "="*60)
    logger.info("TEKNOFEST 2026 - Mevcut Veri Setleri")
    logger.info("="*60)
    
    for name, info in DATASETS.items():
        logger.info(f"\n📦 {name}")
        logger.info(f"   Açıklama: {info['description']}")
    
    logger.info("\n" + "="*60)
    logger.info("Google Drive Linki:")
    logger.info(f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}")
    logger.info("="*60 + "\n")

def main():
    """Ana fonksiyon"""
    logger.info("Teknofest 2026 Dataset İndirici")
    logger.info("="*60)
    
    # Mevcut datasetleri göster
    list_available_datasets()
    
    # gdown kontrol et
    if not check_gdown():
        logger.warning("gdown yüklemek için: pip install gdown")
        sys.exit(1)
    
    # Çıkış dizini
    output_dir = Path("./data/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nDataset indirme konumu: {output_dir}")
    logger.info("\nSeçenekler:")
    logger.info("1. Google Drive'dan tüm datasetleri indir")
    logger.info("2. Klasör yapısını oluştur (boş)")
    logger.info("3. Çıkış")
    
    choice = input("\nSeçiminiz (1-3): ").strip()
    
    if choice == "1":
        logger.info("\nGoogle Drive'dan indiriliyor...")
        if download_from_drive(DRIVE_FOLDER_ID, str(output_dir)):
            setup_dataset_structure(output_dir)
            logger.info("✓ Tüm işlemler tamamlandı!")
        else:
            logger.error("✗ İndirme başarısız oldu")
            sys.exit(1)
    
    elif choice == "2":
        logger.info("\nKlasör yapısı oluşturuluyor...")
        setup_dataset_structure(output_dir)
        logger.info("✓ Klasör yapısı oluşturuldu!")
        logger.info(f"Veri dosyalarını şu konuma kopyalayın: {output_dir}")
    
    elif choice == "3":
        logger.info("Çıkılıyor...")
        sys.exit(0)
    
    else:
        logger.error("Geçersiz seçim!")
        sys.exit(1)

if __name__ == "__main__":
    main()
