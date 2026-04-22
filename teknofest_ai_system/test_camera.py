"""
Kamera Testi - Gerçek Zamanlı Nesne Tespiti
YOLOv8 ile webcam'den canlı nesne tespiti
"""

import cv2
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_camera():
    """Kamerayı test et"""
    try:
        from ultralytics import YOLO
        
        logger.info("="*60)
        logger.info("Teknofest 2026 - Kamera Testi")
        logger.info("="*60)
        
        # Model yükle
        model_path = Path("models/yolov8n.pt")
        if not model_path.exists():
            logger.error(f"Model bulunamadı: {model_path}")
            sys.exit(1)
        
        logger.info(f"Model yükleniyor: {model_path}")
        model = YOLO(str(model_path))
        logger.info("✓ Model yüklendi")
        
        # Kamerayı aç
        logger.info("\nKamera açılıyor...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("✗ Kamera açılamadı!")
            sys.exit(1)
        
        logger.info("✓ Kamera açıldı")
        logger.info("\nKontroller:")
        logger.info("  - 'q' tuşu: Çıkış")
        logger.info("  - 's' tuşu: Ekran görüntüsü kaydet")
        logger.info("  - 'c' tuşu: Kamera değiştir")
        logger.info("\n" + "="*60)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                logger.error("Frame okunamadı")
                break
            
            frame_count += 1
            
            # Nesne tespiti yap
            results = model(frame, verbose=False)
            
            # Sonuçları çiz
            annotated_frame = results[0].plot()
            
            # FPS hesapla
            if frame_count % 30 == 0:
                logger.info(f"Frame: {frame_count}")
            
            # Ekranda göster
            cv2.imshow("Teknofest 2026 - Nesne Tespiti", annotated_frame)
            
            # Tuş kontrolü
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Çıkılıyor...")
                break
            elif key == ord('s'):
                filename = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                logger.info(f"✓ Ekran görüntüsü kaydedildi: {filename}")
            elif key == ord('c'):
                logger.info("Kamera değiştiriliyor...")
                cap.release()
                camera_id = int(input("Kamera ID girin (0, 1, 2...): "))
                cap = cv2.VideoCapture(camera_id)
                if not cap.isOpened():
                    logger.error(f"Kamera {camera_id} açılamadı")
                    break
                logger.info(f"✓ Kamera {camera_id} açıldı")
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"\n✓ Test tamamlandı ({frame_count} frame işlendi)")
        
    except ImportError:
        logger.error("ultralytics kütüphanesi yüklü değil")
        logger.info("Kurulum: pip install ultralytics")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Hata: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_camera()
