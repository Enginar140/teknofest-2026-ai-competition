# Teknofest AI System

Havacılıkta Yapay Zeka Yarışması (OTR) için kapsamlı, modüler AI sistemi.

## 🎯 Özellikler

### 1. Nesne Tespiti
- **YOLOv8** entegrasyonu (n/s/m/l/x boyutları)
- **ByteTrack** nesne takibi
- **SAHI** (Sliced Aided Hyper Inference) büyük görüntü desteği
- Gerçek zamanlı tespit ve tracking

### 2. Pozisyon Kestirimi
- **Visual Odometry** (VO) ile kamera hareketi kestirimi
- **Kinematik EKF** (Extended Kalman Filter) ile pozisyon filtreleme
- **Smooth State Refinement (SSR)** ile titreşim giderme

### 3. Görüntü Eşleme
- **XoFTR** / **LightGlue** / **ORB** / **SIFT** destekli
- **Kalman Filter** ile feature tracking
- RANSAC tabanlı homografi kestirimi

### 4. Sunucu Bağlantısı
- **JSON API** üzerinden Teknofest sunucusuna bağlantı
- Otomatik yeniden bağlanma
- Heartbeat mekanizması
- Frame başı ve tespit sonuç gönderimi

### 5. Performans İzleme
- Real-time FPS, CPU, GPU, bellek takibi
- Performans darboğaz analizi
- Optimizasyon önerileri

### 6. Dinamik Model Seçimi
- Runtime'da model değiştirme
- Performansa göre otomatik model seçimi
- Model registry ve benchmark sistemi

### 7. PyQt6 Arayüz
- Dashboard paneli
- Kamera paneli
- Model seçim paneli
- Metrik paneli
- Sunucu bağlantı paneli
- Dosya yöneticisi

## 📁 Proje Yapısı

```
teknofest_ai_system/
├── __init__.py
├── main.py                      # Ana giriş noktası
├── core/                        # Çekirdek modül
│   ├── __init__.py
│   ├── config_manager.py        # Konfigürasyon yönetimi
│   ├── constants.py             # Sabitler
│   ├── metrics.py               # Performans metrikleri
│   └── settings.py              # Sistem ayarları
├── data/                        # Veri işleme
│   ├── __init__.py
│   ├── preprocessor.py          # Görüntü ön işleme
│   ├── augmentation.py          # Veri artırma
│   └── dataset.py               # Dataset yönetimi
├── models/                      # AI modelleri
│   ├── __init__.py
│   ├── detection.py             # Nesne tespiti (YOLOv8)
│   ├── position.py              # Pozisyon kestirimi
│   ├── matching.py              # Görüntü eşleme
│   └── management.py            # Model yönetimi
├── server/                      # Sunucu bağlantısı
│   ├── __init__.py
│   └── connection.py            # TCP/IP bağlantı
├── camera/                      # Kamera entegrasyonu
│   ├── __init__.py
│   └── processor.py             # Kamera ve frame işleme
├── ui/                          # PyQt6 arayüz
│   ├── __init__.py
│   ├── main_window.py           # Ana pencere
│   └── panels/                  # UI panelleri
├── testing/                     # Test ve benchmark
│   ├── __init__.py
│   └── runner.py                # Test ve benchmark
├── config/                      # Konfigürasyon dosyaları
│   └── default_config.json
├── requirements.txt
└── README.md
```

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- CUDA 11.8+ (GPU için)
- Windows 10/11 veya Linux

### Adımlar

```bash
# Depoyu klonlayın
git clone <repo-url>

# Sanal ortam oluşturun
python -m venv venv

# Sanal ortamı aktifleştirin
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Model dosyalarını indirin (otomatik)
python main.py --download-models
```

## 📖 Kullanım

### GUI Uygulaması

```bash
python main.py
```

### Komut Satırı

```bash
# Kamera ile çalıştır
python main.py --camera 0 --model yolov8l.pt

# Video dosyası ile çalıştır
python main.py --video path/to/video.mp4

# Sunucuya bağlan
python main.py --server localhost:10000 --team-id YOUR_TEAM_ID
```

### Python API

```python
from teknofest_ai_system import (
    ObjectTracker,
    PositionEstimator,
    CameraManager,
    ConnectionManager,
)

# Kamera yöneticisi
camera_manager = CameraManager()
camera_manager.initialize(
    camera_config=config.camera,
    detector=tracker,
    position_estimator=position_estimator,
    matcher=matcher,
    connection_manager=connection_manager,
    performance_monitor=monitor,
)

# Başlat
camera_manager.start()
```

## 🔧 Konfigürasyon

Konfigürasyon dosyası: `config/system_config.json`

```json
{
  "camera": {
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "detection": {
    "model_path": "yolov8l.pt",
    "device": "cuda",
    "conf_threshold": 0.45
  },
  "server": {
    "enabled": true,
    "host": "localhost",
    "port": 10000
  }
}
```

## 📊 Performans

| Model | FPS | GPU Bellek | mAP |
|-------|-----|-----------|-----|
| YOLOv8n | 120 | 1GB | 0.45 |
| YOLOv8s | 80 | 2GB | 0.55 |
| YOLOv8m | 50 | 4GB | 0.65 |
| YOLOv8l | 30 | 6GB | 0.72 |
| YOLOv8x | 20 | 8GB | 0.76 |

## 🧪 Test

```bash
# Sistem testleri
python -m teknofest_ai_system.testing.runner --test-all

# Benchmark
python -m teknofest_ai_system.testing.runner --benchmark
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 👥 Takım

- Teknofest AI Team

## 🙏 Teşekkürler

- Ultralytics - YOLOv8
- ByteTrack
- SAHI
- PyQt6
