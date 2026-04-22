# Kamera Paneli Görev Bazlı Yeniden Yapılandırma Planı

## 📊 Mevcut Durum Analizi

### Mevcut Kontroller (Görev Bazlı Değerlendirme)

| Kontrol | Görev 1 (Detection) | Görev 2 (Position) | Görev 3 (Matching) | Durum |
|---------|---------------------|-------------------|-------------------|--------|
| source_type_combo | ✅ Gerekli | ✅ Gerekli | ✅ Gerekli | Ortak |
| camera_id_spin | ✅ Gerekli | ✅ Gerekli | ✅ Gerekli | Ortak |
| video_path_label | ✅ Gerekli | ✅ Gerekli | ✅ Gerekli | Ortak |
| select_video_btn | ✅ Gerekli | ✅ Gerekli | ✅ Gerekli | Ortak |
| model_combo | ✅ Gerekli | ❌ Gerekli DEĞIL | ❌ Gerekli DEĞIL | Sadece Görev 1 |
| conf_spin | ✅ Gerekli | ❌ Gerekli DEĞIL | ❌ Gerekli DEĞIL | Sadece Görev 1 |
| iou_spin | ✅ Gerekli | ❌ Gerekli DEĞIL | ⚠️ Opşyonel | Sadece Görev 1 |
| camera_angle_spin | ❌ Gerekli DEĞIL | ❌ Gerekli DEĞIL | ❌ Gerekli DEĞIL | KALDIRILACAK |
| detection_check | ✅ Gerekli | ❌ Gerekli DEĞIL | ❌ Gerekli DEĞIL | Sadece Görev 1 |
| tracking_check | ✅ Gerekli | ❌ Gerekli DEĞIL | ❌ Gerekli DEĞIL | Sadece Görev 1 |

## 🎯 Görev Bazlı Gerekli Parametreler

### Görev 1: Nesne Tespiti (Detection)
```
ORtak Kontroller:
- Kaynak seçimi (Kamera/Video)
- Kamera ID
- Video dosyası seçimi

Görev 1 Özel Kontroller:
- Model seçimi (yolov8n, yolov8s, yolov8m, yolov8l)
- Confidence Threshold (1-100%)
- IoU Threshold (1-100%)
- Tespit aktif/pasif
- Tracking aktif/pasif

Görüntüleme:
- 4 sınıf tespit sayıları (Taşıt, İnsan, UAP, UAİ)
- Hareket durumu göstergesi
- İniş durumu göstergesi (UAP/UAİ için)
```

### Görev 2: Pozisyon Tespiti (Position)
```
ORtak Kontroller:
- Kaynak seçimi (Kamera/Video)
- Kamera ID
- Video dosyası seçimi

Görev 2 Özel Kontroller:
- Kamera seçimi (RGB_2024, Thermal_2025, RGB_4K_2025)
- Kamera kalibrasyon parametreleri (otomatik yükleme)
- GPS health durumu göstergesi
- Visual Odometry seçenekleri
- EKF (Extended Kalman Filter) ayarları
- Referans yükseklik (z0)

Kamera Kalibrasyon Parametreleri (2025):
RGB Camera:
- FocalLength: [2792.2, 2795.2]
- PrincipalPoint: [1988.0, 1562.2]
- ImageSize: [3000, 4000]
- RadialDistortion: [0.0798, -0.1867]

Thermal Camera:
- FocalLength: [731.8, 732.0]
- PrincipalPoint: [319.2, 251.2]
- ImageSize: [512, 640]
- RadialDistortion: [-0.3507, 0.1137]

Görüntüleme:
- Pozisyon bilgisi (X, Y, Z)
- GPS Health durumu
- RMSE değerleri
- Referans konum vs Kestirilmiş konum
```

### Görev 3: Görüntü Eşleme (Matching)
```
ORtak Kontroller:
- Kaynak seçimi (Kamera/Video)
- Kamera ID
- Video dosyası seçimi

Görev 3 Özel Kontroller:
- Referans nesne yükleme
- Feature matcher seçimi (ORB, SIFT, XoFTR, LightGlue)
- Matching threshold (IoU/eşleşme oranı)
- Cross-modal eşleşme aktif/pasif
- Ensemble model aktif/pasif
- Max eşleşme sayısı

Görüntüleme:
- Tespit edilen referans nesneler
- Eşleşme skorları
- Homografi matrisi göstergesi
```

## 🗑️ Kaldırılacak Kontroller

- `camera_angle_spin` - Kamera açısı manuel ayarı gerekli değil (kalibrasyon dosyasından alınıyor)

## 📋 Yeni UI Yapısı

### Ortak Panel (Her Görevde Görünür)
```
┌─────────────────────────────────────────┐
│ 📹 Video/Kamera Kaynağı                │
│ • Kaynak Tipi: [Kamera/Video]          │
│ • Kamera ID: [0-10]                    │
│ • Video: [Seçilmedi] [📂 Seç]         │
└─────────────────────────────────────────┘
```

### Görev 1 Paneli (Sadece Detection Modunda Görünür)
```
┌─────────────────────────────────────────┐
│ 🔍 Nesne Tespiti Ayarları               │
│ • Model: [yolov8n▼]                    │
│ • Conf: [45%]                          │
│ • IoU: [50%]                           │
│ ☑ Tespit Aktif                         │
│ ☐ Tracking                             │
└─────────────────────────────────────────┘
```

### Görev 2 Paneli (Sadece Position Modunda Görünür)
```
┌─────────────────────────────────────────┐
│ 📍 Pozisyon Tespiti Ayarları            │
│ • Kamera: [RGB_2025▼]                  │
│   - Focal: [2792.2, 2795.2]           │
│   - Principal: [1988.0, 1562.2]        │
│ • GPS Health: [Gelen bilgilere göre]    │
│ • VO Method: [ORB▼]                    │
│ • EKF Aktif: ☑                         │
└─────────────────────────────────────────┘
```

### Görev 3 Paneli (Sadece Matching Modunda Görünür)
```
┌─────────────────────────────────────────┐
│ 🔗 Görüntü Eşleme Ayarları              │
│ • Referans Nesne: [Yükle▼]            │
│ • Matcher: [XoFTR▼]                    │
│ • Match Threshold: [0.7]               │
│ ☐ Cross-modal Matching                 │
│ ☐ Ensemble Model                       │
└─────────────────────────────────────────┘
```

## 🎨 Görünürlük Kuralları

1. **Ortak Panel**: Her zaman görünür
2. **Görev 1 Paneli**: `self.current_task == 'detection'` ise görünür
3. **Görev 2 Paneli**: `self.current_task == 'position'` ise görünür
4. **Görev 3 Paneli**: `self.current_task == 'matching'` ise görünür

## 📝 Değişiklik Listesi

1. `init_ui()` metodunda görev bazlı paneller oluştur
2. `set_task()` metodunda görünürliği güncelle
3. `camera_angle_spin` kontrolünü kaldır
4. Görev 2 için kamera kalibrasyon parametrelerini ekle
5. Görev 3 için referans nesne yükleme ekle
6. Her görev için özel görüntüleme widget'ları ekle
