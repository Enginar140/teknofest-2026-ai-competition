# BAŞARIY ARTIRMA STRATEJİSİ - OTR Referans Analizi

## 1. MEVCUT DURUM vs OTR GEREKSINIMLERI

### Mevcut Yapı (teknofest_project)
- Temel model sınıfları: ObjectDetection, PositionEstimation, ImageMatching
- Placeholder implementasyonlar
- Tek-process mimarisi

### OTR Gereksinimleri
- **4-Process Asenkron Mimari** (GIL izolasyonu)
- **Donanım**: RTX 4060 8GB, 133ms/frame bütçe
- **Veri Artırımı**: Random Erasing, GridMask, Cutout, Albumentations
- **Özel Algoritmalar**: SSR, ByteTrack, SAHI, Kinematik EKF

---

## 2. BAŞARIY ARTIRMA ALANLARI

### A. MİMARİ IYILEŞTIRMELER

#### 1. 4-Process Asenkron Mimari Implementasyonu
```
Process A: Frame Capture (Kamera/Video)
Process B: GPU Inference (YOLOv8m, ORB-SLAM2)
Process C: VO/SLAM Processing (ORB-SLAM2 VO)
Process D: JSON Management & Timeout Logic
```

**Faydalar:**
- GIL kısıtını ortadan kaldırır
- Zero-copy IPC ile shared_memory kullanımı
- 133ms bütçeyi tutturabilir

**Implementasyon Adımları:**
1. `src/core/multiprocess_manager.py` oluştur
2. `multiprocessing.shared_memory` ile buffer yönetimi
3. Process-to-process queue'lar
4. Graceful shutdown mekanizması

---

### B. NESNE TESPİTİ MODÜLÜ (Görev 1) - %25

#### 1. YOLOv8m TensorRT FP16 Optimizasyonu
**Hedef**: 8-12 ms/kare latency

```python
# Yapılacaklar:
- YOLOv8m → ONNX → TensorRT FP16 dönüşümü
- Batch inference (4-8 batch size)
- Dynamic shape handling
- Inference time profiling
```

#### 2. ByteTrack Entegrasyonu
**Amaç**: Düşük güven skorlu tespitleri (< 0.25) yörüngelerle eşleştir

```python
# Faydalar:
- False Negative azaltma
- Çerçeve kenarından çıkan nesneleri takip etme
- mAP artışı (Zhang et al. ECCV 2022: +2-3%)
```

#### 3. SAHI (Sliced Aided Hyper Inference)
**Hedef**: Küçük nesneleri daha iyi tespit etme

```python
# Konfigürasyon:
- Slice size: 640x640
- Overlap: 0.1
- Merge IoU threshold: 0.5
- Latency bütçe: 60-70 ms/kare
```

#### 4. Veri Artırımı Stratejisi
```python
# Albumentations Pipeline:
- RandomRain (yağmur simülasyonu)
- RandomFog (sis simülasyonu)
- RandomSunFlare (güneş parlaması)
- Random Erasing (occlusion)
- GridMask (parçalı silme)
- Directional Motion Blur (7.5 FPS etkisi)
- Dead Pixel Masking (ölü piksel)
```

#### 5. İniş Alanı Uygunluk Hiyerarşisi
```python
# 4-Adımlı Karar Mantığı:
1. Ellipse Fitting: UAP/UAİ dairesine elips oturt
2. Ground Footprint: Bounding box alt kenarı elips içinde mi?
3. Anomali Tespiti: Piksel yoğunluk analizi (tanımsız nesneler)
4. Karar: Üç koşul negatifse → landing_status = 1
```

#### 6. Scooter Sınıflandırması
```python
# Dinamik Eşik:
- ORB-SLAM2 z-ekseni çıktısından irtifa hesapla
- scale_px_per_m = f / Z (kamera focal length / irtifa)
- Bounding box boyutunu normalize et
- Sürücülü/sürücüsüz karar ver
```

---

### C. POZİSYON KESTİRİMİ MODÜLÜ (Görev 2) - %40

#### 1. Semantik Ölçek Kurtarma (SSR) - IMU-Free
**Temel Fikir**: UAP/UAİ (4.5m) fiziksel boyutunu görsel çapa olarak kullan

```python
# Adımlar:
1. Ray Casting: Bounding box köşelerini normalize görüntü düzlemine yansıt
2. Dünya Koordinatı: ORB-SLAM2 rotasyon matrisiyle dönüştür
3. Ground Plane: Z=0 düzlemiyle kesişim noktalarını hesapla
4. Scale Katsayısı: Scale = 4.5m / D_vo
5. Moving Average: N=10 kare penceresiyle düzleştir
6. Outlier Filtresi: Fiziksel olarak imkansız sıçramaları reddet
```

**Matematiksel Formülasyon:**
```
d = K⁻¹ · [u, v, 1]ᵀ  (Ray casting)
d_world = Rᵀ · d      (Dünya koordinatı)
Scale = 4.5 / D_vo    (Ölçek katsayısı)
```

#### 2. Kinematik EKF
**Durum Vektörü**: [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]

```python
# İsınma Fazı (İlk 450 kare):
- GPS referans koordinatlarından V ve A vektörleri hesapla
- Sabit İvme (CA) modeline oturt

# Tahmin Adımı (GPS Kesilince):
- X_new = X_old + V·Δt + ½·A·Δt²
- Kovaryans matrisi belirsizliği izle

# Göreli Güncelleme (Her Karede):
- ORB-SLAM2 bağıl yer değiştirmesi ölçüm olarak ver
- Anlık scale katsayısıyla metrik uzaya taşı

# Mutlak Güncelleme (SSR Tetiklenince):
- UAP/UAİ tespitinde SSR çalış
- EKF kovaryansı sıfırla (absolute reset)
```

#### 3. ORB-SLAM2 Entegrasyonu
```python
# Dinamik FAST Threshold:
- Kare Shannon entropisi hesapla
- Düşük entropi (sis): threshold yükselt (overdetection önle)
- Yüksek entropi (yağmur): threshold düşür (feature kaybı önle)
- Mapping: threshold = base_thr + α·(H_max – H_frame)

# Görüntü Donması Tespiti:
- İki ardışık kare SSIM > 0.995 → donmuş
- VO güncellemesi kes, EKF kinematik predict devam et
- Görüntü geri gelince relocalization yap

# Ölü Piksel Maskeleme:
- Kalibrasyon aşamasında statik maske oluştur
- Feature extraction öncesi maskelenen bölgeleri sustur
```

#### 4. Metrikleme
```python
# Ortalama Hata (RMSE):
- Tahmin edilen konum vs referans konum
- X, Y, Z ayrı ayrı hesapla
- Toplam RMSE = sqrt(RMSE_x² + RMSE_y² + RMSE_z²)
```

---

### D. GÖRÜNTÜ EŞLEME MODÜLÜ (Görev 3) - %25

#### 1. XoFTR (Çapraz Modalite Transformatör)
**Avantajlar:**
- Termal-RGB eşleme için optimize
- ONNX + TensorRT ile hızlı
- Pseudo-thermal artırmayla eğitilmiş

```python
# Implementasyon:
- ONNX model yükle
- TensorRT optimize et
- Batch processing
```

#### 2. LightGlue (Fallback)
**Avantajlar:**
- SuperGlue'dan 3× hızlı
- CLAHE normalizasyonu ile termal-RGB farkı azalt

```python
# Adaptif CLAHE:
- Kare histogram dağılımı analiz et
- clip_limit = f(histogram_entropy)
- Sis yoğunluğuna göre dinamik ayar
```

#### 3. Kalman Tracker (Fallback)
**Amaç**: Eşleme algoritmaları başarısız olunca nesne konumunu tahmin et

```python
# Mekanizma:
- Son bilinen konum + VO çıktısı (ego-motion)
- Bounding box tahmin et
- Düşük güven skoru ile işaretle
```

#### 4. Ensemble Yöntemi
```python
# Birleştirme Stratejisi:
1. XoFTR çalış → başarılı ise sonuç döndür
2. Başarısız → LightGlue çalış
3. LightGlue başarısız → Kalman Tracker tahmin et
4. Güven skorları: XoFTR > LightGlue > Kalman
```

---

### E. VERI ARTIRIMI STRATEJISI

#### 1. Fiziksel Hava Koşulu Artırımı
```python
# Albumentations Pipeline:
transforms = A.Compose([
    A.RandomRain(p=0.3),           # Yağmur
    A.RandomFog(p=0.3),            # Sis
    A.RandomSunFlare(p=0.2),       # Güneş parlaması
    A.GaussNoise(p=0.2),           # Gürültü
    A.MotionBlur(p=0.2),           # Hareket bulanıklığı
    A.CoarseDropout(p=0.2),        # Dead pixel
])
```

#### 2. Uzamsal Artırım
```python
# Occlusion Handling:
- Random Erasing: %15 mAP artışı (Zhong et al. AAAI 2020)
- GridMask: Dengeli bilgi silimi
- Cutout: Parçalı silme
```

#### 3. Eğitim Stratejisi
```python
# Aşamalar:
1. COCO pre-training (YOLOv8m)
2. VisDrone fine-tuning (UAV veri seti)
3. TEKNOFEST veri seti eğitimi
4. Ensemble modeli oluşturma (3-5 model)
```

---

### F. SUNUCU BAĞLANTI ARAYÜZÜ

#### 1. JSON API Yönetimi
```python
# Request Format:
{
    "task": 1,  # 1: Nesne Tespiti, 2: Pozisyon, 3: Görüntü Eşleme
    "frame_id": 123,
    "image_data": "base64_encoded"
}

# Response Format:
{
    "task": 1,
    "frame_id": 123,
    "detections": [...],
    "metrics": {...},
    "confidence": 0.95
}
```

#### 2. Hata Yönetimi
```python
# Retry Mekanizması:
- Exponential backoff
- Max retry: 3
- Timeout: 5 saniye

# Bağlantı Durumu:
- Connected / Disconnected
- Last heartbeat
- Queue size
```

---

### G. ARAYÜZ GELİŞTİRME (PyQt6)

#### 1. Sekme Sistemi
```
Tab 1: Görev 1 (Nesne Tespiti)
  - Model seçim (YOLOv8n/s/m/l)
  - Gerçek zamanlı video
  - mAP metriği
  - Bounding box görselleştirme

Tab 2: Görev 2 (Pozisyon Kestirimi)
  - ORB-SLAM2 tracking
  - SSR scale değeri
  - EKF tahminleri
  - 3D konum grafiği

Tab 3: Görev 3 (Görüntü Eşleme)
  - Referans görüntü yükleme
  - Eşleşme skoru
  - Feature noktaları
  - Ensemble sonuçları

Tab 4: Ayarlar
  - Kamera kalibrasyonu
  - Model parametreleri
  - Sunucu bağlantısı
  - Veri kayıt seçenekleri
```

#### 2. Gerçek Zamanlı Grafikler
```python
# Görev 1:
- mAP vs Epoch
- Confidence distribution
- Class-wise AP

# Görev 2:
- X, Y, Z konum vs Zaman
- Hız vektörü
- RMSE vs Zaman

# Görev 3:
- Eşleşme skoru vs Zaman
- Feature sayısı
- Güven değeri
```

---

## 3. IMPLEMENTASYON ÖNCELIK SIRASI

### Faz 1: Temel Altyapı (Hafta 1-2)
- [ ] 4-Process mimarisi
- [ ] Shared memory yönetimi
- [ ] Process communication queues
- [ ] Graceful shutdown

### Faz 2: Nesne Tespiti (Hafta 2-3)
- [ ] YOLOv8m TensorRT FP16
- [ ] ByteTrack entegrasyonu
- [ ] SAHI implementasyonu
- [ ] Veri artırımı pipeline

### Faz 3: Pozisyon Kestirimi (Hafta 3-4)
- [ ] ORB-SLAM2 entegrasyonu
- [ ] Semantik Ölçek Kurtarma (SSR)
- [ ] Kinematik EKF
- [ ] Dinamik FAST threshold

### Faz 4: Görüntü Eşleme (Hafta 4-5)
- [ ] XoFTR entegrasyonu
- [ ] LightGlue fallback
- [ ] Kalman Tracker
- [ ] Ensemble yöntemi

### Faz 5: Arayüz & Entegrasyon (Hafta 5-6)
- [ ] PyQt6 arayüz
- [ ] Sunucu bağlantısı
- [ ] Metrikleme sistemi
- [ ] End-to-end test

---

## 4. PERFORMANS HEDEFLERI

### Görev 1: Nesne Tespiti
- **mAP@0.5**: > 45% (VisDrone benchmark)
- **Latency**: 8-12 ms (YOLOv8m) + 60-70 ms (SAHI)
- **FPS**: 7.5 FPS (133ms bütçe)

### Görev 2: Pozisyon Kestirimi
- **RMSE**: < 2 meter
- **Latency**: < 50 ms
- **Scale accuracy**: ±10%

### Görev 3: Görüntü Eşleme
- **Eşleşme Skoru**: > 0.8
- **Güven Değeri**: > 0.9
- **Latency**: < 100 ms

---

## 5. KRITIK BAŞARIY FAKTÖRLERI

### 1. Veri Kalitesi
- VisDrone2019 + UAVDT veri setleri
- 70-90 derece nadir açı filtrelemesi
- Termal veri seti (HIT-UAV) entegrasyonu

### 2. Model Seçimi
- YOLOv8m (hız-doğruluk dengesi)
- ORB-SLAM2 (monoküler VO)
- XoFTR (termal-RGB eşleme)

### 3. Algoritma Tasarımı
- Semantik Ölçek Kurtarma (SSR)
- Kinematik EKF
- ByteTrack MOT

### 4. Sistem Optimizasyonu
- 4-Process mimarisi
- TensorRT FP16
- Zero-copy IPC

### 5. Hata Toleransı
- Dinamik FAST threshold
- SSIM-based görüntü donması tespiti
- Ölü piksel maskeleme
- Anomali filtresi

---

## 6. BILINEN SINIRLAMALAR & ÇÖZÜMLER

| Sınırlama | Çözüm | Etki |
|-----------|-------|------|
| Ground Plane varsayımı | Simülasyon kalibrasyonu | Köprü/eğimli arazi toleransı |
| CA modeli ani manevra | CTRA modeli değerlendirmesi | Yaw/pitch manevra esnekliği |
| Loop Closure eksikliği | SSR absolute reset | Drift azaltma |
| Tracking Loss | Relocalization + Kinematik | Güven kovaryansı eşiği |
| Düşük dokulu zemin | Dinamik FAST threshold | Feature extraction iyileştirme |
| Scooter geçiş anı | Temporal voting (3 kare) | Belirsizlik bölgesi çözümü |

---

## 7. REFERANS KAYNAKLAR (OTR'den)

1. **SAHI**: Akyon et al. (ICIP 2022) - +6.8% AP
2. **ByteTrack**: Zhang et al. (ECCV 2022) - False Negative azaltma
3. **Random Erasing**: Zhong et al. (AAAI 2020) - +15% mAP
4. **GridMask**: Chen et al. (arXiv 2020) - Dengeli silim
5. **Fiziksel Artırım**: Tremblay et al. (IJCV 2020) - Hava koşulu
6. **XoFTR**: Tuzcuoğlu et al. (CVPRW 2024) - Termal-RGB
7. **CubeSLAM**: Yang & Scherer (IEEE T-RO 2019) - Scale recovery
8. **LightGlue**: Lindenberger et al. (ICCV 2023) - Hızlı matching

---

## 8. SONUÇ

Bu strateji OTR'de belirtilen tüm gereksinimleri kapsar ve başarıyı maksimize etmek için:

1. **Mimari**: 4-Process asenkron sistem
2. **Algoritmalar**: SSR, ByteTrack, XoFTR, Kinematik EKF
3. **Veri**: Fiziksel artırım + VisDrone + UAVDT
4. **Optimizasyon**: TensorRT FP16 + Zero-copy IPC
5. **Tolerans**: Dinamik threshold + Anomali filtresi

Uygulanması halinde mAP > 45%, RMSE < 2m, Eşleşme Skoru > 0.8 hedeflerine ulaşılabilir.
