# YARIŞMA KOŞULLARINA GÖRE EN İYİ MODEL SEÇİMİ
## Doğruluk Odaklı Analiz

---

## 🎯 YARIŞMA GERÇEĞİ: FPS'NİN ÖNEMİ YOK

### Kritik Bilgi

Yarışma şartnamesine göre:

```
┌─────────────────────────────────────────────────────────────┐
│              YARIŞMA İŞLEME ŞARTLARI                       │
├─────────────────────────────────────────────────────────────┤
│ 1. Video: 5 dakika, 7.5 FPS = 2250 kare                  │
│ 2. İşleme zamanı: 60 dakika                               │
│ 3. Gereken hız: 2250 kare / 3600 saniye = 0.6 FPS         │
│ 4. Dolayısıyla: Saniyede 1 kare yeterli (0.6 FPS)         │
│ 5. Algoritma hızı puanlamada DİKKATE ALINMIYOR            │
└─────────────────────────────────────────────────────────────┘
```

### Sonuç

**FPS'nin hiçbir önemi yok!** Sisteminiz saniyede 0.6 kare işleyebildiği sürece yeterli.

**ÖNEMLİ OLAN TEK ŞEY:** Doğruluk (mAP)

---

## 📊 MODEL KARŞILAŞTIRMASI: DOĞRULUK ODAKLI

### YOLOv8 Serisi - COCO Dataset (Benchmark)

| Model | Parametre | mAP@0.5 | mAP@0.5:0.95 | FPS (RTX 4060) | Karar |
|-------|-----------|---------|---------------|----------------|--------|
| YOLOv8n | 3.2M | 0.527 | 0.348 | 150 | ❌ En düşük doğruluk |
| YOLOv8s | 11.2M | 0.627 | 0.443 | 120 | ❌ Orta doğruluk |
| **YOLOv8m** | 25.9M | **0.675** | **0.518** | 80 | ✅ Dengeli |
| **YOLOv8l** | 43.7M | **0.699** | **0.553** | 50 | ⭐ **EN İYİ** |
| YOLOv8x | 68.2M | **0.708** | **0.571** | 30 | ⚠️ Daha yavaş |

### Analiz

**YOLOv8l vs YOLOv8x:**
- YOLOv8l: mAP 0.699
- YOLOv8x: mAP 0.708
- Fark: Sadece **0.009** (0.9%)

**Sonuç:** YOLOv8x, YOLOv8l'den sadece %0.9 daha doğru ama çok daha yavaş.

**KARAR:** **YOLOv8l** = En iyi performans/fayda oranı

---

## 🎯 SİSTEMİNİZ İÇİN EN İYİ MODEL: YOLOv8l

### Neden YOLOv8l?

1. ✅ **En yüksek doğruluk** (büyük modeller arasında)
2. ✅ **Sisteminiz rahatça kaldırır** (RTX 4060)
3. ✅ **Uygun eğitim süresi** (~8 saat/50 epoch)
4. ✅ **Yarışma için yeterli hız** (50 FPS >> 0.6 FPS)

### Performans Analizi

```
Yarışma Gereksinimi: 0.6 FPS
Sisteminizin Kapasitesi (YOLOv8l): 50 FPS
Fark: 83x daha hızlı!
```

**Sonuç:** YOLOv8l kullanarak hem hız hem de doğruluk kazanırsınız.

---

## 🔧 YOLOv8l OPTİMUM KONFİGÜRASYON

### Eğitim Parametreleri

```python
from ultralytics import YOLO

# En iyi model
model = YOLO('yolov8l.pt')  # Pre-trained COCO

# 3 Aşamalı Eğitim
stage1 = {
    'data': 'coco.yaml',
    'epochs': 50,
    'batch': 8,              # RTX 4060 için
    'imgsz': 640,
    'lr0': 0.01,
    'device': 0,
    'workers': 8,
    'amp': True,
    'cache': 'ram',
}

stage2 = {
    'data': 'visdrone.yaml',
    'epochs': 50,
    'batch': 8,
    'imgsz': 640,
    'lr0': 0.001,
    'device': 0,
    'workers': 8,
    'amp': True,
    'augment': True,
}

stage3 = {
    'data': 'teknofest.yaml',
    'epochs': 100,
    'batch': 4,              # Daha küçük batch
    'imgsz': 640,
    'lr0': 0.0001,
    'device': 0,
    'workers': 8,
    'amp': True,
    'augment': True,
    'mosaic': 1.0,
    'mixup': 0.15,
}
```

### Batch Size Seçimi

| Batch Size | VRAM Kullanımı | Eğitim Süresi | Öneri |
|------------|-----------------|---------------|-------|
| 16 | ~6GB | Daha hızlı | ❌ VRAM yetmeyebilir |
| **8** | ~4GB | ** Dengeli** | ⭐ **EN İYİ** |
| 4 | ~2.5GB | Daha yavaş | ✅ Güvenli |

**KARAR:** Batch size = 8 (en stabil)

---

## 📈 BEKLENEN PERFORMANS

### Eğitim Süresi

| Aşama | Model | Epoch | Batch | Süre |
|-------|-------|-------|-------|------|
| Stage 1 | YOLOv8l (COCO) | 50 | 8 | ~6 saat |
| Stage 2 | YOLOv8l (VisDrone) | 50 | 8 | ~7 saat |
| Stage 3 | YOLOv8l (TEKNOFEST) | 100 | 4 | ~12 saat |
| **Toplam** | - | **200** | - | **~25 saat** |

### Çıkarım Hızı

```
YOLOv8l Performansı (RTX 4060):
- Batch=1: 50 FPS
- Batch=8: 40 FPS
- Batch=16: 25 FPS

Yarışma Gereksinimi: 0.6 FPS
Sisteminiz: 50 FPS
Fark: 83x daha hızlı ✅
```

### Beklenen Doğruluk

| Metrik | Değer | Yorum |
|--------|-------|-------|
| mAP@0.5 | **0.70-0.75** | Mükemmel |
| Taşıt | %80-85 | Çok iyi |
| İnsan | %75-80 | Çok iyi |
| UAP/UAİ | %70-75 | İyi |

---

## 🎯 ALTERNATİF: YOLOv8x (Maksimum Doğruluk)

### YOLOv8l vs YOLOv8x Karşılaştırması

```
┌────────────────────────────────────────────────────────────┐
│              DOĞRULUK FARKI ANALİZİ                       │
├────────────────────────────────────────────────────────────┤
│ YOLOv8l mAP: 0.699                                       │
│ YOLOv8x mAP: 0.708                                       │
│ Fark: 0.009 (0.9%)                                       │
│                                                          │
│ Eğitim Farkı:                                            │
│ YOLOv8l: ~25 saat                                        │
│ YOLOv8x: ~40 saat                                        │
│                                                          │
│ Fark: +15 saat (çok uzun)                                │
└────────────────────────────────────────────────────────────┘
```

### Karar

**YOLOv8l SEÇİN** çünkü:
1. %0.9 doğruluk farkı 15 saate değmez
2. YOLOv8l zaten çok yüksek doğrulukta
3. Daha kısa sürede eğitilir
4. Daha stabil çalışır

---

## 🚀 OPTİMİZASYON STRATEJİLERİ

### 1. Test Time Augmentation (TTA)

```python
# Test zamanı augmentation ile +2-3% mAP
results = model.predict(
    image,
    augment=True,
    agnostic_nms=True,
)

# TTA kullan
results = model.predict(
    image,
    augment=True,
    half=True,
)
```

### 2. Multi-Scale Inference

```python
# Farklı ölçeklerde tahmin
scales = [0.8, 1.0, 1.2]
all_results = []

for scale in scales:
    img = cv2.resize(image, None, fx=scale, fy=scale)
    result = model(img)
    all_results.append(result)

# Ensemble
final = weighted_average(all_results)
```

### 3. Model Ensemble

```python
# 3 farklı modeli birleştir
models = [
    YOLO('yolov8l_teknofest.pt'),
    YOLO('yolov8m_teknofest.pt'),
    YOLO('yolov8s_teknofest.pt'),
]

# Weighted averaging
results = []
for model in models:
    result = model.predict(image)
    results.append(result)

# En iyi tahmini seç (Non-Maximum Suppression)
final = nms(results)
```

---

## 📋 EN İYİ KONFİGÜRASYON ÖZETİ

```python
BEST_CONFIG = {
    # Model (En yüksek doğruluk)
    'model': 'YOLOv8l',
    
    # Eğitim
    'batch_size': 8,        # RTX 4060 için optimal
    'image_size': 640,
    'total_epochs': 200,
    
    # Optimizasyon
    'optimizer': 'AdamW',
    'learning_rate': 0.001,
    'scheduler': 'CosineAnnealingLR',
    'weight_decay': 0.0005,
    
    # Hızlandırma
    'amp': True,            # Mixed precision
    'cache': 'ram',
    
    # Augmentation
    'augment': True,
    'mosaic': 1.0,
    'mixup': 0.15,
    
    # Performans
    'expected_map': 0.70,   # Yüksek doğruluk
    'expected_fps': 50,     # Yeterli hız
    'training_time': '~25 saat',
    
    # Yarışma
    'competition_fps': 0.6, # Gereken
    'your_fps': 50,         # Sahip olduğunuz
    'margin': '83x',        # Fazlalık
}
```

---

## 🎯 SON KARAR: YOLOv8l

### Neden YOLOv8l?

1. **En İyi Doğruluk/Fayda Oranı**
   - YOLOv8x'e göre sadece %0.9 daha düşük
   - Eğitim süresi %40 daha az

2. **Sisteminiz İçin Uygun**
   - RTX 4060 ile rahatça çalışır
   - Batch size = 8 stabil
   - VRAM kullanımı optimal

3. **Yarışma İçin Yeterli**
   - 50 FPS >> 0.6 FPS gereksinimi
   - Hız bir sorun değil

4. **Beklenen Performans**
   - mAP: 0.70-0.75
   - Tüm sınıflarda yüksek doğruluk

---

**Son Güncelleme:** 20 Nisan 2026

*Bu analiz TEKNOFEST 2026 yarışma koşullarına göre yapılmıştır.*
