# TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması
# Ön Tasarım Raporu (OTR)
# TEKNOFEST 2026 - Nisan 2026

---

## 📋 İçindekiler

1. [Takım Şeması](#1-takım-şeması)
2. [Proje Mevcut Durum Değerlendirmesi](#2-proje-mevcut-durum-değerlendirmesi)
3. [Algoritmalar ve Sistem Mimarisi](#3-algoritmalar-ve-sistem-mimarisi)
4. [Özgünlük](#4-özgünlük)
5. [Proje Takvimi](#5-proje-takvimi)
6. [Sonuçlar ve İnceleme](#6-sonuçlar-ve-inceleme)

---

## 1. Takım Şeması

Takım iki üyeden oluşmaktadır. Şartname gereği kişisel bilgilere yer verilmemekte; yalnızca teknik sorumluluklar belirtilmektedir.

---

## 2. Proje Mevcut Durum Değerlendirmesi (10 Puan)

Bu bölümde teorik tasarım değil, yürütülen somut mühendislik çalışmaları özetlenmektedir.

### Tamamlanan Çalışmalar:

- **Donanım Profili:** RTX 4060 8GB VRAM üzerinde YOLOv8m TensorRT FP16 için ~8-12 ms/kare latency ölçülmüştür
- **Python GIL Çözümü:** multiprocessing.Process + shared_memory (zero-copy) mimarisi tasarlanmıştır
- **ORB-SLAM2 Scale Drift:** IMU gerektirmeyen Semantik Ölçek Kurtarma (SSR) yöntemi tasarlanmıştır
- **İniş Alanı Tespiti:** YOLO'nun 'tanımsız nesne körlüğü' zafiyeti tespit edilmiş; çözüm tasarlanmıştır
- **Termal Görüntü:** XoFTR → LightGlue → Kalman Tracker üç kademeli failover hiyerarşisi oluşturulmuştur
- **ByteTrack MOT Entegrasyonu:** Temporal hafıza mekanizması belirlenmiştir
- **Çevresel Bozulma Toleransı:** ORB-SLAM2 FAST threshold dinamik ayarı tasarlanmıştır
- **Veri Setleri:** VisDrone2019 ve UAVDT indirilmiştir

---

## 3. Algoritmalar ve Sistem Mimarisi (30 Puan)

### 3.1. Veri Setleri (10 Puan)

#### 3.1.1. Nesne Tespiti (Görev 1)

**Veri Kalitesi Kontrolü:**
- Sürücülü motosiklet/bisiklet etiketi 'İnsan' → 'Taşıt' olarak yeniden etiketlenmiştir
- Scooter: sürücüsüz → ID:0 (Taşıt), sürücülü → ID:1 (İnsan)

**Uzamsal Artırım:**
- Random Erasing (AAAI 2020)
- GridMask (arXiv 2020)
- Cutout ile parçalı/kesilmiş nesne öğrenimi

**Fiziksel Hava Koşulu Artırımı:**
- Albumentations RandomRain, RandomFog, RandomSunFlare
- Yönlü hareket bulanıklığı (Directional Motion Blur)
- Dead Pixel Masking

#### 3.1.2. Pozisyon Kestirimi (Görev 2)

- TEKNOFEST örnek video + referans koordinatlar
- EuRoC MAV Veri Seti

#### 3.1.3. Görüntü Eşleme (Görev 3)

- Oturum başında sağlanan referans nesne görselleri
- XoFTR pseudo-thermal artırmayla eğitildiğinden ek termal veri ihtiyacı minimumdur

### 3.2. Algoritmalar (15 Puan)

#### 3.2.1. Donanım Profili ve Zaman/Bellek Bütçesi

**Donanım:** Intel Core i5-12xxx, NVIDIA RTX 4060 8 GB VRAM, 16 GB RAM

#### 3.2.2. Görev 1: Nesne Tespiti ve İniş Alanı Analizi

**Ana Model:** YOLOv8m TensorRT FP16 formatında deploy edilmiştir.

**ByteTrack Entegrasyonu:**
- Düşük güven skorlu tespitleri (confidence < 0.25) çöpe atmak yerine önceki karelerin yüksek skorlu yörüngeleriyle eşleştirir
- False Negative sayısını azaltarak mAP'a doğrudan katkı sağlar

**İniş Alanı Uygunluk Hiyerarşisi:**
1. **Adım 1 - Ellipse Fitting:** UAP/UAİ dairesine OpenCV ile elips oturtulur
2. **Adım 2 - Ground Footprint Kontrolü:** Bounding box alt kenarının orta noktası elips içindeyse kontrolü
3. **Adım 3 - Anomali Tespiti:** UAP/UAİ iç ROI piksel yoğunluk analizi
4. **Adım 4 - Karar:** Üç koşulun hiçbiri tetiklenmiyorsa → landing_status = 1

**Scooter Sınıflandırması:**
- ORB-SLAM2 VO z-ekseni çıktısından türetilen anlık irtifa ve scale_px_per_m oranına göre normalize edilir

**Tren Wagon Ayrıştırması:**
- SAHI ile üretilen lokomotif/vagon tespitlerine Linear Alignment filtresi uygulanır
- Bounding box merkezleri arasındaki açı ray doğrultusuyla kıyaslanır (eşik ±15°)

**Çevresel Bozulma Toleransı:**
- **Ölü piksel maskeleme:** Kalibrasyon aşamasında statik maske oluşturulur
- **Yağmur/sis:** ORB-SLAM2 FAST threshold değeri Shannon entropisine göre dinamik olarak ayarlanır
- **Görüntü donması:** SSIM > 0.995 eşiği aşılırsa kare 'donmuş' sayılır

#### 3.2.3. Görev 2: Pozisyon Kestirimi – Kinematik EKF + Semantik Ölçek Kurtarma

**Kinematik EKF Mimarisi:**
- **Isınma Fazı (GPS Sağlıklı – İlk 450 Kare):** GPS referans koordinatlarından V ve A vektörleri hesaplanır
- **Tahmin Adımı (GPS Kesilince):** EKF'nin predict adımı kalibre edilmiş kinematik modele dayanır
- **Göreli Güncelleme:** ORB-SLAM2'nin bağıl yer değiştirme vektörü EKF'ye ölçüm olarak verilir
- **Mutlak Güncelleme (SSR):** Yalnızca ID:2 (UAP) veya ID:3 (UAİ) tespitinde SSR çalışır

**Semantik Ölçek Kurtarma (SSR) – Matematiksel Formülasyon:**
1. **Ray Casting:** Bounding box köşeleri normalize görüntü düzlemine yansıtılır
2. **Dünya Koordinatına Dönüşüm:** Işın dünya uzayına çevrilir
3. **Ground Plane Kesişimi:** Z=0 düzlemi varsayılır
4. **Ölçek Katsayısı:** Scale = 4.5 / D_vo
5. **Güncelleme Filtresi:** 1D Moving Average kuyruğuna eklenir (N=10)

#### 3.2.4. Görev 3: Görüntü Eşleme – XoFTR + LightGlue + Kalman Tracker

**Birincil:** XoFTR – Çapraz modalite transformatör
**İkincil:** LightGlue – SuperGlue'dan ~3× hızlı
**Üçüncül:** Kalman Tracker – Nesnenin son bilinen konumu kamera ego-motion kullanılarak kompanse edilir

### 3.3. Sistem Mimarisi ve Akış Şemaları (5 Puan)

**Şekil 1: 4-Process Asenkron Mimari**
- Python GIL'i çözmek için multiprocessing.Process + shared_memory kullanılır

**Şekil 2: SSR – Ray Casting Geometrisi**

**Şekil 3: Veri Akış Diyagramı – Tek Kare**

**Şekil 4: GPS/VO Geçiş Mekanizması**

**Şekil 5: İniş Alanı Uygunluk Hiyerarşisi**

---

## 4. Özgünlük (10 Puan)

Özgünlük, hazır modellerin parametrik ince ayarlarında değil; yarışmanın somut donanım ve şartname kısıtlarına yapılan mimari adaptasyonlarda yatmaktadır.

### Semantik Ölçek Kurtarma (SSR) – IMU-Free Scale Recovery
- IMU verisi olmayan monoküler kamerada scale drift problemini çözmek için UAP/UAİ işaretlerini 'Görsel Çapa' olarak kullanır

### Piksel Yoğunluk Tabanlı Anomali Tespiti
- YOLO'nun sınıf tanımadığı nesneleri tespit edemeyen körlüğünü aşmak için ROI doku homojenlik analizi

### Kinematik EKF ile IMU-Simüle Navigasyon
- GPS kesildiğinde IMU verisi olmaksızın drone'un uçuş dinamiklerini GPS ısınma fazından kalibre edilmiş V ve A vektörleriyle modelleme

### Ground Footprint + Perspektif Yanılgısı Ayrımı
- İniş uygunluğunu IoU tabanlı kutu kesişimi yerine nesnenin yere temas noktası üzerinden belirleme

### 4-Process Zero-Copy Mimari
- Python GIL kısıtını multiprocessing.shared_memory ile zero-copy IPC kullanarak çözme

---

## 5. Proje Takvimi (10 Puan)

---

## 6. Sonuçlar ve İnceleme (30 Puan)

### Literatür Destekli Performans Tahminleri:

| Metrik | Hedef | Literatür Kaynağı |
|--------|-------|-------------------|
| SAHI AP Artışı | +6.8% | Akyon et al. (ICIP 2022) |
| mAP50 (SAHI) | 45.6 | Zhang et al. (2023) |
| Sisli Ortam mAP Kaybı | ~15% | Tremblay et al. (IJCV 2020) |

### Bilinen Sınırlılıklar:

1. **Ground Plane Varsayımı (Z=0):** Köprü, eğimli arazi veya yüksek yapı üzerinde hatalı scale üretebilir
2. **Kinematik EKF CA Modeli:** Ani yaw veya pitch manevralarında drone'un gerçek ivmesini yansıtmaz
3. **ORB-SLAM2 Loop Closure:** Tek yönlü uçuşta düşük tetiklenme ihtimali
4. **Tracking Lost:** Düşük doku veya aşırı hızda relocalization başarısızlığı
5. **Düşük Dokulu Zemin:** Çimen, su yüzeyi, homojen asfalt
6. **Scooter Geçiş Anı:** %40-60 confidence aralığında belirsizlik
7. **SAHI Latency:** 80-100 ms, darboğaz durumunda dilim sayısı 4'e düşürülebilir
8. **GPS Kalitesi:** Kötü GPS sinyali V/A vektörlerini hatalı kalibre edebilir

---

## 7. Kaynakça (5 Puan)

---

## 8. Genel Rapor Düzeni (5 Puan)

---

*Bu rapor TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması Ön Tasarım Raporu şartnamesine uygun olarak hazırlanmıştır.*
