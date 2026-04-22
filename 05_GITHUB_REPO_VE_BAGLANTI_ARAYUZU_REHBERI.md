# TEKNOFEST 2026 - GITHUB REPOSU ANALİZİ VE BAĞLANTI ARAYÜZÜ REHBERİ

---

## 📁 REPO YAPISI

```
havacilikta-yapay-zeka-yarismasi/
├── README.md                                    # Yarışma bilgileri
├── banner.jpg                                   # Yarışma görseli
├── nesne tespiti.gif                            # Görev 1 animasyonu
├── nesnetespiti.png                             # Akış diyagramı
├── pozisyon kestirimi.gif                       # Görev 2 animasyonu
│
├── Kamera_Kalibrasyon/                          # Kamera parametreleri
│   ├── Kamera_Kalibrasyon_Parametreleri_2024.txt
│   └── Kamera_Kalibrasyon_Parametreleri_2025.txt
│
└── TAKIM_BAGLANTI_ARAYUZU/                      # Sunucu bağlantı kodları
    ├── README.MD                                # Kurulum rehberi
    ├── main.py                                  # Ana çalıştırma dosyası
    ├── requirements.txt                         # Python paketleri
    ├── config/
    │   └── example.env                          # Konfigürasyon şablonu
    └── src/
        ├── connection_handler.py                # Sunucu bağlantı sınıfı
        ├── constants.py                         # Sınıf ve durum sabitleri
        ├── detected_object.py                   # Tespit edilen nesne sınıfı
        ├── detected_translation.py              # Pozisyon sonucu sınıfı
        ├── frame_predictions.py                 # Kare tahmin sınıfı
        ├── object_detection_model.py            # Model entegrasyon sınıfı ⭐
        └── translation.py                       # Yer değiştirme sınıfı
```

---

## 🎯 TEMEL BİLGİLER

### Veri Setleri
Google Drive üzerinden paylaşılıyor:
🔗 [TEKNOFEST Veri Setleri](https://drive.google.com/drive/folders/18_VqLBbyTubVSWAXG_CgmuJWGCx0mcBd)

### Görevler
1. **Nesne Tespiti:** mAP (IoU threshold = 0.5)
2. **Pozisyon Kestirimi:** RPG Trajectory Evaluation (RMSE)

---

## 🚪 BAĞLANTI ARAYÜZÜ KURULUMU

### 1. Ortam Kurulumu

```bash
# Conda ortamı oluştur
conda create -n teknofest_yarisma python=3.7
conda activate teknofest_yarisma

# Gerekli paketleri yükle
cd havacilikta-yapay-zeka-yarismasi/TAKIM_BAGLANTI_ARAYUZU
pip install -r requirements.txt
```

### 2. Konfigürasyon

`.env` dosyası oluştur:
```bash
cd config
copy example.env .env
```

`.env` dosyasını düzenle:
```env
TEAM_NAME=takim_adi
PASSWORD=sifre
EVALUATION_SERVER_URL="SUNUCU BILGISI YARISMADA PAYLASILACAK"
SESSION_NAME=oturum_ismi
```

### 3. Gerekli Paketler

```
python-decouple      # Konfigürasyon yönetimi
requests~=2.25.1     # HTTP istekleri
pillow~=8.2.0        # Görüntü işleme
tqdm                 # İlerleme çubuğu
```

---

## 💻 KOD YAPISI VE KULLANIMI

### Ana Dosya: main.py

```python
def run():
    # 1. Konfigürasyonu yükle
    team_name = config('TEAM_NAME')
    password = config('PASSWORD')
    evaluation_server_url = config("EVALUATION_SERVER_URL")
    
    # 2. Modeli başlat
    detection_model = ObjectDetectionModel(evaluation_server_url)
    
    # 3. Sunucuya bağlan
    server = ConnectionHandler(evaluation_server_url, username=team_name, password=password)
    
    # 4. Frameleri çek
    frames_json = server.get_frames(force_download=True)
    translations_json = server.get_translations(force_download=True)
    
    # 5. Her frame için:
    for frame, translation in tqdm(zip(frames_json, translations_json)):
        # a) Tahmin objesi oluştur
        predictions = FramePredictions(...)
        
        # b) Modeli çalıştır
        predictions = detection_model.process(predictions, ...)
        
        # c) Sonuçları gönder
        result = server.send_prediction(predictions)
```

### Model Entegrasyonu: object_detection_model.py

**ÖNEMLİ:** Yarışmacılar sadece `object_detection_model.py` dosyasında değişiklik yapmalıdır!

#### ObjectDetectionModel Sınıfı

```python
class ObjectDetectionModel:
    def __init__(self, evaluation_server_url):
        # Modelinizi burada başlatın
        # self.model = ...
        
    def process(self, prediction, evaluation_server_url, health_status, images_folder, images_files):
        # 1. Görüntüyü indir
        self.download_image(...)
        
        # 2. Pre-processing (isteğe bağlı)
        # img = preprocess(image)
        
        # 3. Modeli çalıştır
        frame_results = self.detect(prediction, health_status)
        
        return frame_results
        
    def detect(self, prediction, health_status):
        # BURAYA KENDİ MODELİNİZİ ENTEGRE EDİN
        
        # Nesne tespiti sonuçları:
        for detected_object in your_model_results:
            d_obj = DetectedObject(
                cls,              # Sınıf: "0", "1", "2", "3"
                landing_status,   # İniş: "-1", "0", "1"
                top_left_x,
                top_left_y,
                bottom_right_x,
                bottom_right_y
            )
            prediction.add_detected_object(d_obj)
        
        # Pozisyon kestirimi:
        if health_status == '0':  # GPS çalışmıyor
            # Kendi algoritmanızı çalıştırın
            pred_translation_x = your_position_x
            pred_translation_y = your_position_y
            pred_translation_z = your_position_z
        else:  # GPS çalışıyor
            pred_translation_x = prediction.gt_translation_x
            pred_translation_y = prediction.gt_translation_y
            pred_translation_z = prediction.gt_translation_z
        
        trans_obj = DetectedTranslation(pred_translation_x, pred_translation_y, pred_translation_z)
        prediction.add_translation_object(trans_obj)
        
        return prediction
```

---

## 📊 SABİTLER (constants.py)

### Sınıflar:
```python
classes = {
    "TASI": "0",      # Taşıt
    "INSAN": "1",     # İnsan
    "UAP": "2",       # Uçan Araba Park
    "UAI": "3"        # Uçan Ambulans İniş
}
```

### İniş Durumları:
```python
landing_statuses = {
    "Inilemez": "0",    # Uygun Değil
    "Inilebilir": "1",  # Uygun
    "InilmeAlaniDegil": "-1"  # İniş Alanı Değil
}
```

### Hareket Durumları:
```python
motion_statuses = {
    "Hareketsiz": "0",
    "Hareketli": "1",
    "TasitDegil": "-1"
}
```

---

## 📷 KAMERA PARAMETRELERİ

### RGB Camera (2025):
```
Focal Length: [2792.2, 2795.2]
Principal Point: [1988.0, 1562.2]
Image Size: [3000, 4000]
Radial Distortion: [0.0798, -0.1867]
```

### Termal Camera (2025):
```
Focal Length: [731.8, 732.0]
Principal Point: [319.2, 251.2]
Image Size: [512, 640]
Radial Distortion: [-0.3507, 0.1137]
```

**Kullanım Alanı:** Pozisyon kestirimi için kamera kalibrasyonu gereklidir.

---

## 🔧 MODEL ENTEGRASYONU ÖRNEĞİ

### YOLOv8 Entegrasyonu:

```python
# object_detection_model.py içinde

from ultralytics import YOLO
import cv2
import numpy as np

class ObjectDetectionModel:
    def __init__(self, evaluation_server_url):
        # YOLOv8 modelini yükle
        self.model = YOLO('yolov8n.pt')
        self.evaluation_server = evaluation_server_url
        
    def detect(self, prediction, health_status):
        # Görüntüyü yükle
        img_path = "./downloaded_images/" + prediction.image_url.split("/")[-1]
        img = cv2.imread(img_path)
        
        # YOLO ile tespit yap
        results = self.model(img)
        
        # Sonuçları işle
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Koordinatları al
                x1, y1, x2, y2 = box.xyxy[0]
                
                # Sınıf ve confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Sınıf ID'sini dönüştür
                cls = str(cls_id)
                
                # İniş durumu (taşıtlar için -1)
                landing_status = "-1"
                
                # Hareket durumu (taşıtlar için)
                motion_status = "1" if conf > 0.5 else "0"
                
                # Nesneyi ekle
                d_obj = DetectedObject(
                    cls,
                    landing_status,
                    int(x1), int(y1),
                    int(x2), int(y2)
                )
                prediction.add_detected_object(d_obj)
        
        # Pozisyon kestirimi
        if health_status == '0':
            # Kendi algoritmanız
            pred_x, pred_y, pred_z = self.estimate_position(img)
        else:
            pred_x = prediction.gt_translation_x
            pred_y = prediction.gt_translation_y
            pred_z = prediction.gt_translation_z
        
        trans_obj = DetectedTranslation(str(pred_x), str(pred_y), str(pred_z))
        prediction.add_translation_object(trans_obj)
        
        return prediction
```

---

## 📝 LOG SİSTEMİ

Log dosyaları `_logs` klasöründe saklanır:
- Format: `{takim_adi}_{yil}_{ay}_{gun}__{saat}_{dakika}_{saniye}_{mikrosaniye}.log`
- Yarışma sırasında itirazlarda log dosyası değerlendirmeye alınır

Log örneği:
```
2026-07-09 10:30:15 - INFO - Started...
2026-07-09 10:30:16 - INFO - Created Object Detection Model
2026-07-09 10:30:17 - INFO - Download Finished in 0.5 seconds
```

---

## ⚠️ ÖNEMLİ NOTLAR

1. **Sadece `object_detection_model.py` dosyasında değişiklik yapın**
2. **Test için `.env` dosyasını doğru şekilde yapılandırın**
3. **Log dosyalarını takip edin**
4. **Health Status = '0' olduğunda pozisyon kestirimi yapın**
5. **Her frame için bir sonuç gönderin**
6. **İnternet bağlantısı yoktur (yerel ağ)**

---

## 🎓 ÖĞRENME SIRASI

1. **Python Temelleri** (1 hafta)
2. **OpenCV** (1 hafta)
3. **YOLOv8** (2 hafta)
4. **Bağlantı arayüzünü anlama** (3 gün)
5. **Model entegrasyonu** (1 hafta)
6. **Test ve optimizasyon** (1 hafta)

---

**Son Güncelleme:** 20 Nisan 2026

*Bu rehber TEKNOFEST resmi GitHub reposuna dayanarak hazırlanmıştır.*
