"""
TEKNOFEST 2026 - Sabit Değerler
"""

# Görev Sabitleri
TASK_1_NAME = "Nesne Tespiti"
TASK_2_NAME = "Pozisyon Kestirimi"
TASK_3_NAME = "Görüntü Eşleme"

# Sınıf ID Bilgileri
CLASS_ID = {
    "TASIT": 0,
    "INSAN": 1,
    "UAP": 2,    # Uçan Araba Park
    "UAI": 3     # Uçan Ambulans İniş
}

# İniş Durumu ID Bilgileri
LANDING_STATUS = {
    "UYGUN_DEGIL": 0,
    "UYGUN": 1,
    "INIS_ALANI_DEGIL": -1
}

# Hareket Durumu ID Bilgileri
MOTION_STATUS = {
    "HAREKETSIZ": 0,
    "HAREKETLI": 1,
    "TASIT_DEGIL": -1
}

# UAP/UAİ Fiziksel Özellikleri
UAP_UAI_DIAMETER = 4.5  # metre

# Yarışma Sabitleri
VIDEO_FPS = 7.5
TOTAL_FRAMES = 2250  # 5 dakika * 60 saniye * 7.5 FPS
VIDEO_DURATION_MINUTES = 5

# Çözünürlükler
RESOLUTIONS = {
    "FULL_HD": (1080, 1920),
    "4K": (3000, 4000),
    "THERMAL": (512, 640)
}

# IoU Eşik Değeri
IOU_THRESHOLD = 0.5

# GPS Sağlık Durumu
GPS_HEALTHY = 1
GPS_UNHEALTHY = 0

# GPS İsınma Fazı (kare sayısı)
GPS_WARMUP_FRAMES = 450  # İlk 1 dakika

# Puanlama Ağırlıkları
TASK_WEIGHTS = {
    "task1": 0.25,  # Nesne Tespiti
    "task2": 0.40,  # Pozisyon Kestirimi
    "task3": 0.25,  # Görüntü Eşleme
    "report": 0.05, # Final Tasarım Raporu
    "presentation": 0.05  # Yarışma Sunumu
}

# Başarı Kriteri
SUCCESS_THRESHOLD = 0.70  # %70

# YOLOv8 Modelleri
YOLOV8_MODELS = {
    "yolov8n": {"params": "3.2M", "map50": 0.527, "fps": 150, "vram": "1.2GB"},
    "yolov8s": {"params": "11.2M", "map50": 0.627, "fps": 120, "vram": "2.1GB"},
    "yolov8m": {"params": "25.9M", "map50": 0.675, "fps": 80, "vram": "3.8GB"},
    "yolov8l": {"params": "43.7M", "map50": 0.699, "fps": 50, "vram": "5.2GB"},
    "yolov8x": {"params": "68.2M", "map50": 0.708, "fps": 30, "vram": "7.1GB"}
}

# Renk Paleti (UI için)
COLORS = {
    "background": "#1e1e1e",
    "foreground": "#ffffff",
    "accent": "#007acc",
    "success": "#4caf50",
    "warning": "#ff9800",
    "error": "#f44336",
    "info": "#2196f3"
}

# Grafik Renkleri
GRAPH_COLORS = {
    "task1": "#ff6b6b",
    "task2": "#4ecdc4",
    "task3": "#45b7d1",
    "fps": "#96ceb4",
    "latency": "#ffeaa7",
    "gpu": "#dfe6e9",
    "ram": "#74b9ff"
}

# Log Seviyeleri
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# Dosya Yolları
PATHS = {
    "config": "./config/",
    "logs": "./logs/",
    "results": "./results/",
    "models": "./models/",
    "data": "./data/",
    "temp": "./temp/"
}

# UI Güncelleme Aralığı (ms)
UI_UPDATE_INTERVAL = 100

# Metrik Kayıt Aralığı (kare)
METRICS_LOG_INTERVAL = 10
