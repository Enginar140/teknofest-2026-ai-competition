"""
Metrikleme ve Performans Takibi Modülü
Real-time sistem performansı ve AI modeli metrikleri
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
from enum import Enum
import logging
import psutil
import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Metrik tipleri"""
    FPS = "fps"
    INFERENCE_TIME = "inference_time"
    GPU_MEMORY = "gpu_memory"
    CPU_USAGE = "cpu_usage"
    DETECTION_COUNT = "detection_count"
    TRACKING_COUNT = "tracking_count"
    POSITION_CONFIDENCE = "position_confidence"
    FRAME_LATENCY = "frame_latency"


@dataclass
class MetricValue:
    """Metrik değeri"""
    timestamp: float
    value: float
    unit: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PerformanceStats:
    """Performans istatistikleri"""
    timestamp: float = field(default_factory=time.time)
    
    # FPS ve timing
    fps: float = 0.0
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    std_inference_time_ms: float = 0.0
    
    # Sistem kaynakları
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    
    # AI metrikleri
    detection_count: int = 0
    tracking_count: int = 0
    avg_detection_confidence: float = 0.0
    avg_position_confidence: float = 0.0
    
    # Latency
    frame_latency_ms: float = 0.0
    network_latency_ms: float = 0.0
    
    # Durum
    is_healthy: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        return data


class MetricsCollector:
    """
    Metrikleme Toplayıcısı
    Sistem ve AI metriklerini toplar
    """
    
    def __init__(
        self,
        window_size: int = 100,
        update_interval: float = 1.0,
    ):
        """
        Metrikleme toplayıcısı başlat
        
        Args:
            window_size: Hareketli pencere boyutu
            update_interval: Güncelleme aralığı (saniye)
        """
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Metrik deque'leri
        self.metrics: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=window_size)
            for metric_type in MetricType
        }
        
        # Sistem metrikleri
        self.cpu_usage_history = deque(maxlen=window_size)
        self.memory_usage_history = deque(maxlen=window_size)
        self.gpu_memory_history = deque(maxlen=window_size)
        
        # Timing
        self.frame_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        
        # Sayaçlar
        self.frame_count = 0
        self.detection_count = 0
        self.tracking_count = 0
        
        # Thread
        self.monitor_thread = None
        self.running = False
        
        logger.info("MetricsCollector başlatıldı")
    
    def start(self):
        """Metrikleme toplayıcısını başlat"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
            )
            self.monitor_thread.start()
    
    def stop(self):
        """Metrikleme toplayıcısını durdur"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Sistem metrikleri izleme döngüsü"""
        while self.running:
            try:
                # CPU ve bellek
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_info = psutil.virtual_memory()
                
                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory_info.used / (1024 ** 2))  # MB
                
                # GPU metrikleri (opsiyonel)
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                        self.gpu_memory_history.append(gpu_memory)
                except:
                    pass
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Monitor loop hatası: {e}")
    
    def record_frame_time(self, elapsed_ms: float):
        """Frame zamanını kaydet"""
        self.frame_times.append(elapsed_ms)
        self.frame_count += 1
    
    def record_inference_time(self, elapsed_ms: float):
        """Inference zamanını kaydet"""
        self.inference_times.append(elapsed_ms)
    
    def record_detection(self, count: int, avg_confidence: float = 0.0):
        """Tespit sayısını kaydet"""
        self.detection_count += count
        if count > 0:
            self.metrics[MetricType.DETECTION_COUNT].append(
                MetricValue(time.time(), count)
            )
    
    def record_tracking(self, count: int):
        """Takip sayısını kaydet"""
        self.tracking_count += count
        self.metrics[MetricType.TRACKING_COUNT].append(
            MetricValue(time.time(), count)
        )
    
    def record_position_confidence(self, confidence: float):
        """Pozisyon güvenini kaydet"""
        self.metrics[MetricType.POSITION_CONFIDENCE].append(
            MetricValue(time.time(), confidence)
        )
    
    def record_frame_latency(self, latency_ms: float):
        """Frame latency'sini kaydet"""
        self.metrics[MetricType.FRAME_LATENCY].append(
            MetricValue(time.time(), latency_ms)
        )
    
    def get_stats(self) -> PerformanceStats:
        """Performans istatistiklerini al"""
        stats = PerformanceStats()
        
        # FPS hesapla
        if len(self.frame_times) > 0:
            avg_frame_time = np.mean(self.frame_times)
            stats.fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        # Inference time istatistikleri
        if len(self.inference_times) > 0:
            times = np.array(self.inference_times)
            stats.avg_inference_time_ms = float(np.mean(times))
            stats.min_inference_time_ms = float(np.min(times))
            stats.max_inference_time_ms = float(np.max(times))
            stats.std_inference_time_ms = float(np.std(times))
        
        # CPU ve bellek
        if len(self.cpu_usage_history) > 0:
            stats.cpu_usage_percent = float(np.mean(self.cpu_usage_history))
        
        if len(self.memory_usage_history) > 0:
            stats.memory_usage_mb = float(np.mean(self.memory_usage_history))
        
        if len(self.gpu_memory_history) > 0:
            stats.gpu_memory_mb = float(np.mean(self.gpu_memory_history))
        
        # AI metrikleri
        if self.frame_count > 0:
            stats.detection_count = self.detection_count // max(self.frame_count, 1)
            stats.tracking_count = self.tracking_count // max(self.frame_count, 1)
        
        # Sağlık kontrolü
        stats.is_healthy = self._check_health(stats)
        stats.warnings = self._get_warnings(stats)
        
        return stats
    
    def _check_health(self, stats: PerformanceStats) -> bool:
        """Sistem sağlığını kontrol et"""
        # FPS çok düşük
        if stats.fps < 10.0 and stats.fps > 0:
            return False
        
        # CPU çok yüksek
        if stats.cpu_usage_percent > 95.0:
            return False
        
        # Bellek çok yüksek
        if stats.memory_usage_mb > 8000:  # 8GB
            return False
        
        return True
    
    def _get_warnings(self, stats: PerformanceStats) -> List[str]:
        """Uyarıları al"""
        warnings = []
        
        if stats.fps < 20.0 and stats.fps > 0:
            warnings.append(f"Düşük FPS: {stats.fps:.1f}")
        
        if stats.cpu_usage_percent > 80.0:
            warnings.append(f"Yüksek CPU: {stats.cpu_usage_percent:.1f}%")
        
        if stats.memory_usage_mb > 6000:
            warnings.append(f"Yüksek bellek: {stats.memory_usage_mb:.0f}MB")
        
        if stats.gpu_memory_mb > 7000:
            warnings.append(f"Yüksek GPU belleği: {stats.gpu_memory_mb:.0f}MB")
        
        if stats.avg_inference_time_ms > 100:
            warnings.append(f"Yavaş inference: {stats.avg_inference_time_ms:.1f}ms")
        
        return warnings
    
    def reset(self):
        """Metrikleri sıfırla"""
        for metric_deque in self.metrics.values():
            metric_deque.clear()
        
        self.frame_times.clear()
        self.inference_times.clear()
        self.cpu_usage_history.clear()
        self.memory_usage_history.clear()
        self.gpu_memory_history.clear()
        
        self.frame_count = 0
        self.detection_count = 0
        self.tracking_count = 0


class PerformanceMonitor:
    """
    Performans İzleyici
    Real-time performans izleme ve raporlama
    """
    
    def __init__(
        self,
        on_stats_update: Optional[Callable[[PerformanceStats], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ):
        """
        Performans izleyici başlat
        
        Args:
            on_stats_update: İstatistik güncelleme callback'i
            on_warning: Uyarı callback'i
        """
        self.collector = MetricsCollector()
        self.on_stats_update = on_stats_update
        self.on_warning = on_warning
        
        # Geçmiş istatistikler
        self.stats_history: deque = deque(maxlen=1000)
        
        # Monitoring thread
        self.monitor_thread = None
        self.running = False
        
        logger.info("PerformanceMonitor başlatıldı")
    
    def start(self):
        """İzlemeyi başlat"""
        if not self.running:
            self.running = True
            self.collector.start()
            
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
            )
            self.monitor_thread.start()
    
    def stop(self):
        """İzlemeyi durdur"""
        self.running = False
        self.collector.stop()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """İzleme döngüsü"""
        last_warning_time = {}
        
        while self.running:
            try:
                stats = self.collector.get_stats()
                self.stats_history.append(stats)
                
                # Callback'i çağır
                if self.on_stats_update:
                    self.on_stats_update(stats)
                
                # Uyarıları işle (spam'ı önlemek için)
                for warning in stats.warnings:
                    last_time = last_warning_time.get(warning, 0)
                    if time.time() - last_time > 5.0:  # 5 saniyede bir
                        if self.on_warning:
                            self.on_warning(warning)
                        last_warning_time[warning] = time.time()
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Monitor loop hatası: {e}")
    
    def record_frame(self, elapsed_ms: float):
        """Frame zamanını kaydet"""
        self.collector.record_frame_time(elapsed_ms)
    
    def record_inference(self, elapsed_ms: float):
        """Inference zamanını kaydet"""
        self.collector.record_inference_time(elapsed_ms)
    
    def record_detection(self, count: int, avg_confidence: float = 0.0):
        """Tespit sayısını kaydet"""
        self.collector.record_detection(count, avg_confidence)
    
    def record_tracking(self, count: int):
        """Takip sayısını kaydet"""
        self.collector.record_tracking(count)
    
    def record_position_confidence(self, confidence: float):
        """Pozisyon güvenini kaydet"""
        self.collector.record_position_confidence(confidence)
    
    def get_current_stats(self) -> PerformanceStats:
        """Mevcut istatistikleri al"""
        return self.collector.get_stats()
    
    def get_stats_history(self, last_n: int = 100) -> List[PerformanceStats]:
        """İstatistik geçmişini al"""
        return list(self.stats_history)[-last_n:]
    
    def get_average_stats(self, window_size: int = 100) -> PerformanceStats:
        """Ortalama istatistikleri hesapla"""
        history = self.get_stats_history(window_size)
        
        if not history:
            return PerformanceStats()
        
        avg_stats = PerformanceStats()
        
        fps_values = [s.fps for s in history if s.fps > 0]
        if fps_values:
            avg_stats.fps = float(np.mean(fps_values))
        
        inference_times = [s.avg_inference_time_ms for s in history if s.avg_inference_time_ms > 0]
        if inference_times:
            avg_stats.avg_inference_time_ms = float(np.mean(inference_times))
        
        cpu_values = [s.cpu_usage_percent for s in history]
        if cpu_values:
            avg_stats.cpu_usage_percent = float(np.mean(cpu_values))
        
        memory_values = [s.memory_usage_mb for s in history]
        if memory_values:
            avg_stats.memory_usage_mb = float(np.mean(memory_values))
        
        gpu_values = [s.gpu_memory_mb for s in history if s.gpu_memory_mb > 0]
        if gpu_values:
            avg_stats.gpu_memory_mb = float(np.mean(gpu_values))
        
        detection_counts = [s.detection_count for s in history]
        if detection_counts:
            avg_stats.detection_count = int(np.mean(detection_counts))
        
        return avg_stats
    
    def get_report(self) -> Dict[str, Any]:
        """Performans raporu oluştur"""
        current = self.get_current_stats()
        average = self.get_average_stats()
        
        return {
            'timestamp': time.time(),
            'current': asdict(current),
            'average': asdict(average),
            'history_size': len(self.stats_history),
            'is_healthy': current.is_healthy,
            'warnings': current.warnings,
        }
    
    def reset(self):
        """Metrikleri sıfırla"""
        self.collector.reset()
        self.stats_history.clear()


class MetricsExporter:
    """
    Metrikleme Dışa Aktarıcısı
    Metrikleri farklı formatlarda dışa aktarır
    """
    
    @staticmethod
    def to_json(stats: PerformanceStats) -> str:
        """JSON formatına dönüştür"""
        import json
        return json.dumps(asdict(stats), indent=2)
    
    @staticmethod
    def to_csv_header() -> str:
        """CSV başlığı"""
        return ",".join([
            "timestamp",
            "fps",
            "avg_inference_time_ms",
            "cpu_usage_percent",
            "memory_usage_mb",
            "gpu_memory_mb",
            "detection_count",
            "tracking_count",
            "is_healthy",
        ])
    
    @staticmethod
    def to_csv_row(stats: PerformanceStats) -> str:
        """CSV satırı"""
        return ",".join([
            str(stats.timestamp),
            f"{stats.fps:.2f}",
            f"{stats.avg_inference_time_ms:.2f}",
            f"{stats.cpu_usage_percent:.2f}",
            f"{stats.memory_usage_mb:.2f}",
            f"{stats.gpu_memory_mb:.2f}",
            str(stats.detection_count),
            str(stats.tracking_count),
            str(stats.is_healthy),
        ])
    
    @staticmethod
    def export_to_file(
        stats_list: List[PerformanceStats],
        filepath: str,
        format: str = "csv",
    ):
        """
        Metrikleri dosyaya dışa aktar
        
        Args:
            stats_list: İstatistik listesi
            filepath: Dosya yolu
            format: 'csv' veya 'json'
        """
        if format == "csv":
            with open(filepath, 'w') as f:
                f.write(MetricsExporter.to_csv_header() + "\n")
                for stats in stats_list:
                    f.write(MetricsExporter.to_csv_row(stats) + "\n")
        
        elif format == "json":
            import json
            data = [asdict(s) for s in stats_list]
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Metrikleme dışa aktarıldı: {filepath}")


class PerformanceAnalyzer:
    """
    Performans Analiz Aracı
    Metrikleri analiz eder ve öneriler sunar
    """
    
    @staticmethod
    def analyze_bottleneck(stats: PerformanceStats) -> Dict[str, Any]:
        """
        Performans darboğazını analiz et
        
        Returns:
            Darboğaz analizi
        """
        bottleneck = {
            'primary': None,
            'secondary': None,
            'severity': 'low',  # low, medium, high
            'recommendations': [],
        }
        
        # CPU darboğazı
        if stats.cpu_usage_percent > 80:
            bottleneck['primary'] = 'CPU'
            bottleneck['recommendations'].append(
                "CPU kullanımı yüksek. Model boyutunu küçültmeyi veya preprocessing'i optimize etmeyi deneyin."
            )
        
        # GPU belleği darboğazı
        if stats.gpu_memory_mb > 6000:
            if bottleneck['primary'] is None:
                bottleneck['primary'] = 'GPU Memory'
            else:
                bottleneck['secondary'] = 'GPU Memory'
            bottleneck['recommendations'].append(
                "GPU belleği yüksek. Batch boyutunu azaltmayı veya daha küçük model kullanmayı deneyin."
            )
        
        # Inference zamanı darboğazı
        if stats.avg_inference_time_ms > 50:
            if bottleneck['primary'] is None:
                bottleneck['primary'] = 'Inference Time'
            else:
                bottleneck['secondary'] = 'Inference Time'
            bottleneck['recommendations'].append(
                "Inference zamanı yüksek. Daha hızlı model seçmeyi veya quantization kullanmayı deneyin."
            )
        
        # Şiddet belirle
        if stats.fps < 10:
            bottleneck['severity'] = 'high'
        elif stats.fps < 20:
            bottleneck['severity'] = 'medium'
        
        return bottleneck
    
    @staticmethod
    def get_optimization_suggestions(stats: PerformanceStats) -> List[str]:
        """
        Optimizasyon önerileri al
        
        Returns:
            Öneriler listesi
        """
        suggestions = []
        
        # FPS önerileri
        if stats.fps < 15:
            suggestions.append("🔴 FPS çok düşük - acil optimizasyon gerekli")
        elif stats.fps < 25:
            suggestions.append("🟡 FPS düşük - model boyutunu küçültmeyi deneyin")
        
        # CPU önerileri
        if stats.cpu_usage_percent > 90:
            suggestions.append("🔴 CPU kullanımı kritik - preprocessing'i optimize edin")
        elif stats.cpu_usage_percent > 75:
            suggestions.append("🟡 CPU kullanımı yüksek - multi-threading kullanmayı deneyin")
        
        # Bellek önerileri
        if stats.memory_usage_mb > 7000:
            suggestions.append("🔴 Sistem belleği kritik - uygulamayı yeniden başlatmayı deneyin")
        elif stats.memory_usage_mb > 5000:
            suggestions.append("🟡 Sistem belleği yüksek - gereksiz modelleri bellekten çıkarın")
        
        # GPU önerileri
        if stats.gpu_memory_mb > 7000:
            suggestions.append("🔴 GPU belleği kritik - batch boyutunu azaltın")
        elif stats.gpu_memory_mb > 5000:
            suggestions.append("🟡 GPU belleği yüksek - daha küçük model kullanmayı deneyin")
        
        # Inference önerileri
        if stats.avg_inference_time_ms > 100:
            suggestions.append("🔴 Inference zamanı çok yüksek - model quantization'ı deneyin")
        elif stats.avg_inference_time_ms > 50:
            suggestions.append("🟡 Inference zamanı yüksek - TensorRT veya ONNX kullanmayı deneyin")
        
        if not suggestions:
            suggestions.append("✅ Sistem performansı iyi")
        
        return suggestions
