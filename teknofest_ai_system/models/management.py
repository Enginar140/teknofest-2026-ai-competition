"""
Model Yönetimi ve Dinamik Seçim Sistemi
Farklı modeller arası geçiş ve performans takibi
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model tipleri"""
    DETECTION = "detection"
    POSITION = "position"
    MATCHING = "matching"
    TRACKING = "tracking"


class ModelSize(Enum):
    """Model boyutları"""
    NANO = "n"      # YOLOv8n - en hızlı, en az doğru
    SMALL = "s"     # YOLOv8s - dengeli
    MEDIUM = "m"    # YOLOv8m - daha doğru
    LARGE = "l"     # YOLOv8l - çok doğru
    XLARGE = "x"    # YOLOv8x - en doğru, en yavaş


@dataclass
class ModelMetadata:
    """Model meta verileri"""
    name: str
    type: ModelType
    size: ModelSize
    path: str
    framework: str = "ultralytics"
    input_size: Tuple[int, int] = (640, 640)
    num_classes: int = 14
    parameters: int = 0
    flops: int = 0
    
    # Performans metrikleri
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    inference_time_ms: float = 0.0
    fps: float = 0.0
    
    # Tarihler
    created_at: float = field(default_factory=time.time)
    trained_at: float = 0.0
    last_used: float = 0.0
    
    # Etiketler ve notlar
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def calculate_hash(self) -> str:
        """Model dosyasının hash'ini hesapla"""
        if not os.path.exists(self.path):
            return ""
        
        hash_md5 = hashlib.md5()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def to_dict(self) -> Dict:
        """Dict'e çevir"""
        data = asdict(self)
        data['type'] = self.type.value
        data['size'] = self.size.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Dict'ten oluştur"""
        data = data.copy()
        data['type'] = ModelType(data['type'])
        data['size'] = ModelSize(data['size'])
        return cls(**data)


@dataclass
class ModelBenchmark:
    """Model benchmark sonuçları"""
    model_name: str
    timestamp: float = field(default_factory=time.time)
    
    # Test verisi
    dataset_name: str = ""
    num_frames: int = 0
    resolution: Tuple[int, int] = (640, 640)
    
    # Performans metrikleri
    avg_inference_time_ms: float = 0.0
    min_inference_time_ms: float = 0.0
    max_inference_time_ms: float = 0.0
    std_inference_time_ms: float = 0.0
    
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    
    # Doğruluk metrikleri
    mAP: float = 0.0
    mAP50: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    # Kaynak kullanımı
    gpu_memory_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def calculate_score(self) -> float:
        """
        Genel performans skoru hesapla
        Daha yüksek = daha iyi
        
        Ağırlıklar:
        - FPS: %40
        - mAP: %40
        - GPU kullanımı: %20 (daha az = daha iyi)
        """
        fps_score = min(self.avg_fps / 60.0, 1.0) * 0.4
        map_score = self.mAP * 0.4
        memory_score = max(1.0 - self.gpu_memory_mb / 8000.0, 0.0) * 0.2
        
        return fps_score + map_score + memory_score
    
    def to_dict(self) -> Dict:
        """Dict'e çevir"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelBenchmark':
        """Dict'ten oluştur"""
        return cls(**data)


class ModelRegistry:
    """
    Model Kayıt Sistemi
    Mevcut modelleri takip eder ve yönetir
    """
    
    def __init__(self, registry_path: str = "models/registry.json"):
        """
        Model registry başlat
        
        Args:
            registry_path: Registry dosya yolu
        """
        self.registry_path = Path(registry_path)
        self.models: Dict[str, ModelMetadata] = {}
        self.benchmarks: Dict[str, List[ModelBenchmark]] = {}
        
        # Registry'yi yükle
        self.load()
        
        logger.info(f"ModelRegistry başlatıldı: {len(self.models)} model")
    
    def register_model(self, metadata: ModelMetadata) -> bool:
        """
        Model kaydet
        
        Args:
            metadata: Model meta verileri
            
        Returns:
            Başarılı mı?
        """
        if not os.path.exists(metadata.path):
            logger.error(f"Model dosyası bulunamadı: {metadata.path}")
            return False
        
        self.models[metadata.name] = metadata
        self.save()
        
        logger.info(f"Model kaydedildi: {metadata.name}")
        return True
    
    def unregister_model(self, model_name: str) -> bool:
        """
        Model sil
        
        Args:
            model_name: Model adı
            
        Returns:
            Başarılı mı?
        """
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.benchmarks:
                del self.benchmarks[model_name]
            self.save()
            logger.info(f"Model silindi: {model_name}")
            return True
        return False
    
    def get_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Model meta verilerini al"""
        return self.models.get(model_name)
    
    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        model_size: Optional[ModelSize] = None,
    ) -> List[ModelMetadata]:
        """
        Modelleri listele
        
        Args:
            model_type: Filtre: model tipi
            model_size: Filtre: model boyutu
            
        Returns:
            Model meta verileri listesi
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.type == model_type]
        
        if model_size:
            models = [m for m in models if m.size == model_size]
        
        return models
    
    def add_benchmark(self, benchmark: ModelBenchmark):
        """Benchmark sonucu ekle"""
        if benchmark.model_name not in self.benchmarks:
            self.benchmarks[benchmark.model_name] = []
        
        self.benchmarks[benchmark.model_name].append(benchmark)
        self.save()
    
    def get_benchmarks(self, model_name: str) -> List[ModelBenchmark]:
        """Model benchmark sonuçlarını al"""
        return self.benchmarks.get(model_name, [])
    
    def get_latest_benchmark(self, model_name: str) -> Optional[ModelBenchmark]:
        """Son benchmark sonucunu al"""
        benchmarks = self.get_benchmarks(model_name)
        if benchmarks:
            return max(benchmarks, key=lambda b: b.timestamp)
        return None
    
    def get_average_performance(self, model_name: str) -> Optional[Dict[str, float]]:
        """Modelin ortalama performansını hesapla"""
        benchmarks = self.get_benchmarks(model_name)
        if not benchmarks:
            return None
        
        return {
            'avg_fps': sum(b.avg_fps for b in benchmarks) / len(benchmarks),
            'avg_inference_time_ms': sum(b.avg_inference_time_ms for b in benchmarks) / len(benchmarks),
            'avg_map': sum(b.mAP for b in benchmarks) / len(benchmarks),
            'avg_gpu_memory_mb': sum(b.gpu_memory_mb for b in benchmarks) / len(benchmarks),
        }
    
    def save(self):
        """Registry'yi kaydet"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'models': {name: m.to_dict() for name, m in self.models.items()},
            'benchmarks': {
                name: [b.to_dict() for b in benchmarks]
                for name, benchmarks in self.benchmarks.items()
            },
        }
        
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self):
        """Registry'yi yükle"""
        if not self.registry_path.exists():
            return
        
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.models = {
                name: ModelMetadata.from_dict(m_data)
                for name, m_data in data.get('models', {}).items()
            }
            
            self.benchmarks = {
                name: [ModelBenchmark.from_dict(b_data) for b_data in benchmarks]
                for name, benchmarks in data.get('benchmarks', {}).items()
            }
            
            logger.info(f"Registry yüklendi: {len(self.models)} model")
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Registry yükleme hatası: {e}")


class ModelSelector:
    """
    Dinamik Model Seçici
    Koşullara göre en iyi modeli seçer
    """
    
    def __init__(
        self,
        registry: ModelRegistry,
        target_fps: float = 30.0,
        max_gpu_memory_mb: float = 4000.0,
        min_map: float = 0.5,
        priority: str = "balanced",  # "speed", "accuracy", "balanced"
    ):
        """
        Model seçici başlat
        
        Args:
            registry: Model registry
            target_fps: Hedef FPS
            max_gpu_memory_mb: Maksimum GPU belleği
            min_map: Minimum mAP skoru
            priority: Öncelik stratejisi
        """
        self.registry = registry
        self.target_fps = target_fps
        self.max_gpu_memory_mb = max_gpu_memory_mb
        self.min_map = min_map
        self.priority = priority
        
        self.current_model: Optional[str] = None
        self.selection_history: List[Tuple[float, str]] = []
        
        logger.info(f"ModelSelector başlatıldı: priority={priority}")
    
    def select_model(
        self,
        model_type: ModelType,
        current_fps: Optional[float] = None,
        current_gpu_memory: Optional[float] = None,
    ) -> Optional[ModelMetadata]:
        """
        En iyi modeli seç
        
        Args:
            model_type: Model tipi
            current_fps: Mevcut FPS (adaptasyon için)
            current_gpu_memory: Mevcut GPU kullanımı (adaptasyon için)
            
        Returns:
            Seçilen model meta verileri
        """
        candidates = self.registry.list_models(model_type=model_type)
        
        if not candidates:
            logger.warning(f"Uygun model bulunamadı: {model_type}")
            return None
        
        # Benchmark sonuçları ile filtrele
        valid_models = []
        for model in candidates:
            perf = self.registry.get_average_performance(model.name)
            
            if perf is None:
                # Benchmark yoksa varsayılan değerleri kullan
                if model.inference_time_ms > 0:
                    fps = 1000.0 / model.inference_time_ms
                else:
                    # Boyuta göre tahmin
                    fps = self._estimate_fps(model.size)
                
                perf = {
                    'avg_fps': fps,
                    'avg_map': model.accuracy,
                    'avg_gpu_memory_mb': 1000.0,  # Tahmini
                }
            
            # Minimum gereksinimleri kontrol et
            if perf['avg_map'] < self.min_map:
                continue
            
            if perf['avg_gpu_memory_mb'] > self.max_gpu_memory_mb:
                continue
            
            valid_models.append((model, perf))
        
        if not valid_models:
            logger.warning("Gereksinimleri karşılayan model yok, en hızlı model seçiliyor")
            # Yedek: en küçük modeli seç
            candidates_by_size = sorted(candidates, key=lambda m: self._size_order(m.size))
            return candidates_by_size[0] if candidates_by_size else None
        
        # Skorla ve sırala
        scored_models = []
        for model, perf in valid_models:
            score = self._calculate_score(perf, current_fps, current_gpu_memory)
            scored_models.append((score, model, perf))
        
        # Skora göre sırala (düşük = daha iyi for sorting, high score = better)
        scored_models.sort(key=lambda x: x[0], reverse=True)
        
        # En iyi modeli seç
        best_model = scored_models[0][1]
        
        # Model değişikliği kontrolü
        if self.current_model != best_model.name:
            logger.info(f"Model değişikliği: {self.current_model} -> {best_model.name}")
            self.current_model = best_model.name
            self.selection_history.append((time.time(), best_model.name))
        
        return best_model
    
    def _calculate_score(
        self,
        perf: Dict[str, float],
        current_fps: Optional[float],
        current_gpu_memory: Optional[float],
    ) -> float:
        """
        Model performans skoru hesapla
        
        Daha yüksek = daha iyi
        """
        fps = perf['avg_fps']
        map_score = perf['avg_map']
        gpu_memory = perf['avg_gpu_memory_mb']
        
        # Normalize [0, 1]
        fps_norm = min(fps / self.target_fps, 1.0)
        map_norm = map_score  # zaten [0, 1]
        memory_norm = max(1.0 - gpu_memory / self.max_gpu_memory_mb, 0.0)
        
        # Önceliğe göre ağırlıklandır
        if self.priority == "speed":
            weights = {'fps': 0.7, 'map': 0.2, 'memory': 0.1}
        elif self.priority == "accuracy":
            weights = {'fps': 0.2, 'map': 0.7, 'memory': 0.1}
        else:  # balanced
            weights = {'fps': 0.4, 'map': 0.4, 'memory': 0.2}
        
        # Adaptif ağırlıklandırme (mevcut duruma göre)
        if current_fps is not None and current_fps < self.target_fps * 0.8:
            # FPS düşükse hıza öncelik ver
            weights['fps'] = min(weights['fps'] + 0.2, 0.8)
            weights['map'] = max(weights['map'] - 0.1, 0.1)
        
        score = (
            fps_norm * weights['fps'] +
            map_norm * weights['map'] +
            memory_norm * weights['memory']
        )
        
        return score
    
    def _estimate_fps(self, model_size: ModelSize) -> float:
        """Model boyutuna göre FPS tahmin et"""
        fps_map = {
            ModelSize.NANO: 120.0,
            ModelSize.SMALL: 80.0,
            ModelSize.MEDIUM: 50.0,
            ModelSize.LARGE: 30.0,
            ModelSize.XLARGE: 20.0,
        }
        return fps_map.get(model_size, 30.0)
    
    def _size_order(self, model_size: ModelSize) -> int:
        """Model boyutu sıralaması (küçük -> büyük)"""
        order = {
            ModelSize.NANO: 0,
            ModelSize.SMALL: 1,
            ModelSize.MEDIUM: 2,
            ModelSize.LARGE: 3,
            ModelSize.XLARGE: 4,
        }
        return order.get(model_size, 2)
    
    def should_switch_model(
        self,
        current_fps: float,
        current_gpu_memory: float,
    ) -> bool:
        """
        Model değiştirilmeli mi?
        
        Args:
            current_fps: Mevcut FPS
            current_gpu_memory: Mevcut GPU kullanımı
            
        Returns:
            Değiştirilmeli mi?
        """
        # FPS çok düşükse daha hızlı modele geç
        if current_fps < self.target_fps * 0.7:
            return True
        
        # GPU belleği dolmak üzereyse daha küçük modele geç
        if current_gpu_memory > self.max_gpu_memory_mb * 0.9:
            return True
        
        # FPS çok yüksekse ve GPU boşsa daha doğru modele geç
        if current_fps > self.target_fps * 1.5 and current_gpu_memory < self.max_gpu_memory_mb * 0.5:
            return True
        
        return False
    
    def get_selection_history(self) -> List[Tuple[float, str]]:
        """Seçim geçmişini al"""
        return self.selection_history.copy()


class DynamicModelManager:
    """
    Dinamik Model Yöneticisi
    Runtime'da model değiştirme ve optimizasyon
    """
    
    def __init__(
        self,
        registry_path: str = "models/registry.json",
        target_fps: float = 30.0,
        max_gpu_memory_mb: float = 4000.0,
    ):
        """
        Dinamik model yöneticisi başlat
        
        Args:
            registry_path: Registry dosya yolu
            target_fps: Hedef FPS
            max_gpu_memory_mb: Maksimum GPU belleği
        """
        self.registry = ModelRegistry(registry_path)
        self.selector = ModelSelector(
            self.registry,
            target_fps=target_fps,
            max_gpu_memory_mb=max_gpu_memory_mb,
        )
        
        self.loaded_models: Dict[str, Any] = {}
        self.active_models: Dict[ModelType, str] = {}
        
        logger.info("DynamicModelManager başlatıldı")
    
    def load_model(self, model_name: str) -> Any:
        """
        Modeli yükle
        
        Args:
            model_name: Model adı
            
        Returns:
            Yüklenen model objesi
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        metadata = self.registry.get_model(model_name)
        if not metadata:
            logger.error(f"Model bulunamadı: {model_name}")
            return None
        
        # Modeli yükle (framework'e göre)
        try:
            if metadata.framework == "ultralytics":
                from ultralytics import YOLO
                model = YOLO(metadata.path)
            else:
                logger.error(f"Bilinmeyen framework: {metadata.framework}")
                return None
            
            self.loaded_models[model_name] = model
            logger.info(f"Model yüklendi: {model_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model yükleme hatası ({model_name}): {e}")
            return None
    
    def unload_model(self, model_name: str):
        """
        Modeli bellekten çıkar
        
        Args:
            model_name: Model adı
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Model bellekten çıkarıldı: {model_name}")
    
    def get_active_model(self, model_type: ModelType) -> Optional[Any]:
        """
        Aktif modeli al
        
        Args:
            model_type: Model tipi
            
        Returns:
            Aktif model objesi
        """
        model_name = self.active_models.get(model_type)
        if model_name:
            return self.loaded_models.get(model_name)
        return None
    
    def update_active_models(
        self,
        current_fps: Optional[float] = None,
        current_gpu_memory: Optional[float] = None,
    ):
        """
        Aktif modelleri güncelle (dinamik seçim)
        
        Args:
            current_fps: Mevcut FPS
            current_gpu_memory: Mevcut GPU kullanımı
        """
        for model_type in ModelType:
            selected = self.selector.select_model(
                model_type,
                current_fps,
                current_gpu_memory,
            )
            
            if selected:
                current_name = self.active_models.get(model_type)
                
                # Model değişikliği gerekli mi?
                if current_name != selected.name:
                    # Yeni modeli yükle
                    self.load_model(selected.name)
                    
                    # Eski modeli çıkar (eğer başka yerde kullanılmıyorsa)
                    if current_name and current_name not in self.active_models.values():
                        self.unload_model(current_name)
                    
                    self.active_models[model_type] = selected.name
    
    def benchmark_model(
        self,
        model_name: str,
        test_images: List[Any],
        num_runs: int = 100,
    ) -> Optional[ModelBenchmark]:
        """
        Modeli benchmark et
        
        Args:
            model_name: Model adı
            test_images: Test görüntüleri
            num_runs: Çalıştırma sayısı
            
        Returns:
            Benchmark sonucu
        """
        model = self.load_model(model_name)
        if not model:
            return None
        
        metadata = self.registry.get_model(model_name)
        if not metadata:
            return None
        
        # Warmup
        for _ in range(10):
            if test_images:
                model(test_images[0])
        
        # Timing
        inference_times = []
        for _ in range(num_runs):
            if not test_images:
                break
            
            start = time.time()
            model(test_images[0])
            elapsed = (time.time() - start) * 1000  # ms
            inference_times.append(elapsed)
        
        if not inference_times:
            return None
        
        # İstatistikler
        import numpy as np
        times = np.array(inference_times)
        
        benchmark = ModelBenchmark(
            model_name=model_name,
            num_frames=len(inference_times),
            resolution=metadata.input_size,
            avg_inference_time_ms=float(np.mean(times)),
            min_inference_time_ms=float(np.min(times)),
            max_inference_time_ms=float(np.max(times)),
            std_inference_time_ms=float(np.std(times)),
            avg_fps=1000.0 / float(np.mean(times)),
        )
        
        self.registry.add_benchmark(benchmark)
        
        logger.info(f"Benchmark tamamlandı: {model_name}, FPS={benchmark.avg_fps:.2f}")
        
        return benchmark
    
    def get_recommendation(
        self,
        target_fps: Optional[float] = None,
        target_map: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Model önerisi al
        
        Args:
            target_fps: Hedef FPS
            target_map: Hedef mAP
            
        Returns:
            Öneri dict'i
        """
        models = self.registry.list_models()
        
        recommendations = []
        for model in models:
            perf = self.registry.get_average_performance(model.name)
            
            if perf:
                fps_ok = not target_fps or perf['avg_fps'] >= target_fps
                map_ok = not target_map or perf['avg_map'] >= target_map
                
                if fps_ok and map_ok:
                    score = perf['avg_fps'] * perf['avg_map']
                    recommendations.append((score, model, perf))
        
        if recommendations:
            recommendations.sort(key=lambda x: x[0], reverse=True)
            best = recommendations[0]
            return {
                'model': best[1].name,
                'expected_fps': best[2]['avg_fps'],
                'expected_map': best[2]['avg_map'],
                'gpu_memory_mb': best[2]['avg_gpu_memory_mb'],
            }
        
        return {'model': None, 'reason': 'No suitable model found'}
