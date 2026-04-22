"""
Test ve Optimizasyon Modülü
Sistem testleri, benchmark'lar ve optimizasyon araçları
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import cv2

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test sonucu"""
    test_name: str
    passed: bool
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Dict'e çevir"""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'duration': self.duration,
            'details': self.details,
            'error_message': self.error_message,
        }


@dataclass
class BenchmarkResult:
    """Benchmark sonucu"""
    benchmark_name: str
    duration: float
    iterations: int
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    throughput: float  # ops per second
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Dict'e çevir"""
        return {
            'benchmark_name': self.benchmark_name,
            'duration': self.duration,
            'iterations': self.iterations,
            'avg_time': self.avg_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'std_time': self.std_time,
            'throughput': self.throughput,
            'metadata': self.metadata,
        }


class SystemTester:
    """
    Sistem Testçisi
    Sistem bileşenlerini test eder
    """
    
    def __init__(self):
        """Sistem testçisi başlat"""
        self.results: List[TestResult] = []
        logger.info("SystemTester başlatıldı")
    
    def run_all_tests(self) -> List[TestResult]:
        """Tüm testleri çalıştır"""
        self.results = []
        
        # Testleri çalıştır
        self.results.append(self.test_camera_access())
        self.results.append(self.test_gpu_availability())
        self.results.append(self.test_model_files())
        self.results.append(self.test_config_files())
        self.results.append(self.test_memory_allocation())
        
        return self.results
    
    def test_camera_access(self) -> TestResult:
        """Kamera erişimini test et"""
        start_time = time.time()
        
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return TestResult(
                    test_name="Camera Access",
                    passed=False,
                    duration=time.time() - start_time,
                    error_message="Kamera açılamadı",
                )
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return TestResult(
                    test_name="Camera Access",
                    passed=False,
                    duration=time.time() - start_time,
                    error_message="Frame okunamadı",
                )
            
            return TestResult(
                test_name="Camera Access",
                passed=True,
                duration=time.time() - start_time,
                details={'frame_shape': frame.shape},
            )
            
        except Exception as e:
            return TestResult(
                test_name="Camera Access",
                passed=False,
                duration=time.time() - start_time,
                error_message=str(e),
            )
    
    def test_gpu_availability(self) -> TestResult:
        """GPU kullanılabilirliğini test et"""
        start_time = time.time()
        
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            
            details = {
                'cuda_available': cuda_available,
            }
            
            if cuda_available:
                details['cuda_version'] = torch.version.cuda
                details['gpu_name'] = torch.cuda.get_device_name(0)
                details['gpu_memory_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            
            return TestResult(
                test_name="GPU Availability",
                passed=cuda_available,
                duration=time.time() - start_time,
                details=details,
            )
            
        except Exception as e:
            return TestResult(
                test_name="GPU Availability",
                passed=False,
                duration=time.time() - start_time,
                error_message=str(e),
            )
    
    def test_model_files(self) -> TestResult:
        """Model dosyalarını test et"""
        start_time = time.time()
        
        try:
            model_files = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
            found_files = []
            missing_files = []
            
            for model_file in model_files:
                if Path(model_file).exists():
                    found_files.append(model_file)
                else:
                    missing_files.append(model_file)
            
            return TestResult(
                test_name="Model Files",
                passed=len(found_files) > 0,
                duration=time.time() - start_time,
                details={
                    'found_files': found_files,
                    'missing_files': missing_files,
                },
            )
            
        except Exception as e:
            return TestResult(
                test_name="Model Files",
                passed=False,
                duration=time.time() - start_time,
                error_message=str(e),
            )
    
    def test_config_files(self) -> TestResult:
        """Konfigürasyon dosyalarını test et"""
        start_time = time.time()
        
        try:
            config_paths = [
                'config/default_config.json',
                'config/system_config.json',
            ]
            
            found_files = []
            missing_files = []
            
            for config_path in config_paths:
                if Path(config_path).exists():
                    found_files.append(config_path)
                else:
                    missing_files.append(config_path)
            
            return TestResult(
                test_name="Config Files",
                passed=len(found_files) >= 1,  # En az biri olmalı
                duration=time.time() - start_time,
                details={
                    'found_files': found_files,
                    'missing_files': missing_files,
                },
            )
            
        except Exception as e:
            return TestResult(
                test_name="Config Files",
                passed=False,
                duration=time.time() - start_time,
                error_message=str(e),
            )
    
    def test_memory_allocation(self) -> TestResult:
        """Bellek ayırmayı test et"""
        start_time = time.time()
        
        try:
            # Büyük bellek ayırmayı test et
            test_size = 1024 * 1024 * 100  # 100MB
            test_array = np.zeros(test_size, dtype=np.float32)
            
            memory_allocated = test_array.nbytes / (1024 ** 2)  # MB
            
            return TestResult(
                test_name="Memory Allocation",
                passed=True,
                duration=time.time() - start_time,
                details={
                    'memory_allocated_mb': memory_allocated,
                },
            )
            
        except Exception as e:
            return TestResult(
                test_name="Memory Allocation",
                passed=False,
                duration=time.time() - start_time,
                error_message=str(e),
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Test özetini al"""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        return {
            'total': len(self.results),
            'passed': passed,
            'failed': failed,
            'success_rate': passed / len(self.results) if self.results else 0,
        }


class BenchmarkRunner:
    """
    Benchmark Çalıştırıcı
    Sistem performansını ölçer
    """
    
    def __init__(self):
        """Benchmark çalıştırıcı başlat"""
        self.results: List[BenchmarkResult] = []
        logger.info("BenchmarkRunner başlatıldı")
    
    def run_detection_benchmark(
        self,
        detector,
        test_image: np.ndarray,
        iterations: int = 100,
    ) -> BenchmarkResult:
        """
        Tespit benchmark'ı çalıştır
        
        Args:
            detector: Nesne tespiti modülü
            test_image: Test görüntüsü
            iterations: İterasyon sayısı
            
        Returns:
            Benchmark sonucu
        """
        logger.info(f"Tespit benchmark'ı başlatıldı: {iterations} iterasyon")
        
        times = []
        detections_count = []
        
        # Warmup
        for _ in range(10):
            detector.process_frame(test_image)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            iter_start = time.time()
            detections = detector.process_frame(test_image)
            iter_time = (time.time() - iter_start) * 1000  # ms
            times.append(iter_time)
            detections_count.append(len(detections))
        
        total_time = time.time() - start_time
        
        times_array = np.array(times)
        
        result = BenchmarkResult(
            benchmark_name="Detection Benchmark",
            duration=total_time,
            iterations=iterations,
            avg_time=float(np.mean(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            std_time=float(np.std(times_array)),
            throughput=iterations / total_time,
            metadata={
                'avg_detections': float(np.mean(detections_count)),
                'image_shape': test_image.shape,
            },
        )
        
        self.results.append(result)
        logger.info(f"Benchmark tamamlandı: {result.avg_time:.2f}ms avg")
        
        return result
    
    def run_position_benchmark(
        self,
        position_estimator,
        test_images: List[np.ndarray],
        iterations: int = 100,
    ) -> BenchmarkResult:
        """
        Pozisyon kestirimi benchmark'ı çalıştır
        
        Args:
            position_estimator: Pozisyon kestirimi modülü
            test_images: Test görüntüleri listesi
            iterations: İterasyon sayısı
            
        Returns:
            Benchmark sonucu
        """
        logger.info(f"Pozisyon benchmark'ı başlatıldı: {iterations} iterasyon")
        
        times = []
        dt = 1.0 / 30.0  # 30 FPS varsayılan
        
        # Benchmark
        start_time = time.time()
        for i in range(iterations):
            iter_start = time.time()
            image = test_images[i % len(test_images)]
            position = position_estimator.process_frame(image, dt=dt)
            iter_time = (time.time() - iter_start) * 1000  # ms
            times.append(iter_time)
        
        total_time = time.time() - start_time
        
        times_array = np.array(times)
        
        result = BenchmarkResult(
            benchmark_name="Position Estimation Benchmark",
            duration=total_time,
            iterations=iterations,
            avg_time=float(np.mean(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            std_time=float(np.std(times_array)),
            throughput=iterations / total_time,
        )
        
        self.results.append(result)
        logger.info(f"Benchmark tamamlandı: {result.avg_time:.2f}ms avg")
        
        return result
    
    def run_matching_benchmark(
        self,
        matcher,
        test_images: List[np.ndarray],
        iterations: int = 100,
    ) -> BenchmarkResult:
        """
        Görüntü eşleme benchmark'ı çalıştır
        
        Args:
            matcher: Görüntü eşleme modülü
            test_images: Test görüntüleri listesi
            iterations: İterasyon sayısı
            
        Returns:
            Benchmark sonucu
        """
        logger.info(f"Eşleme benchmark'ı başlatıldı: {iterations} iterasyon")
        
        times = []
        inlier_counts = []
        
        # Benchmark
        start_time = time.time()
        for i in range(iterations):
            iter_start = time.time()
            result = matcher.process_frame(test_images[i % len(test_images)])
            iter_time = (time.time() - iter_start) * 1000  # ms
            times.append(iter_time)
            
            if result and hasattr(result, 'inlier_count'):
                inlier_counts.append(result.inlier_count)
        
        total_time = time.time() - start_time
        
        times_array = np.array(times)
        
        result = BenchmarkResult(
            benchmark_name="Image Matching Benchmark",
            duration=total_time,
            iterations=iterations,
            avg_time=float(np.mean(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            std_time=float(np.std(times_array)),
            throughput=iterations / total_time,
            metadata={
                'avg_inliers': float(np.mean(inlier_counts)) if inlier_counts else 0,
            },
        )
        
        self.results.append(result)
        logger.info(f"Benchmark tamamlandı: {result.avg_time:.2f}ms avg")
        
        return result
    
    def run_full_pipeline_benchmark(
        self,
        detector,
        position_estimator,
        matcher,
        test_images: List[np.ndarray],
        iterations: int = 100,
    ) -> BenchmarkResult:
        """
        Tam pipeline benchmark'ı çalıştır
        
        Args:
            detector: Nesne tespiti modülü
            position_estimator: Pozisyon kestirimi modülü
            matcher: Görüntü eşleme modülü
            test_images: Test görüntüleri listesi
            iterations: İterasyon sayısı
            
        Returns:
            Benchmark sonucu
        """
        logger.info(f"Tam pipeline benchmark'ı başlatıldı: {iterations} iterasyon")
        
        times = []
        detection_counts = []
        dt = 1.0 / 30.0
        
        # Benchmark
        start_time = time.time()
        for i in range(iterations):
            iter_start = time.time()
            image = test_images[i % len(test_images)]
            
            # Tespit
            detections = detector.process_frame(image)
            detection_counts.append(len(detections))
            
            # Pozisyon
            position = position_estimator.process_frame(image, dt=dt)
            
            # Eşleme
            matching_result = matcher.process_frame(image)
            
            iter_time = (time.time() - iter_start) * 1000  # ms
            times.append(iter_time)
        
        total_time = time.time() - start_time
        
        times_array = np.array(times)
        
        result = BenchmarkResult(
            benchmark_name="Full Pipeline Benchmark",
            duration=total_time,
            iterations=iterations,
            avg_time=float(np.mean(times_array)),
            min_time=float(np.min(times_array)),
            max_time=float(np.max(times_array)),
            std_time=float(np.std(times_array)),
            throughput=iterations / total_time,
            metadata={
                'avg_detections': float(np.mean(detection_counts)),
                'estimated_fps': float(1000.0 / np.mean(times_array)),
            },
        )
        
        self.results.append(result)
        logger.info(f"Benchmark tamamlandı: {result.avg_time:.2f}ms avg, {result.metadata['estimated_fps']:.2f} FPS")
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Benchmark özetini al"""
        if not self.results:
            return {}
        
        summary = {
            'total_benchmarks': len(self.results),
            'total_duration': sum(r.duration for r in self.results),
            'benchmarks': [r.to_dict() for r in self.results],
        }
        
        return summary


class PerformanceOptimizer:
    """
    Performans Optimizasyon Aracı
    Sistem performansını optimize eder
    """
    
    def __init__(self):
        """Performans optimizasyon aracı başlat"""
        logger.info("PerformanceOptimizer başlatıldı")
    
    def suggest_optimizations(
        self,
        benchmark_results: List[BenchmarkResult],
        target_fps: float = 30.0,
    ) -> List[str]:
        """
        Optimizasyon önerileri
        
        Args:
            benchmark_results: Benchmark sonuçları
            target_fps: Hedef FPS
            
        Returns:
            Öneriler listesi
        """
        suggestions = []
        
        for result in benchmark_results:
            avg_time_ms = result.avg_time
            current_fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
            
            if current_fps < target_fps:
                gap = target_fps - current_fps
                suggestions.append(
                    f"[{result.benchmark_name}] FPS düşük ({current_fps:.1f}), hedef: {target_fps:.1f}. "
                    f"Eksik: {gap:.1f} FPS"
                )
                
                # Spesifik öneriler
                if "Detection" in result.benchmark_name:
                    suggestions.append("  - Daha küçük model kullanmayı deneyin (e.g., yolov8s)")
                    suggestions.append("  - Model quantization kullanın")
                    suggestions.append("  - SAHI'yı devre dışı bırakın")
                
                elif "Position" in result.benchmark_name:
                    suggestions.append("  - Visual Odometry'yi devre dışı bırakın")
                    suggestions.append("  - Feature sayısını azaltın")
                
                elif "Matching" in result.benchmark_name:
                    suggestions.append("  - ORB matcher kullanın (daha hızlı)")
                    suggestions.append("  - Feature sayısını azaltın")
        
        if not suggestions:
            suggestions.append("✅ Performans hedeflere uygun!")
        
        return suggestions
    
    def calculate_model_size_recommendation(
        self,
        current_fps: float,
        target_fps: float,
        current_model_size: str,
    ) -> str:
        """
        Model boyutu önerisi hesapla
        
        Args:
            current_fps: Mevcut FPS
            target_fps: Hedef FPS
            current_model_size: Mevcut model boyutu
            
        Returns:
            Önerilen model boyutu
        """
        size_order = ['n', 's', 'm', 'l', 'x']
        current_idx = size_order.index(current_model_size)
        
        fps_ratio = current_fps / target_fps
        
        if fps_ratio < 0.5:
            # FPS çok düşük, 2 boyut küçült
            recommended_idx = max(0, current_idx - 2)
        elif fps_ratio < 0.8:
            # FPS düşük, 1 boyut küçült
            recommended_idx = max(0, current_idx - 1)
        elif fps_ratio > 1.5:
            # FPS çok yüksek, 1 boyut büyüt
            recommended_idx = min(len(size_order) - 1, current_idx + 1)
        else:
            # FPS uygun
            recommended_idx = current_idx
        
        return size_order[recommended_idx]
    
    def estimate_improvement(
        self,
        current_avg_time: float,
        optimization_type: str,
    ) -> Dict[str, float]:
        """
        Optimizasyon iyileştirmesini tahmin et
        
        Args:
            current_avg_time: Mevcut ortalama süre (ms)
            optimization_type: Optimizasyon tipi
            
        Returns:
            İyileştirme tahmini
        """
        improvement_factors = {
            'quantization': 0.3,  # %30 hızlanma
            'tensorrt': 0.4,  # %40 hızlanma
            'smaller_model': 0.5,  # %50 hızlanma (kabaca)
            'disable_sahi': 0.2,  # %20 hızlanma
            'disable_tracking': 0.15,  # %15 hızlanma
            'disable_vo': 0.25,  # %25 hızlanma
            'simpler_matcher': 0.3,  # %30 hızlanma
        }
        
        factor = improvement_factors.get(optimization_type, 0.0)
        
        new_time = current_avg_time * (1.0 - factor)
        improvement_percent = (current_avg_time - new_time) / current_avg_time * 100
        
        return {
            'current_time_ms': current_avg_time,
            'estimated_new_time_ms': new_time,
            'improvement_percent': improvement_percent,
            'speedup_factor': current_avg_time / new_time if new_time > 0 else 1.0,
        }


class TestRunner:
    """
    Test Çalıştırıcı
    Testleri ve benchmark'ları koordine eder
    """
    
    def __init__(self):
        """Test çalıştırıcı başlat"""
        self.tester = SystemTester()
        self.benchmark = BenchmarkRunner()
        self.optimizer = PerformanceOptimizer()
        logger.info("TestRunner başlatıldı")
    
    def run_full_test_suite(
        self,
        detector=None,
        position_estimator=None,
        matcher=None,
        test_images: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Tam test paketini çalıştır
        
        Args:
            detector: Nesne tespiti modülü
            position_estimator: Pozisyon kestirimi modülü
            matcher: Görüntü eşleme modülü
            test_images: Test görüntüleri
            
        Returns:
            Test sonuçları
        """
        results = {
            'timestamp': time.time(),
            'tests': {},
            'benchmarks': {},
            'recommendations': [],
        }
        
        # Sistem testleri
        test_results = self.tester.run_all_tests()
        results['tests'] = {
            'summary': self.tester.get_summary(),
            'details': [r.to_dict() for r in test_results],
        }
        
        # Benchmark'lar (modüller varsa)
        if detector and test_images:
            benchmark_results = []
            
            # Tespit benchmark
            det_result = self.benchmark.run_detection_benchmark(
                detector, test_images[0]
            )
            benchmark_results.append(det_result)
            
            # Tam pipeline benchmark
            if position_estimator and matcher:
                full_result = self.benchmark.run_full_pipeline_benchmark(
                    detector, position_estimator, matcher, test_images
                )
                benchmark_results.append(full_result)
            
            results['benchmarks'] = {
                'summary': self.benchmark.get_summary(),
                'details': [r.to_dict() for r in benchmark_results],
            }
            
            # Optimizasyon önerileri
            suggestions = self.optimizer.suggest_optimizations(benchmark_results)
            results['recommendations'] = suggestions
        
        return results
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str = "test_report.json",
    ) -> bool:
        """
        Test raporu oluştur
        
        Args:
            results: Test sonuçları
            output_path: Çıktı dosyası
            
        Returns:
            Başarılı mı?
        """
        try:
            import json
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Test raporu oluşturuldu: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Rapor oluşturma hatası: {e}")
            return False
