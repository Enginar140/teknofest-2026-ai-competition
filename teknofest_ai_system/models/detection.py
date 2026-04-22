"""
Nesne Tespiti Modülü - YOLOv8 + ByteTrack + SAHI
TEKNOFEST Havacılıkta Yapay Zeka (4 sınıf) ile uyumlu isimlendirme
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Tespit edilen nesne"""
    class_id: int
    class_name: str
    confidence: float
    bbox: np.ndarray  # [x1, y1, x2, y2]
    track_id: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Dict'e çevir"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': self.bbox.tolist(),
            'track_id': self.track_id,
        }


class YOLOv8Detector:
    """
    YOLOv8 nesne tespiti sınıfı
    """
    
    # Havacılıkta YZ 2026 — 4 sınıf (özel eğitimli model beklenir)
    CLASS_NAMES = [
        'Tasit',   # 0
        'Insan',   # 1
        'UAP',     # 2
        'UAI',     # 3
    ]
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
    ):
        """
        YOLOv8 detector'ı başlat
        
        Args:
            model_path: Model dosya yolu (.pt)
            device: 'cuda' veya 'cpu'
            conf_threshold: Güven eşiği
            iou_threshold: NMS IOU eşiği
        """
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Model yükle
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(model_path))
            self.model.to(device)
            logger.info(f"YOLOv8 modeli yüklendi: {model_path}")
        except ImportError:
            logger.error("ultralytics kütüphanesi yüklü değil. 'pip install ultralytics' çalıştırın")
            raise
        
        # Warmup
        self._warmup()
    
    def _warmup(self):
        """Model warmup"""
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy_image, verbose=False)
        logger.info("YOLOv8 model warmup tamamlandı")
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Görüntüde nesne tespiti yap
        
        Args:
            image: Giriş görüntüsü (BGR)
            
        Returns:
            Detection listesi
        """
        # Inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else f"class_{class_id}"
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=box,
                    )
                    detections.append(detection)
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Batch görüntülerde tespit yap
        
        Args:
            images: Görüntü listesi
            
        Returns:
            Her görüntü için Detection listesi
        """
        results = self.model(images, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        
        all_detections = []
        
        for result in results:
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else f"class_{class_id}"
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=box,
                    )
                    detections.append(detection)
            
            all_detections.append(detections)
        
        return all_detections


class ByteTrack:
    """
    ByteTrack - Yüksek performanslı nesne takibi
    
    Referans: https://github.com/ifzhang/ByteTrack
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        """
        ByteTrack başlat
        
        Args:
            track_thresh: Takip eşiği
            track_buffer: Takip buffer boyutu (frame)
            match_thresh: Eşleştirme eşiği
            frame_rate: Frame rate
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.next_id = 1
        
        logger.info("ByteTrack başlatıldı")
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """IOU hesapla"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _match_detections(
        self,
        tracked: List[Dict],
        detections: List[Detection],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Tespit edilen nesneleri takip edilen nesnelerle eşleştir
        
        Returns:
            (matched_pairs, unmatched_tracked, unmatched_detections)
        """
        if len(tracked) == 0 or len(detections) == 0:
            return [], list(range(len(tracked))), list(range(len(detections)))
        
        # IOU matrisi hesapla
        iou_matrix = np.zeros((len(tracked), len(detections)))
        
        for i, track in enumerate(tracked):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track['bbox'], det.bbox)
        
        # Eşleştirme (greedy)
        matched_pairs = []
        unmatched_tracked = list(range(len(tracked)))
        unmatched_detections = list(range(len(detections)))
        
        # En yüksek IOU'ları eşleştir
        while True:
            if iou_matrix.size == 0:
                break
            
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            if iou_matrix[i, j] < self.match_thresh:
                break
            
            matched_pairs.append((i, j))
            unmatched_tracked.remove(i)
            unmatched_detections.remove(j)
            
            # Satır ve sütunu sil
            iou_matrix = np.delete(iou_matrix, i, axis=0)
            iou_matrix = np.delete(iou_matrix, j, axis=1)
        
        return matched_pairs, unmatched_tracked, unmatched_detections
    
    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Takibi güncelle
        
        Args:
            detections: Mevcut frame'deki tespitler
            
        Returns:
            Takip ID'si olan tespitler
        """
        self.frame_id += 1
        
        # Takip edilen nesneleri dict'e çevir
        tracked = [
            {
                'id': t['id'],
                'bbox': t['bbox'],
                'class_id': t['class_id'],
                'class_name': t['class_name'],
                'confidence': t['confidence'],
            }
            for t in self.tracked_tracks
        ]
        
        # Eşleştirme yap
        matched_pairs, unmatched_tracked, unmatched_detections = self._match_detections(
            tracked, detections
        )
        
        # Eşleşen tespitleri güncelle
        updated_detections = []
        
        for track_idx, det_idx in matched_pairs:
            track = tracked[track_idx]
            detection = detections[det_idx]
            
            # Takip ID'sini ata
            detection.track_id = track['id']
            updated_detections.append(detection)
            
            # Takip edilen nesneleri güncelle
            self.tracked_tracks[track_idx]['bbox'] = detection.bbox
            self.tracked_tracks[track_idx]['age'] += 1
        
        # Eşleşmeyen tespitleri yeni takip olarak ekle
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            detection.track_id = self.next_id
            
            self.tracked_tracks.append({
                'id': self.next_id,
                'bbox': detection.bbox,
                'class_id': detection.class_id,
                'class_name': detection.class_name,
                'confidence': detection.confidence,
                'age': 1,
                'time_since_update': 0,
            })
            
            self.next_id += 1
            updated_detections.append(detection)
        
        # Eşleşmeyen takipleri güncelle
        for track_idx in unmatched_tracked:
            self.tracked_tracks[track_idx]['time_since_update'] += 1
        
        # Eski takipleri kaldır
        self.tracked_tracks = [
            t for t in self.tracked_tracks
            if t['time_since_update'] < self.track_buffer
        ]
        
        return updated_detections
    
    def reset(self):
        """Takibi sıfırla"""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.next_id = 1


class SAHIDetector:
    """
    SAHI (Sliced Aided Hyper Inference)
    Büyük görüntülerde daha iyi tespit için slice-based inference
    """
    
    def __init__(
        self,
        detector: YOLOv8Detector,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_ratio: float = 0.1,
    ):
        """
        SAHI detector başlat
        
        Args:
            detector: Temel detector (YOLOv8Detector)
            slice_height: Slice yüksekliği
            slice_width: Slice genişliği
            overlap_ratio: Slice'lar arasında overlap oranı
        """
        self.detector = detector
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        
        logger.info(f"SAHI detector başlatıldı: {slice_width}x{slice_height}, overlap={overlap_ratio}")
    
    def _generate_slices(
        self,
        image: np.ndarray,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Görüntüyü slice'lara böl
        
        Returns:
            (slice_image, (x_min, y_min, x_max, y_max)) listesi
        """
        h, w = image.shape[:2]
        
        # Overlap piksel sayısı
        overlap_h = int(self.slice_height * self.overlap_ratio)
        overlap_w = int(self.slice_width * self.overlap_ratio)
        
        stride_h = self.slice_height - overlap_h
        stride_w = self.slice_width - overlap_w
        
        slices = []
        
        y = 0
        while y < h:
            y_end = min(y + self.slice_height, h)
            y_start = max(0, y_end - self.slice_height)
            
            x = 0
            while x < w:
                x_end = min(x + self.slice_width, w)
                x_start = max(0, x_end - self.slice_width)
                
                slice_img = image[y_start:y_end, x_start:x_end]
                slices.append((slice_img, (x_start, y_start, x_end, y_end)))
                
                x += stride_w
                if x >= w:
                    break
            
            y += stride_h
            if y >= h:
                break
        
        return slices
    
    def _merge_detections(
        self,
        slice_detections: List[Tuple[List[Detection], Tuple[int, int, int, int]]],
    ) -> List[Detection]:
        """
        Slice'lardan gelen tespitleri birleştir ve NMS uygula
        """
        all_detections = []
        
        # Tespitleri orijinal koordinatlara çevir
        for detections, (x_min, y_min, x_max, y_max) in slice_detections:
            for det in detections:
                # Koordinatları offset'e göre güncelle
                det.bbox[0] += x_min
                det.bbox[1] += y_min
                det.bbox[2] += x_min
                det.bbox[3] += y_min
                
                all_detections.append(det)
        
        # NMS uygula
        if len(all_detections) == 0:
            return []
        
        # Güven skoruna göre sırala
        all_detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # NMS
        keep = []
        while len(all_detections) > 0:
            current = all_detections.pop(0)
            keep.append(current)
            
            # Çakışan tespitleri kaldır
            remaining = []
            for det in all_detections:
                iou = self._iou(current.bbox, det.bbox)
                if iou < 0.5:  # NMS eşiği
                    remaining.append(det)
            
            all_detections = remaining
        
        return keep
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """IOU hesapla"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        SAHI ile tespit yap
        
        Args:
            image: Giriş görüntüsü
            
        Returns:
            Detection listesi
        """
        h, w = image.shape[:2]
        
        # Eğer görüntü küçükse direkt tespit yap
        if h <= self.slice_height and w <= self.slice_width:
            return self.detector.detect(image)
        
        # Slice'lara böl
        slices = self._generate_slices(image)
        
        # Her slice'da tespit yap
        slice_detections = []
        for slice_img, coords in slices:
            detections = self.detector.detect(slice_img)
            slice_detections.append((detections, coords))
        
        # Tespitleri birleştir
        merged_detections = self._merge_detections(slice_detections)
        
        return merged_detections


class ObjectTracker:
    """
    Nesne takibi yöneticisi
    YOLOv8 + ByteTrack + SAHI'yi birleştiren üst seviye sınıf
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_sahi: bool = False,
        use_tracking: bool = True,
        conf_threshold: float = 0.45,
    ):
        """
        Object tracker başlat
        
        Args:
            model_path: YOLOv8 model yolu
            device: 'cuda' veya 'cpu'
            use_sahi: SAHI kullanma
            use_tracking: ByteTrack kullanma
            conf_threshold: Güven eşiği
        """
        self.detector = YOLOv8Detector(
            model_path=model_path,
            device=device,
            conf_threshold=conf_threshold,
        )
        
        self.use_sahi = use_sahi
        if use_sahi:
            self.sahi_detector = SAHIDetector(self.detector)
        
        self.use_tracking = use_tracking
        if use_tracking:
            self.tracker = ByteTrack()
        
        self.frame_count = 0
        self.inference_times = []
        
        logger.info(f"ObjectTracker başlatıldı (SAHI={use_sahi}, Tracking={use_tracking})")
    
    def process_frame(self, image: np.ndarray) -> Tuple[List[Detection], float]:
        """
        Frame'i işle
        
        Args:
            image: Giriş görüntüsü (BGR)
            
        Returns:
            (detections, inference_time)
        """
        self.frame_count += 1
        
        start_time = time.time()
        
        # Tespit yap
        if self.use_sahi:
            detections = self.sahi_detector.detect(image)
        else:
            detections = self.detector.detect(image)
        
        # Takibi güncelle
        if self.use_tracking:
            detections = self.tracker.update(detections)
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return detections, inference_time
    
    def get_stats(self) -> Dict:
        """İstatistikleri döndür"""
        if len(self.inference_times) == 0:
            return {}
        
        times = np.array(self.inference_times[-100:])  # Son 100 frame
        
        return {
            'frame_count': self.frame_count,
            'avg_inference_time': float(np.mean(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'fps': float(1.0 / np.mean(times)) if np.mean(times) > 0 else 0,
        }
    
    def reset_tracker(self):
        """Tracker'ı sıfırla"""
        if self.use_tracking:
            self.tracker.reset()
        self.frame_count = 0
        self.inference_times = []
