"""
Görüntü Eşleme Modülü
XoFTR + LightGlue + Kalman Filter ile feature matching
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchedPair:
    """Eşleşmiş feature çifti"""
    kp1: Tuple[float, float]  # İlk görüntüdeki keypoint
    kp2: Tuple[float, float]  # İkinci görüntüdeki keypoint
    confidence: float  # Eşleşme güveni
    descriptor: Optional[np.ndarray] = None
    
    @property
    def distance(self) -> float:
        """İki nokta arasındaki mesafe"""
        dx = self.kp1[0] - self.kp2[0]
        dy = self.kp1[1] - self.kp2[1]
        return np.sqrt(dx**2 + dy**2)


@dataclass
class HomographyResult:
    """Homografi sonucu"""
    H: np.ndarray  # 3x3 homografi matrisi
    inlier_count: int
    inlier_ratio: float
    confidence: float
    matches: List[MatchedPair]


class FeatureMatcher:
    """
    Temel Feature Matcher sınıfı
    Farklı matcher algoritmaları için base class
    """
    
    def __init__(
        self,
        min_matches: int = 10,
        ransac_threshold: float = 5.0,
    ):
        """
        Feature matcher başlat
        
        Args:
            min_matches: Minimum gerekli eşleşme sayısı
            ransac_threshold: RANSAC threshold
        """
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        
        self.prev_image = None
        self.prev_kp = None
        self.prev_desc = None
        
        logger.info(f"FeatureMatcher başlatıldı: min_matches={min_matches}")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Feature tespiti ve descriptor hesaplama
        Override edilmeli
        
        Returns:
            (keypoints, descriptors)
        """
        raise NotImplementedError
    
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[cv2.DMatch]:
        """
        Descriptor'ları eşleştir
        Override edilmeli
        
        Returns:
            Match listesi
        """
        raise NotImplementedError
    
    def estimate_homography(
        self,
        kp1: List,
        kp2: List,
        matches: List[cv2.DMatch],
    ) -> HomographyResult:
        """
        RANSAC ile homografi kestir
        
        Args:
            kp1: İlk görüntü keypoint'leri
            kp2: İkinci görüntü keypoint'leri
            matches: Eşleşmeler
            
        Returns:
            HomographyResult
        """
        if len(matches) < self.min_matches:
            return HomographyResult(
                H=np.eye(3),
                inlier_count=0,
                inlier_ratio=0.0,
                confidence=0.0,
                matches=[],
            )
        
        # Eşleşen noktaları al
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # RANSAC ile homografi hesapla
        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            self.ransac_threshold,
        )
        
        if H is None:
            return HomographyResult(
                H=np.eye(3),
                inlier_count=0,
                inlier_ratio=0.0,
                confidence=0.0,
                matches=[],
            )
        
        # Inlier'ları say
        inlier_count = int(np.sum(mask))
        inlier_ratio = inlier_count / len(matches)
        
        # MatchedPair'leri oluştur
        matched_pairs = []
        for i, m in enumerate(matches):
            if mask[i]:
                matched_pairs.append(MatchedPair(
                    kp1=kp1[m.queryIdx].pt,
                    kp2=kp2[m.trainIdx].pt,
                    confidence=1.0 - m.distance,
                ))
        
        # Güven skoru (inlier ratio * inlier count)
        confidence = inlier_ratio * min(inlier_count / 50.0, 1.0)
        
        return HomographyResult(
            H=H,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            confidence=confidence,
            matches=matched_pairs,
        )
    
    def process_frame(
        self,
        image: np.ndarray,
    ) -> Optional[HomographyResult]:
        """
        Frame'i işle ve önceki frame ile eşleştir
        
        Args:
            image: Giriş görüntüsü (BGR)
            
        Returns:
            HomographyResult (veya None ilk frame için)
        """
        # Feature'ları çıkar
        kp, desc = self.detect_and_compute(image)
        
        # İlk frame
        if self.prev_image is None:
            self.prev_image = image
            self.prev_kp = kp
            self.prev_desc = desc
            return None
        
        # Eşleştir
        matches = self.match(self.prev_desc, desc)
        
        # Homografi kestir
        result = self.estimate_homography(self.prev_kp, kp, matches)
        
        # Önceki frame'i güncelle
        self.prev_image = image
        self.prev_kp = kp
        self.prev_desc = desc
        
        return result
    
    def reset(self):
        """Matcher'ı sıfırla"""
        self.prev_image = None
        self.prev_kp = None
        self.prev_desc = None


class ORBMatcher(FeatureMatcher):
    """ORB tabanlı feature matcher"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        # Matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        logger.info("ORB Matcher başlatıldı")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        return kp, desc
    
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []
        
        # KNN match
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        return good_matches


class SIFTMatcher(FeatureMatcher):
    """SIFT tabanlı feature matcher"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=2000)
        
        # Matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        logger.info("SIFT Matcher başlatıldı")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(gray, None)
        return kp, desc
    
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []
        
        # KNN match
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches


class XoFTRMatcher(FeatureMatcher):
    """
    XoFTR (X-former Feature Matching)
    Transformer tabanlı feature matching
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        **kwargs
    ):
        """
        XoFTR matcher başlat
        
        Args:
            model_path: Model dosya yolu
            device: 'cuda' veya 'cpu'
        """
        super().__init__(**kwargs)
        
        self.device = device
        self.model_path = model_path
        
        # Model yükle (opsiyonel - LoFTR kullanabiliriz)
        try:
            # LoFTR benzeri bir yaklaşım
            from kornia.feature import LoFTR
            self.matcher = LoFTR(pretrained='outdoor').to(device)
            self.use_kornia = True
            logger.info("Kornia LoFTR yüklendi")
        except ImportError:
            logger.warning("Kornia LoFTR bulunamadı, ORB fallback kullanılıyor")
            self.orb = cv2.ORB_create(nfeatures=2000)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.use_kornia = False
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Feature tespiti (fallback için)"""
        if self.use_kornia:
            # Kornia LoFTR otomatik feature tespiti yapar
            return [], np.array([])
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, desc = self.orb.detectAndCompute(gray, None)
            return kp, desc
    
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[cv2.DMatch]:
        """Eşleştirme (fallback için)"""
        if self.use_kornia:
            # Kornia LoFTR process_frame içinde çağrılır
            return []
        else:
            if desc1 is None or desc2 is None:
                return []
            
            matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            return good_matches
    
    def process_frame(
        self,
        image: np.ndarray,
    ) -> Optional[HomographyResult]:
        """
        Frame'i işle (Kornia LoFTR ile)
        """
        if self.use_kornia:
            import torch
            import kornia
            
            # Görüntüyü tensor'a çevir
            img_tensor = kornia.image_to_tensor(image, keepdim=False).float() / 255.0
            img_tensor = img_tensor.to(self.device)
            
            # İlk frame
            if self.prev_image is None:
                self.prev_image_tensor = img_tensor
                return None
            
            # Grayscale'e çevir
            img_gray = kornia.color.rgb_to_grayscale(img_tensor)
            prev_gray = kornia.color.rgb_to_grayscale(self.prev_image_tensor)
            
            # LoFTR ile eşleştir
            with torch.no_grad():
                input_dict = {
                    "image0": kornia.color.rgb_to_grayscale(self.prev_image_tensor),
                    "image1": img_gray,
                }
                correspondences = self.matcher(input_dict)
            
            # Keypoint'leri al
            kp1 = correspondences['keypoints0'].cpu().numpy()
            kp2 = correspondences['keypoints1'].cpu().numpy()
            confidence = correspondences['confidence'].cpu().numpy()
            
            # Güvenli eşleşmeleri filtrele
            mask = confidence > 0.5
            kp1 = kp1[mask]
            kp2 = kp2[mask]
            confidence = confidence[mask]
            
            # OpenCV formatına çevir
            kp1_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in kp1]
            kp2_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in kp2]
            
            # Matches oluştur
            matches = [cv2.DMatch(_i=i, _queryIdx=i, _trainIdx=i, _distance=1-c) 
                      for i, c in enumerate(confidence)]
            
            # Homografi kestir
            result = self.estimate_homography(kp1_cv, kp2_cv, matches)
            
            # Önceki frame'i güncelle
            self.prev_image_tensor = img_tensor
            
            return result
        else:
            return super().process_frame(image)


class LightGlueMatcher(FeatureMatcher):
    """
    LightGlue matcher
    Hafif ve hızlı feature matching
    """
    
    def __init__(
        self,
        feature_extractor: str = 'superpoint',
        device: str = 'cuda',
        **kwargs
    ):
        """
        LightGlue matcher başlat
        
        Args:
            feature_extractor: 'superpoint' veya 'disk'
            device: 'cuda' veya 'cpu'
        """
        super().__init__(**kwargs)
        
        self.device = device
        self.feature_extractor = feature_extractor
        
        # Model yükle
        try:
            from lightglue import LightGlue, SuperPoint, DISK
            
            if feature_extractor == 'superpoint':
                self.extractor = SuperPoint(max_num_keypoints=512).to(device)
            elif feature_extractor == 'disk':
                self.extractor = DISK(max_num_keypoints=512).to(device)
            else:
                raise ValueError(f"Bilinmeyen feature extractor: {feature_extractor}")
            
            self.matcher = LightGlue(feature_extractor).to(device)
            self.use_lightglue = True
            
            logger.info(f"LightGlue başlatıldı: {feature_extractor}")
        except ImportError:
            logger.warning("LightGlue bulunamadı, ORB fallback kullanılıyor")
            self.orb = cv2.ORB_create(nfeatures=2000)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.use_lightglue = False
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Feature tespiti (fallback için)"""
        if self.use_lightglue:
            # LightGlue otomatik feature tespiti yapar
            return [], np.array([])
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, desc = self.orb.detectAndCompute(gray, None)
            return kp, desc
    
    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
    ) -> List[cv2.DMatch]:
        """Eşleştirme (fallback için)"""
        if self.use_lightglue:
            # LightGlue process_frame içinde çağrılır
            return []
        else:
            if desc1 is None or desc2 is None:
                return []
            
            matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            return good_matches
    
    def process_frame(
        self,
        image: np.ndarray,
    ) -> Optional[HomographyResult]:
        """
        Frame'i işle (LightGlue ile)
        """
        if self.use_lightglue:
            import torch
            
            # Görüntüyü tensor'a çevir
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(self.device)
            
            # İlk frame
            if self.prev_image is None:
                self.prev_image_tensor = img_tensor
                return None
            
            # Feature extraction ve matching
            with torch.no_grad():
                feats0 = self.extractor.extract(img_tensor)
                feats1 = self.extractor.extract(self.prev_image_tensor)
                matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            
            # Keypoint'leri al
            kp1 = matches01['keypoints0'].cpu().numpy()
            kp2 = matches01['keypoints1'].cpu().numpy()
            confidence = matches01['scores'].cpu().numpy()
            
            # OpenCV formatına çevir
            kp1_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in kp1]
            kp2_cv = [cv2.KeyPoint(x=p[0], y=p[1], size=1) for p in kp2]
            
            # Matches oluştur
            matches = [cv2.DMatch(_i=i, _queryIdx=i, _trainIdx=i, _distance=1-c) 
                      for i, c in enumerate(confidence)]
            
            # Homografi kestir
            result = self.estimate_homography(kp1_cv, kp2_cv, matches)
            
            # Önceki frame'i güncelle
            self.prev_image_tensor = img_tensor
            
            return result
        else:
            return super().process_frame(image)


class KalmanFeatureTracker:
    """
    Kalman Filter ile Feature Tracking
    Feature hareketlerini takip ve kestir
    """
    
    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
        max_features: int = 100,
    ):
        """
        Kalman feature tracker başlat
        
        Args:
            process_noise: Süreç gürültüsü
            measurement_noise: Ölçüm gürültüsü
            max_features: Maksimum takip edilecek feature sayısı
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_features = max_features
        
        # Her feature için Kalman filter
        self.filters = {}  # {feature_id: Kalman filter}
        self.next_id = 1
        
        # Feature geçmişi
        self.feature_history = {}  # {feature_id: deque of positions}
        
        logger.info("Kalman Feature Tracker başlatıldı")
    
    def _create_kalman_filter(self, initial_pos: Tuple[float, float]) -> cv2.KalmanFilter:
        """Yeni Kalman filter oluştur"""
        kf = cv2.KalmanFilter(4, 2)  # State: [x, y, vx, vy], Measurement: [x, y]
        
        # Transition matrix (kinematik model)
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        
        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        
        # Initial state
        kf.statePre = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        kf.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32)
        
        return kf
    
    def update(
        self,
        detections: List[Tuple[float, float]],
    ) -> Dict[int, Tuple[float, float]]:
        """
        Yeni tespitlerle tracker'ı güncelle
        
        Args:
            detections: Tespit edilen pozisyonlar listesi [(x, y), ...]
            
        Returns:
            Feature ID -> Pozisyon mapping
        """
        # Basit nearest neighbor association
        tracked_positions = {}
        
        if not self.filters:
            # İlk frame - yeni feature'lar oluştur
            for pos in detections[:self.max_features]:
                kf = self._create_kalman_filter(pos)
                self.filters[self.next_id] = kf
                self.feature_history[self.next_id] = deque(maxlen=10)
                self.feature_history[self.next_id].append(pos)
                tracked_positions[self.next_id] = pos
                self.next_id += 1
        else:
            # Association ve güncelleme
            detections_array = np.array(detections)
            unassigned = set(range(len(detections)))
            
            # Mevcut feature'ları predict et
            predicted = {}
            for fid, kf in self.filters.items():
                pred = kf.predict()
                predicted[fid] = (pred[0, 0], pred[1, 0])
            
            # Nearest neighbor association
            for fid, pred_pos in predicted.items():
                if not unassigned:
                    break
                
                # En yakın detection'ı bul
                distances = np.array([
                    np.sqrt((pred_pos[0] - d[0])**2 + (pred_pos[1] - d[1])**2)
                    for i, d in enumerate(detections_array)
                    if i in unassigned
                ])
                
                if len(distances) > 0:
                    min_idx = list(unassigned)[np.argmin(distances)]
                    min_dist = np.min(distances)
                    
                    if min_dist < 50:  # Association threshold
                        # Update
                        measurement = np.array([[detections_array[min_idx][0]], [detections_array[min_idx][1]]], dtype=np.float32)
                        kf.correct(measurement)
                        
                        tracked_positions[fid] = detections_array[min_idx]
                        self.feature_history[fid].append(detections_array[min_idx])
                        unassigned.remove(min_idx)
            
            # Yeni feature'lar ekle (capacity varsa)
            for idx in list(unassigned)[:self.max_features - len(self.filters)]:
                pos = detections_array[idx]
                kf = self._create_kalman_filter(pos)
                self.filters[self.next_id] = kf
                self.feature_history[self.next_id] = deque(maxlen=10)
                self.feature_history[self.next_id].append(pos)
                tracked_positions[self.next_id] = pos
                self.next_id += 1
        
        return tracked_positions
    
    def get_predicted_positions(self) -> Dict[int, Tuple[float, float]]:
        """Tüm feature'lar için kestirilen pozisyonları döndür"""
        predicted = {}
        for fid, kf in self.filters.items():
            state = kf.statePost
            predicted[fid] = (state[0, 0], state[1, 0])
        return predicted
    
    def get_velocities(self) -> Dict[int, Tuple[float, float]]:
        """Feature hızlarını döndür"""
        velocities = {}
        for fid, kf in self.filters.items():
            state = kf.statePost
            velocities[fid] = (state[2, 0], state[3, 0])
        return velocities
    
    def reset(self):
        """Tracker'ı sıfırla"""
        self.filters.clear()
        self.feature_history.clear()
        self.next_id = 1


class ImageMatchingPipeline:
    """
    Görüntü Eşleme Pipeline
    Feature matching + Kalman tracking kombinasyonu
    """
    
    def __init__(
        self,
        matcher_type: str = 'orb',
        use_kalman_tracking: bool = True,
        camera_matrix: Optional[np.ndarray] = None,
    ):
        """
        Image matching pipeline başlat
        
        Args:
            matcher_type: 'orb', 'sift', 'xoftr', veya 'lightglue'
            use_kalman_tracking: Kalman tracking kullan
            camera_matrix: Kamera matrisi (opsiyonel)
        """
        self.matcher_type = matcher_type
        self.use_kalman_tracking = use_kalman_tracking
        self.camera_matrix = camera_matrix
        
        # Matcher oluştur
        if matcher_type == 'orb':
            self.matcher = ORBMatcher()
        elif matcher_type == 'sift':
            self.matcher = SIFTMatcher()
        elif matcher_type == 'xoftr':
            self.matcher = XoFTRMatcher()
        elif matcher_type == 'lightglue':
            self.matcher = LightGlueMatcher()
        else:
            raise ValueError(f"Bilinmeyen matcher tipi: {matcher_type}")
        
        # Kalman tracker
        if use_kalman_tracking:
            self.kalman_tracker = KalmanFeatureTracker()
        
        self.homography_history = deque(maxlen=10)
        
        logger.info(f"Image Matching Pipeline başlatıldı: {matcher_type}")
    
    def process_frame(
        self,
        image: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Frame'i işle
        
        Args:
            image: Giriş görüntüsü (BGR)
            
        Returns:
            Dict with keys:
                - homography: Homografi matrisi
                - inlier_count: Inlier sayısı
                - confidence: Güven skoru
                - tracked_features: Takip edilen feature'lar (Kalman varsa)
        """
        # Feature matching
        result = self.matcher.process_frame(image)
        
        if result is None:
            return {
                'homography': np.eye(3),
                'inlier_count': 0,
                'confidence': 0.0,
                'tracked_features': {},
            }
        
        # Homografi history'e ekle
        self.homography_history.append(result.H)
        
        output = {
            'homography': result.H,
            'inlier_count': result.inlier_count,
            'confidence': result.confidence,
        }
        
        # Kalman tracking
        if self.use_kalman_tracking:
            # Matched keypoint'leri detections olarak kullan
            detections = [m.kp2 for m in result.matches]
            tracked = self.kalman_tracker.update(detections)
            output['tracked_features'] = tracked
        else:
            output['tracked_features'] = {}
        
        return output
    
    def get_accumulated_homography(self) -> np.ndarray:
        """Birikmiş homografiyi hesapla"""
        if len(self.homography_history) == 0:
            return np.eye(3)
        
        H_acc = np.eye(3)
        for H in self.homography_history:
            H_acc = H @ H_acc
        
        return H_acc
    
    def reset(self):
        """Pipeline'ı sıfırla"""
        self.matcher.reset()
        if self.use_kalman_tracking:
            self.kalman_tracker.reset()
        self.homography_history.clear()
