"""
Pozisyon Kestirimi Modülü
Visual Odometry + Kinematik EKF + Smooth State Refinement
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pose2D:
    """2D pozisyon ve yönelim"""
    x: float  # X koordinatı (metre)
    y: float  # Y koordinatı (metre)
    theta: float  # Yönelim açısı (radyan)
    timestamp: float = 0.0
    confidence: float = 1.0  # Güven skoru [0, 1]
    
    def to_array(self) -> np.ndarray:
        """Numpy array'e çevir"""
        return np.array([self.x, self.y, self.theta])
    
    def distance_to(self, other: 'Pose2D') -> float:
        """Başka bir pozisyona uzaklık"""
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx**2 + dy**2)
    
    def angle_difference(self, other: 'Pose2D') -> float:
        """Açı farkını hesapla"""
        diff = self.theta - other.theta
        # Normalize to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        return diff


@dataclass
class State:
    """Kinematik model durumu"""
    # Durum vektörü: [x, y, theta, vx, vy, omega]
    x: float
    y: float
    theta: float
    vx: float = 0.0  # X hızı
    vy: float = 0.0  # Y hızı
    omega: float = 0.0  # Açısal hız
    
    # Kovaryans matrisi (6x6)
    P: np.ndarray = field(default_factory=lambda: np.eye(6) * 0.1)
    
    def to_vector(self) -> np.ndarray:
        """Durum vektörü olarak döndür"""
        return np.array([self.x, self.y, self.theta, self.vx, self.vy, self.omega])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, P: Optional[np.ndarray] = None) -> 'State':
        """Vektörden State oluştur"""
        if P is None:
            P = np.eye(6) * 0.1
        return cls(
            x=vec[0], y=vec[1], theta=vec[2],
            vx=vec[3], vy=vec[4], omega=vec[5],
            P=P
        )


class VelocityEstimator:
    """
    Hız kestirici
    Nesne tespitlerinden hız hesaplar
    """
    
    def __init__(self, window_size: int = 5):
        """
        Hız kestirici başlat
        
        Args:
            window_size: Hareketli pencere boyutu
        """
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
    
    def update(self, pose: Pose2D, dt: float):
        """
        Yeni pozisyon ekle ve hızı güncelle
        
        Args:
            pose: Mevcut pozisyon
            dt: Zaman adımı
        """
        self.history.append(pose)
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Hızı hesapla
        
        Returns:
            (vx, vy, omega) - Doğrusal ve açısal hızlar
        """
        if len(self.history) < 2:
            return 0.0, 0.0, 0.0
        
        # Son iki pozisyondan hız hesapla
        pose_new = self.history[-1]
        pose_old = self.history[-2]
        
        dt = pose_new.timestamp - pose_old.timestamp
        if dt <= 0:
            return 0.0, 0.0, 0.0
        
        vx = (pose_new.x - pose_old.x) / dt
        vy = (pose_new.y - pose_old.y) / dt
        omega = pose_new.angle_difference(pose_old) / dt
        
        return vx, vy, omega
    
    def get_smoothed_velocity(self) -> Tuple[float, float, float]:
        """
        Hareketli ortalama ile düzgünleştirilmiş hız
        """
        if len(self.history) < 2:
            return 0.0, 0.0, 0.0
        
        velocities = []
        for i in range(1, len(self.history)):
            pose_new = self.history[i]
            pose_old = self.history[i-1]
            
            dt = pose_new.timestamp - pose_old.timestamp
            if dt > 0:
                vx = (pose_new.x - pose_old.x) / dt
                vy = (pose_new.y - pose_old.y) / dt
                omega = pose_new.angle_difference(pose_old) / dt
                velocities.append((vx, vy, omega))
        
        if not velocities:
            return 0.0, 0.0, 0.0
        
        # Ortalama al
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        avg_omega = np.mean([v[2] for v in velocities])
        
        return avg_vx, avg_vy, avg_omega


class VisualOdometry:
    """
    Visual Odometry - Görüntü akışından kamera hareketi kestirimi
    """
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        feature_type: str = 'orb',
        min_features: int = 20,
    ):
        """
        Visual Odometry başlat
        
        Args:
            camera_matrix: Kamera matrisi (3x3)
            dist_coeffs: Distorsiyon katsayıları
            feature_type: 'orb', 'sift', veya 'fast'
            min_features: Minimum gerekli feature sayısı
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.min_features = min_features
        
        # Feature detector
        if feature_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=1000)
        elif feature_type == 'sift':
            self.detector = cv2.SIFT_create()
        else:  # fast
            self.detector = cv2.FastFeatureDetector_create()
        
        # Descriptor extractor (FAST için ORB kullan)
        if feature_type == 'fast':
            self.extractor = cv2.ORB_create()
        else:
            self.extractor = self.detector
        
        # Matcher
        if feature_type == 'sift':
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        # Durum
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        
        self.pose = Pose2D(0, 0, 0)
        self.trajectory = [self.pose]
        
        logger.info(f"Visual Odometry başlatıldı: {feature_type}")
    
    def _extract_features(self, gray: np.ndarray) -> Tuple[List, np.ndarray]:
        """Feature'ları çıkar"""
        kp = self.detector.detect(gray, None)
        kp, desc = self.extractor.compute(gray, kp)
        return kp, desc
    
    def _match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        kp1: List,
        kp2: List,
    ) -> Tuple[List[cv2.DMatch], List[cv2.DMatch]]:
        """Feature'ları eşleştir"""
        if desc1 is None or desc2 is None:
            return [], []
        
        # KNN match
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches, matches
    
    def _estimate_motion(
        self,
        kp1: List,
        kp2: List,
        matches: List[cv2.DMatch],
    ) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Feature eşleşmelerinden hareket kestir
        
        Returns:
            (R, t, inlier_matches) - Dönüş matrisi, öteleme vektörü, inlier'lar
        """
        if len(matches) < self.min_features:
            return np.eye(3), np.zeros(3), []
        
        # Eşleşen feature'ların koordinatları
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Essential matrix hesapla
        E, mask = cv2.findEssentialMat(
            src_pts, dst_pts,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        
        if E is None or mask is None:
            return np.eye(3), np.zeros(3), []
        
        # Pose'u recover et
        _, R, t, mask = cv2.recoverPose(
            E, src_pts, dst_pts,
            self.camera_matrix,
            mask=mask,
        )
        
        # Inlier match'leri filtrele
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        
        return R, t, inlier_matches
    
    def process_frame(self, image: np.ndarray) -> Optional[Pose2D]:
        """
        Frame'i işle ve pozisyon güncelle
        
        Args:
            image: Giriş görüntüsü (BGR)
            
        Returns:
            Güncellenen pozisyon (veya None başlangıç için)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Undistort
        gray = cv2.undistort(gray, self.camera_matrix, self.dist_coeffs)
        
        # Feature'ları çıkar
        kp, desc = self._extract_features(gray)
        
        # İlk frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_kp = kp
            self.prev_desc = desc
            return None
        
        # Feature'ları eşleştir
        good_matches, _ = self._match_features(self.prev_desc, desc, self.prev_kp, kp)
        
        # Hareketi kestir
        R, t, inliers = self._estimate_motion(self.prev_kp, kp, good_matches)
        
        # Yeterli inlier varsa pozisyon güncelle
        if len(inliers) >= self.min_features:
            # Dönüş matrisinden açıyı al
            theta = np.arctan2(R[1, 0], R[0, 0])
            
            # Öteleme
            dx, dy = t[0, 0], t[1, 0]
            
            # Global koordinatlara çevir (basit approach)
            cos_theta = np.cos(self.pose.theta)
            sin_theta = np.sin(self.pose.theta)
            
            # Rotasyon matrisi
            R_global = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])
            
            # Global hareket
            d_global = R_global @ np.array([dx, dy])
            
            # Pozisyon güncelle
            self.pose.x += d_global[0]
            self.pose.y += d_global[1]
            self.pose.theta += theta
            
            # Normalize angle
            self.pose.theta = (self.pose.theta + np.pi) % (2 * np.pi) - np.pi
            
            self.trajectory.append(Pose2D(
                self.pose.x, self.pose.y, self.pose.theta,
                confidence=len(inliers) / len(good_matches) if good_matches else 0
            ))
        
        # Önceki frame'i güncelle
        self.prev_gray = gray
        self.prev_kp = kp
        self.prev_desc = desc
        
        return self.pose
    
    def get_trajectory(self) -> List[Pose2D]:
        """Trajektoriyi döndür"""
        return self.trajectory
    
    def reset(self):
        """VO'yu sıfırla"""
        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.pose = Pose2D(0, 0, 0)
        self.trajectory = [self.pose]


class KinematicEKF:
    """
    Kinematik Extended Kalman Filter
    Durum: [x, y, theta, vx, vy, omega]
    Ölçüm: [x, y, theta] (görüntü eşleme veya tespit)
    """
    
    def __init__(
        self,
        process_noise: float = 0.1,
        measurement_noise: float = 0.5,
        initial_covariance: float = 0.1,
    ):
        """
        Kinematik EKF başlat
        
        Args:
            process_noise: Süreç gürültü kovaryansı
            measurement_noise: Ölçüm gürültü kovaryansı
            initial_covariance: Başlangıç kovaryansı
        """
        # Durum vektörü: [x, y, theta, vx, vy, omega]
        self.state = np.zeros(6)
        
        # Kovaryans matrisi
        self.P = np.eye(6) * initial_covariance
        
        # Süreç gürültü kovaryansı
        self.Q = np.eye(6) * process_noise
        
        # Ölçüm gürültü kovaryansı (sadece x, y, theta ölçülüyor)
        self.R = np.diag([measurement_noise, measurement_noise, measurement_noise * 0.5])
        
        # Ölçüm matrisi (x, y, theta ölçülüyor)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # theta
        
        self.initialized = False
        
        logger.info("Kinematik EKF başlatıldı")
    
    def init_from_measurement(self, measurement: np.ndarray):
        """
        İlk ölçümle başlat
        
        Args:
            measurement: [x, y, theta]
        """
        self.state[:3] = measurement
        self.state[3:] = 0  # Hızlar sıfır
        self.initialized = True
    
    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        """
        Tahmin adımı
        
        Args:
            dt: Zaman adımı
            control: Kontrol girişi [ax, ay, alpha] (opsiyonel)
        """
        if not self.initialized:
            return
        
        # Durum转移 (kinematik model)
        x, y, theta, vx, vy, omega = self.state
        
        # Yeni pozisyon
        x_new = x + vx * dt
        y_new = y + vy * dt
        theta_new = theta + omega * dt
        
        # Kontrol girişi varsa hızları güncelle
        if control is not None:
            ax, ay, alpha = control
            vx_new = vx + ax * dt
            vy_new = vy + ay * dt
            omega_new = omega + alpha * dt
        else:
            vx_new = vx
            vy_new = vy
            omega_new = omega
        
        # Jacobian matrisi (durum转移 için)
        self.F = np.eye(6)
        self.F[0, 3] = dt  # dx/dvx
        self.F[1, 4] = dt  # dy/dvy
        self.F[2, 5] = dt  # theta/domega
        
        # Durumu güncelle
        self.state = np.array([x_new, y_new, theta_new, vx_new, vy_new, omega_new])
        
        # Kovaryansı güncelle
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray):
        """
        Güncelleme adımı
        
        Args:
            measurement: [x, y, theta]
        """
        if not self.initialized:
            self.init_from_measurement(measurement)
            return
        
        # Ölçüm yenilemesi
        z = measurement
        z_pred = self.H @ self.state
        
        # Kalman kazancı
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Durumu güncelle
        y = z - z_pred
        self.state = self.state + K @ y
        
        # Açıyı normalize et
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi
        
        # Kovaryansı güncelle
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
    
    def get_state(self) -> State:
        """Durumu döndür"""
        return State.from_vector(self.state, self.P)
    
    def get_position(self) -> Pose2D:
        """Pozisyonu döndür"""
        return Pose2D(
            x=self.state[0],
            y=self.state[1],
            theta=self.state[2],
            confidence=1.0 / (1.0 + np.trace(self.P[:3, :3]))
        )
    
    def reset(self):
        """EKF'yi sıfırla"""
        self.state = np.zeros(6)
        self.P = np.eye(6) * 0.1
        self.initialized = False


class SmoothStateRefinement:
    """
    Smooth State Refinement (SSR)
    Pozisyon kestirimlerini düzgünleştirir ve düzeltir
    """
    
    def __init__(
        self,
        window_size: int = 10,
        smoothing_factor: float = 0.5,
        max_acceleration: float = 10.0,
        max_angular_acceleration: float = 5.0,
    ):
        """
        SSR başlat
        
        Args:
            window_size: Düzgünleştirme penceresi
            smoothing_factor: Düzgünleştirme faktörü [0, 1]
            max_acceleration: Maksimum doğrusal ivme (m/s²)
            max_angular_acceleration: Maksimum açısal ivme (rad/s²)
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.max_acceleration = max_acceleration
        self.max_angular_acceleration = max_angular_acceleration
        
        self.poses = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size)
        
        self.current_pose = Pose2D(0, 0, 0)
        
        logger.info("Smooth State Refinement başlatıldı")
    
    def update(self, new_pose: Pose2D, dt: float) -> Pose2D:
        """
        Yeni pozisyon ekle ve düzgünleştir
        
        Args:
            new_pose: Yeni ham pozisyon
            dt: Zaman adımı
            
        Returns:
            Düzgünleştirilmiş pozisyon
        """
        self.poses.append(new_pose)
        
        # Hız hesapla
        if len(self.poses) >= 2:
            prev_pose = self.poses[-2]
            vx = (new_pose.x - prev_pose.x) / dt
            vy = (new_pose.y - prev_pose.y) / dt
            omega = new_pose.angle_difference(prev_pose) / dt
            self.velocities.append((vx, vy, omega))
        
        # Fiziksel kısıtlamaları kontrol et
        if len(self.velocities) >= 2:
            prev_vx, prev_vy, prev_omega = self.velocities[-2]
            curr_vx, curr_vy, curr_omega = self.velocities[-1]
            
            # İvmeleri hesapla
            ax = (curr_vx - prev_vx) / dt
            ay = (curr_vy - prev_vy) / dt
            alpha = (curr_omega - prev_omega) / dt
            
            # İvmeleri sınırla
            linear_acc = np.sqrt(ax**2 + ay**2)
            if linear_acc > self.max_acceleration:
                scale = self.max_acceleration / linear_acc
                curr_vx = prev_vx + ax * dt * scale
                curr_vy = prev_vy + ay * dt * scale
            
            if abs(alpha) > self.max_angular_acceleration:
                scale = self.max_angular_acceleration / abs(alpha)
                curr_omega = prev_omega + alpha * dt * scale
            
            # Hızları güncelle
            self.velocities[-1] = (curr_vx, curr_vy, curr_omega)
        
        # Düzgünleştirme
        if len(self.poses) >= 3:
            # Hareketli ortalama
            poses_list = list(self.poses)
            
            # Ağırlıklı ortalama (daha yeni pozisyonlara daha fazla ağırlık)
            weights = np.linspace(0.5, 1.0, len(poses_list))
            weights = weights / weights.sum()
            
            smooth_x = sum(p.x * w for p, w in zip(poses_list, weights))
            smooth_y = sum(p.y * w for p, w in zip(poses_list, weights))
            
            # Açıyı düzgünleştir (daha dikkatli, wrapping sorunları yüzünden)
            angles = [p.theta for p in poses_list]
            smooth_theta = self._smooth_angles(angles, weights)
            
            # Ham ve düzgünleştirilmiş arasında interpolasyon
            self.current_pose = Pose2D(
                x=new_pose.x * (1 - self.smoothing_factor) + smooth_x * self.smoothing_factor,
                y=new_pose.y * (1 - self.smoothing_factor) + smooth_y * self.smoothing_factor,
                theta=self._interpolate_angle(
                    new_pose.theta, smooth_theta, self.smoothing_factor
                ),
            )
        else:
            self.current_pose = new_pose
        
        return self.current_pose
    
    def _smooth_angles(self, angles: List[float], weights: np.ndarray) -> float:
        """Açıları düzgünleştir (wrapping consideration)"""
        # Referans açı
        ref_angle = angles[-1]
        
        # Tüm açıları referansa göre normalize et
        normalized = []
        for angle in angles:
            diff = angle - ref_angle
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            normalized.append(ref_angle + diff)
        
        # Ağırlıklı ortalama
        smooth_angle = sum(a * w for a, w in zip(normalized, weights))
        
        return smooth_angle
    
    def _interpolate_angle(self, angle1: float, angle2: float, t: float) -> float:
        """İki açı arasında interpolasyon"""
        diff = angle2 - angle1
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        return angle1 + diff * t
    
    def get_pose(self) -> Pose2D:
        """Mevcut pozisyonu döndür"""
        return self.current_pose
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Mevcut hızı döndür"""
        if len(self.velocities) > 0:
            return self.velocities[-1]
        return 0.0, 0.0, 0.0
    
    def reset(self):
        """SSR'yi sıfırla"""
        self.poses.clear()
        self.velocities.clear()
        self.current_pose = Pose2D(0, 0, 0)


class PositionEstimator:
    """
    Pozisyon Kestirici
    Visual Odometry + Kinematik EKF + Smooth State Refinement kombinasyonu
    """
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        ekf_process_noise: float = 0.1,
        ekf_measurement_noise: float = 0.5,
        ssr_window_size: int = 10,
    ):
        """
        Pozisyon kestirici başlat
        
        Args:
            camera_matrix: Kamera matrisi
            dist_coeffs: Distorsiyon katsayıları
            ekf_process_noise: EKF süreç gürültüsü
            ekf_measurement_noise: EKF ölçüm gürültüsü
            ssr_window_size: SSR pencere boyutu
        """
        # Visual Odometry
        self.vo = VisualOdometry(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        
        # Kinematik EKF
        self.ekf = KinematicEKF(
            process_noise=ekf_process_noise,
            measurement_noise=ekf_measurement_noise,
        )
        
        # Smooth State Refinement
        self.ssr = SmoothStateRefinement(
            window_size=ssr_window_size,
        )
        
        self.current_pose = Pose2D(0, 0, 0)
        self.initialized = False
        
        logger.info("Pozisyon Kestirici başlatıldı")
    
    def process_frame(
        self,
        image: np.ndarray,
        dt: float,
        external_measurement: Optional[np.ndarray] = None,
    ) -> Pose2D:
        """
        Frame'i işle ve pozisyon kestir
        
        Args:
            image: Giriş görüntüsü
            dt: Zaman adımı
            external_measurement: Harici ölçüm [x, y, theta] (opsiyonel)
            
        Returns:
            Kestirilen pozisyon
        """
        # Visual Odometry ile hareket kestir
        vo_pose = self.vo.process_frame(image)
        
        # Ölçümü belirle (external measurement varsa onu kullan)
        if external_measurement is not None:
            measurement = external_measurement
        elif vo_pose is not None:
            measurement = vo_pose.to_array()
        else:
            # Ölçüm yok, sadece tahmin
            self.ekf.predict(dt)
            ekf_state = self.ekf.get_position()
            
            # SSR uygula
            self.current_pose = self.ssr.update(ekf_state, dt)
            return self.current_pose
        
        # EKF predict
        self.ekf.predict(dt)
        
        # EKF update
        self.ekf.update(measurement)
        
        # EKF'den pozisyon al
        ekf_pose = self.ekf.get_position()
        
        # SSR uygula
        self.current_pose = self.ssr.update(ekf_pose, dt)
        
        return self.current_pose
    
    def get_pose(self) -> Pose2D:
        """Mevcut pozisyonu döndür"""
        return self.current_pose
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Mevcut hızı döndür"""
        return self.ssr.get_velocity()
    
    def get_trajectory(self) -> List[Pose2D]:
        """VO trajektoriğini döndür"""
        return self.vo.get_trajectory()
    
    def reset(self):
        """Pozisyon kestiriciyi sıfırla"""
        self.vo.reset()
        self.ekf.reset()
        self.ssr.reset()
        self.current_pose = Pose2D(0, 0, 0)
        self.initialized = False
