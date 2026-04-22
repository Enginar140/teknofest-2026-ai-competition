"""
Sunucu Bağlantı Modülü
Teknofest OTR yarışması sunucusuna JSON API üzerinden bağlantı
"""

import json
import socket
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Bağlantı durumları"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ServerConfig:
    """Sunucu konfigürasyonu"""
    host: str = "localhost"
    port: int = 10000
    team_id: str = ""
    team_password: str = ""
    reconnect_interval: float = 5.0
    timeout: float = 10.0
    max_retries: int = 3


@dataclass
class DetectionResult:
    """Tespit sonucu"""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x_center, y_center, width, height] (normalized)
    track_id: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PositionResult:
    """Pozisyon sonucu"""
    x: float
    y: float
    theta: float
    confidence: float
    timestamp: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TelemetryData:
    """Telemetri verisi"""
    timestamp: float
    frame_id: int
    fps: float
    inference_time: float
    detection_count: int
    position: Optional[PositionResult] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.position:
            data['position'] = self.position.to_dict()
        return data


class TeknofestProtocol:
    """
    Teknofest OTR protokolü
    Sunucu ile iletişim formatını yönetir
    """
    
    # Mesaj tipleri
    MSG_TYPE_AUTH = "AUTH"
    MSG_TYPE_DETECTION = "DETECTION"
    MSG_TYPE_POSITION = "POSITION"
    MSG_TYPE_FRAME_START = "FRAME_START"
    MSG_TYPE_FRAME_END = "FRAME_END"
    MSG_TYPE_HEARTBEAT = "HEARTBEAT"
    MSG_TYPE_ERROR = "ERROR"
    
    @staticmethod
    def create_auth_message(team_id: str, password: str) -> Dict:
        """Kimlik doğrulama mesajı oluştur"""
        return {
            "type": TeknofestProtocol.MSG_TYPE_AUTH,
            "team_id": team_id,
            "password": password,
            "timestamp": time.time(),
        }
    
    @staticmethod
    def create_detection_message(
        frame_id: int,
        detections: List[DetectionResult],
    ) -> Dict:
        """Tespit mesajı oluştur"""
        return {
            "type": TeknofestProtocol.MSG_TYPE_DETECTION,
            "frame_id": frame_id,
            "detections": [d.to_dict() for d in detections],
            "timestamp": time.time(),
        }
    
    @staticmethod
    def create_position_message(
        frame_id: int,
        position: PositionResult,
    ) -> Dict:
        """Pozisyon mesajı oluştur"""
        return {
            "type": TeknofestProtocol.MSG_TYPE_POSITION,
            "frame_id": frame_id,
            "position": position.to_dict(),
            "timestamp": time.time(),
        }
    
    @staticmethod
    def create_frame_start_message(frame_id: int) -> Dict:
        """Frame başlangıç mesajı oluştur"""
        return {
            "type": TeknofestProtocol.MSG_TYPE_FRAME_START,
            "frame_id": frame_id,
            "timestamp": time.time(),
        }
    
    @staticmethod
    def create_frame_end_message(
        frame_id: int,
        inference_time: float,
    ) -> Dict:
        """Frame bitiş mesajı oluştur"""
        return {
            "type": TeknofestProtocol.MSG_TYPE_FRAME_END,
            "frame_id": frame_id,
            "inference_time": inference_time,
            "timestamp": time.time(),
        }
    
    @staticmethod
    def create_heartbeat_message() -> Dict:
        """Heartbeat mesajı oluştur"""
        return {
            "type": TeknofestProtocol.MSG_TYPE_HEARTBEAT,
            "timestamp": time.time(),
        }
    
    @staticmethod
    def encode_message(message: Dict) -> bytes:
        """Mesajı JSON'a kodla ve boyut ekle"""
        json_str = json.dumps(message)
        message_bytes = json_str.encode('utf-8')
        
        # Format: [4-byte length][JSON data]
        length = len(message_bytes)
        header = length.to_bytes(4, byteorder='big')
        
        return header + message_bytes
    
    @staticmethod
    def decode_message(data: bytes) -> Optional[Dict]:
        """Mesajı çöz"""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Mesaj decode hatası: {e}")
            return None


class ServerConnection:
    """
    Sunucu bağlantı sınıfı
    TCP/IP üzerinden JSON tabanlı iletişim
    """
    
    def __init__(
        self,
        config: ServerConfig,
        on_status_change: Optional[Callable[[ConnectionStatus], None]] = None,
        on_message_received: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Sunucu bağlantısı başlat
        
        Args:
            config: Sunucu konfigürasyonu
            on_status_change: Durum değişikliği callback'i
            on_message_received: Mesaj alma callback'i
        """
        self.config = config
        self.on_status_change = on_status_change
        self.on_message_received = on_message_received
        
        self.socket = None
        self.status = ConnectionStatus.DISCONNECTED
        
        # Thread'ler
        self.receive_thread = None
        self.send_thread = None
        self.heartbeat_thread = None
        
        # Queue'lar
        self.send_queue = Queue()
        self.receive_queue = Queue()
        
        # Kontrol
        self.running = False
        self.authenticated = False
        
        # İstatistikler
        self.bytes_sent = 0
        self.bytes_received = 0
        self.messages_sent = 0
        self.messages_received = 0
        self.last_heartbeat_time = 0
        
        logger.info(f"ServerConnection başlatıldı: {config.host}:{config.port}")
    
    def _set_status(self, status: ConnectionStatus):
        """Durumu güncelle"""
        self.status = status
        if self.on_status_change:
            self.on_status_change(status)
        logger.info(f"Bağlantı durumu: {status.value}")
    
    def connect(self) -> bool:
        """
        Sunucuya bağlan
        
        Returns:
            Bağlantı başarılı mı?
        """
        if self.status == ConnectionStatus.CONNECTED:
            logger.warning("Zaten bağlı")
            return True
        
        self._set_status(ConnectionStatus.CONNECTING)
        
        try:
            # Socket oluştur
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.config.timeout)
            
            # Bağlan
            logger.info(f"Bağlanılıyor: {self.config.host}:{self.config.port}")
            self.socket.connect((self.config.host, self.config.port))
            
            # Thread'leri başlat
            self.running = True
            self.receive_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True,
            )
            self.receive_thread.start()
            
            self.send_thread = threading.Thread(
                target=self._send_loop,
                daemon=True,
            )
            self.send_thread.start()
            
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
            )
            self.heartbeat_thread.start()
            
            # Kimlik doğrulama
            auth_msg = TeknofestProtocol.create_auth_message(
                self.config.team_id,
                self.config.team_password,
            )
            self.send_message(auth_msg)
            
            self._set_status(ConnectionStatus.CONNECTED)
            return True
            
        except (socket.error, ConnectionRefusedError, socket.timeout) as e:
            logger.error(f"Bağlantı hatası: {e}")
            self._set_status(ConnectionStatus.ERROR)
            self.disconnect()
            return False
    
    def disconnect(self):
        """Bağlantıyı kes"""
        self.running = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        # Thread'leri bekle
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        if self.send_thread:
            self.send_thread.join(timeout=1.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
        
        self.authenticated = False
        self._set_status(ConnectionStatus.DISCONNECTED)
        
        logger.info("Bağlantı kesildi")
    
    def send_message(self, message: Dict):
        """
        Mesaj gönder (queue'ye ekle)
        
        Args:
            message: Gönderilecek mesaj
        """
        self.send_queue.put(message)
    
    def _send_loop(self):
        """Mesaj gönderme thread'i"""
        while self.running:
            try:
                # Queue'den mesaj al (timeout ile kontrol)
                try:
                    message = self.send_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Encode ve gönder
                data = TeknofestProtocol.encode_message(message)
                
                if self.socket:
                    try:
                        self.socket.sendall(data)
                        self.bytes_sent += len(data)
                        self.messages_sent += 1
                    except socket.error as e:
                        logger.error(f"Gönderme hatası: {e}")
                        self._set_status(ConnectionStatus.ERROR)
                        break
                
            except Exception as e:
                logger.error(f"Send loop hatası: {e}")
    
    def _receive_loop(self):
        """Mesaj alma thread'i"""
        while self.running:
            try:
                if not self.socket:
                    break
                
                # Header oku (4-byte length)
                header = b''
                while len(header) < 4:
                    chunk = self.socket.recv(4 - len(header))
                    if not chunk:
                        raise ConnectionError("Bağlantı kesildi")
                    header += chunk
                
                # Length'i çöz
                message_length = int.from_bytes(header, byteorder='big')
                
                # Mesajı oku
                data = b''
                while len(data) < message_length:
                    chunk = self.socket.recv(message_length - len(data))
                    if not chunk:
                        raise ConnectionError("Bağlantı kesildi")
                    data += chunk
                
                # Decode
                message = TeknofestProtocol.decode_message(data)
                if message:
                    self.bytes_received += len(data)
                    self.messages_received += 1
                    self.receive_queue.put(message)
                    
                    if self.on_message_received:
                        self.on_message_received(message)
            
            except socket.timeout:
                continue
            except (socket.error, ConnectionError) as e:
                logger.error(f"Alma hatası: {e}")
                self._set_status(ConnectionStatus.ERROR)
                break
    
    def _heartbeat_loop(self):
        """Heartbeat thread'i"""
        while self.running:
            time.sleep(5.0)  # 5 saniyede bir
            
            if self.status == ConnectionStatus.CONNECTED:
                heartbeat = TeknofestProtocol.create_heartbeat_message()
                self.send_message(heartbeat)
                self.last_heartbeat_time = time.time()
    
    def send_detection(
        self,
        frame_id: int,
        detections: List[DetectionResult],
    ):
        """Tespit sonuçlarını gönder"""
        message = TeknofestProtocol.create_detection_message(
            frame_id, detections
        )
        self.send_message(message)
    
    def send_position(
        self,
        frame_id: int,
        position: PositionResult,
    ):
        """Pozisyon sonucunu gönder"""
        message = TeknofestProtocol.create_position_message(
            frame_id, position
        )
        self.send_message(message)
    
    def send_frame_start(self, frame_id: int):
        """Frame başlangıç mesajı gönder"""
        message = TeknofestProtocol.create_frame_start_message(frame_id)
        self.send_message(message)
    
    def send_frame_end(self, frame_id: int, inference_time: float):
        """Frame bitiş mesajı gönder"""
        message = TeknofestProtocol.create_frame_end_message(
            frame_id, inference_time
        )
        self.send_message(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        return {
            "status": self.status.value,
            "authenticated": self.authenticated,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "last_heartbeat": self.last_heartbeat_time,
        }


class ConnectionManager:
    """
    Bağlantı Yöneticisi
    Otomatik yeniden bağlanma ve hata yönetimi
    """
    
    def __init__(
        self,
        config: ServerConfig,
        on_status_change: Optional[Callable[[ConnectionStatus], None]] = None,
        on_message_received: Optional[Callable[[Dict], None]] = None,
    ):
        """
        Bağlantı yöneticisi başlat
        
        Args:
            config: Sunucu konfigürasyonu
            on_status_change: Durum değişikliği callback'i
            on_message_received: Mesaj alma callback'i
        """
        self.config = config
        self.connection = ServerConnection(
            config, on_status_change, on_message_received
        )
        
        # Yeniden bağlanma thread'i
        self.reconnect_thread = None
        self.reconnect_running = False
        
        logger.info("ConnectionManager başlatıldı")
    
    def start(self):
        """Bağlantı yöneticisini başlat"""
        if self.connection.connect():
            self._start_reconnect_thread()
        else:
            self._start_reconnect_thread()
    
    def stop(self):
        """Bağlantı yöneticisini durdur"""
        self.reconnect_running = False
        if self.reconnect_thread:
            self.reconnect_thread.join(timeout=2.0)
        self.connection.disconnect()
    
    def _start_reconnect_thread(self):
        """Yeniden bağlanma thread'ini başlat"""
        if self.reconnect_thread is None or not self.reconnect_thread.is_alive():
            self.reconnect_running = True
            self.reconnect_thread = threading.Thread(
                target=self._reconnect_loop,
                daemon=True,
            )
            self.reconnect_thread.start()
    
    def _reconnect_loop(self):
        """Yeniden bağlanma döngüsü"""
        retry_count = 0
        
        while self.reconnect_running:
            time.sleep(1.0)
            
            if self.connection.status == ConnectionStatus.CONNECTED:
                retry_count = 0
                continue
            
            if retry_count >= self.config.max_retries:
                logger.error("Maksimum yeniden bağlanma sayısına ulaşıldı")
                break
            
            logger.info(f"Yeniden bağlanma denemesi {retry_count + 1}/{self.config.max_retries}")
            time.sleep(self.config.reconnect_interval)
            
            if self.connection.connect():
                retry_count = 0
            else:
                retry_count += 1
    
    def send_detection(self, frame_id: int, detections: List[DetectionResult]):
        """Tespit sonuçlarını gönder"""
        self.connection.send_detection(frame_id, detections)
    
    def send_position(self, frame_id: int, position: PositionResult):
        """Pozisyon sonucunu gönder"""
        self.connection.send_position(frame_id, position)
    
    def send_frame_start(self, frame_id: int):
        """Frame başlangıç mesajı gönder"""
        self.connection.send_frame_start(frame_id)
    
    def send_frame_end(self, frame_id: int, inference_time: float):
        """Frame bitiş mesajı gönder"""
        self.connection.send_frame_end(frame_id, inference_time)
    
    def get_stats(self) -> Dict[str, Any]:
        """İstatistikleri döndür"""
        return self.connection.get_stats()
    
    @property
    def status(self) -> ConnectionStatus:
        """Bağlantı durumu"""
        return self.connection.status
    
    @property
    def is_connected(self) -> bool:
        """Bağlı mı?"""
        return self.connection.status == ConnectionStatus.CONNECTED


# Helper fonksiyonlar

def create_detection_from_dict(data: Dict) -> DetectionResult:
    """Dict'den DetectionResult oluştur"""
    return DetectionResult(
        class_id=data['class_id'],
        class_name=data['class_name'],
        confidence=data['confidence'],
        bbox=data['bbox'],
        track_id=data.get('track_id'),
    )


def create_position_from_dict(data: Dict) -> PositionResult:
    """Dict'den PositionResult oluştur"""
    return PositionResult(
        x=data['x'],
        y=data['y'],
        theta=data['theta'],
        confidence=data['confidence'],
        timestamp=data['timestamp'],
    )
