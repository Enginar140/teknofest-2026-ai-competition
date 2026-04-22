"""
Sunucu Modülü
Teknofest HYZ sunucusuna JSON API üzerinden bağlantı

İki tip bağlantı sınıfı:
1. TeknofestConnectionHandler: Resmi yarışma sunucusu için (HTTP REST API)
2. ServerConnection: Özel sunucu bağlantısı için (WebSocket)
"""

from .connection import (
    ConnectionStatus,
    ServerConfig,
    DetectionResult,
    PositionResult,
    TelemetryData,
    TeknofestProtocol,
    ServerConnection,
    ConnectionManager,
    create_detection_from_dict,
    create_position_from_dict,
)

from .teknofest_connection import TeknofestConnectionHandler
from .detected_translation import DetectedTranslation

__all__ = [
    # Orijinal bağlantı sınıfları
    'ConnectionStatus',
    'ServerConfig',
    'DetectionResult',
    'PositionResult',
    'TelemetryData',
    'TeknofestProtocol',
    'ServerConnection',
    'ConnectionManager',
    'create_detection_from_dict',
    'create_position_from_dict',
    # Resmi yarışma sınıfları
    'TeknofestConnectionHandler',
    'DetectedTranslation',
]
