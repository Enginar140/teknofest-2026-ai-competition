"""
Teknofest Resmi DetectedTranslation Sınıfı
Havacılıkta Yapay Zeka Yarışması için tahmin gönderme sınıfı
"""


class DetectedTranslation:
    """
    Tespit Edilmiş Translation (Konum) Verisi
    
    Görev 2: Pozisyon Tespiti için sunucuya gönderilecek veri formatı
    
    Kullanım:
        translation = DetectedTranslation(
            translation_x=1.5,
            translation_y=2.3,
            translation_z=100.0
        )
        # Sunucuya gönder
        connection.send_prediction(translation)
    """
    
    def __init__(self,
                 translation_x: float,
                 translation_y: float,
                 translation_z: float):
        """
        Args:
            translation_x: X koordinatı (metre)
            translation_y: Y koordinatı (metre)
            translation_z: Z koordinatı (metre, yükseklik)
        """
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.translation_z = translation_z
    
    def create_payload(self, base_url: str) -> dict:
        """
        Sunucuya gönderilecek payload'ı oluştur
        
        Args:
            base_url: Sunucu base URL'i
        
        Returns:
            dict: Sunucuya gönderilecek JSON formatındaki veri
        """
        return {
            'translation_x': str(self.translation_x),
            'translation_y': str(self.translation_y),
            'translation_z': str(self.translation_z)
        }
    
    def to_dict(self) -> dict:
        """Sözlük formatında döndür"""
        return {
            'translation_x': float(self.translation_x),
            'translation_y': float(self.translation_y),
            'translation_z': float(self.translation_z)
        }
    
    def __repr__(self) -> str:
        return f"DetectedTranslation(x={self.translation_x}, y={self.translation_y}, z={self.translation_z})"
