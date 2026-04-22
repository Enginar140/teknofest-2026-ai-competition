"""
Teknofest Resmi Sunucu Bağlantı Sınıfı
Havacılıkta Yapay Zeka Yarışması resmi bağlantı arayüzü
"""

import json
import logging
import requests
import time
import os
from typing import List, Dict, Optional
from pathlib import Path

from core.rate_limit import PredictionThrottle


class TeknofestConnectionHandler:
    """
    Teknofest HYZ Yarışması Resmi Sunucu Bağlantı Sınıfı
    
    Özellikler:
    - Login with username/password
    - get_frames(): Frame yollarını alır
    - get_translations(): Konum verilerini alır (translation_x, translation_y, translation_z)
    - send_prediction(): Tahmin sonuçlarını gönderir
    
    Rate Limiting:
    - get_frames: 5 istek/dakika
    - send_prediction: 80 frame/dakika
    """
    
    def __init__(self, base_url: str, username: str = None, password: str = None):
        self.base_url = base_url
        self.auth_token = None
        self.classes = None
        self.frames = None
        self.translations = None
        self.frames_file = "frames.json"
        self.translations_file = "translations.json"
        self.video_name = ''
        self.img_save_path = './_images/'
        
        # URL'leri tanımla
        self.url_login = self.base_url + "auth/"
        self.url_frames = self.base_url + "frames/"
        self.url_translations = self.base_url + "translation/"
        self.url_prediction = self.base_url + "prediction/"
        self.url_session = self.base_url + "session/"
        
        self.logger = logging.getLogger(__name__)
        
        self._prediction_throttle = PredictionThrottle(max_per_minute=80)

        if username and password:
            self.login(username, password)
    
    def login(self, username: str, password: str) -> bool:
        """Sunucuya login ol"""
        payload = {'username': username, 'password': password}
        files = []
        try:
            response = requests.post(self.url_login, data=payload, files=files, timeout=10)
            response_json = json.loads(response.text)
            if response.status_code == 200:
                self.auth_token = response_json['token']
                self.logger.info(f"Login başarılı: {username}")
                return True
            else:
                self.logger.error(f"Login başarısız: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Login isteği başarısız: {e}")
            return False
    
    def get_session_name(self) -> Optional[str]:
        """Oturum ismini al"""
        try:
            from decouple import config
            config.search_path = "../config/"
            return config("SESSION_NAME", default=None)
        except:
            return None
    
    def save_session_name(self, session_name: str) -> str:
        """Oturum ismini kaydet"""
        env_path = Path("./config/.env")
        env_path.parent.mkdir(parents=True, exist_ok=True)
        
        found = False
        change = False
        
        if env_path.exists():
            with open(env_path, "r+") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if line.startswith("SESSION_NAME="):
                        if session_name == line.split("=")[-1].strip():
                            found = True
                            self.logger.info(f"{session_name} zaten var, aynısı kullanılıyor")
                            return session_name
                        else:
                            lines[i] = f"SESSION_NAME={session_name}\n"
                            change = True
                
                if change:
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
                    self.logger.info(f"Oturum değiştirildi: {session_name}")
                    return session_name
                
                if not found:
                    lines.append(f"\nSESSION_NAME={session_name}\n")
                    f.seek(0)
                    f.writelines(lines)
                    f.truncate()
                    self.logger.info(f"Yeni oturum kaydedildi: {session_name}")
                
                return session_name
        else:
            with open(env_path, "w") as f:
                f.write(f"SESSION_NAME={session_name}\n")
            self.logger.info(f"Yeni .env dosyası oluşturuldu: {session_name}")
            return session_name
    
    def create_img_folder(self, path: str):
        """Görüntü kaydetme klasörünü oluştur"""
        post_path = os.path.join(self.img_save_path, path)
        os.makedirs(post_path, exist_ok=True)
    
    def save_frames_to_file(self, frames: List[Dict]):
        """Frameleri dosyaya kaydet"""
        try:
            self.video_name = frames[0]['video_name'] + '/'
            self.create_img_folder(self.video_name)
            session_name = self.save_session_name(frames[0]['video_name'])
            
            frames_path = os.path.join(self.img_save_path, self.video_name, self.frames_file)
            
            with open(frames_path, 'w') as f:
                json.dump(frames, f)
            self.logger.info(f"Frameler kaydedildi: {frames_path}")
        except (IndexError, KeyError) as e:
            self.logger.error(f"Frame kaydetme hatası: {e}")
            raise
    
    def load_frames_from_file(self, session_name: str) -> Optional[List[Dict]]:
        """Frameleri dosyadan yükle"""
        base_path = os.path.join(self.img_save_path, session_name, self.frames_file)
        if os.path.exists(base_path):
            with open(base_path, 'r') as f:
                frames = json.load(f)
            self.logger.info(f"Frameler yüklendi: {base_path}")
            return frames
        self.logger.warning(f"{base_path} bulunamadı")
        return None
    
    def get_frames(self, force_download: bool = False, retries: int = 5, initial_wait_time: float = 0.1) -> Optional[List[Dict]]:
        """
        Frameleri sunucudan al
        
        Dikkat: Bir dakika içerisinde maksimum 5 adet get_frames isteği atabilirsiniz.
        """
        if not force_download:
            try:
                if os.path.exists(self.img_save_path):
                    session_name = self.get_session_name()
                    if session_name:
                        frames = self.load_frames_from_file(session_name)
                        if frames:
                            self.video_name = session_name + "/"
                            self.logger.info("Frameler dosyadan yüklendi (cache)")
                            return frames
            except Exception as e:
                self.logger.info(f"Frame dosyası bozuk: {e}")
        
        payload = {}
        headers = {'Authorization': f'Token {self.auth_token}'}
        wait_time = initial_wait_time
        
        for attempt in range(retries):
            try:
                response = requests.get(self.url_frames, headers=headers, data=payload, timeout=60)
                self.frames = json.loads(response.text)
                
                if response.status_code == 200:
                    self.logger.info(f"get_frames başarılı: {len(self.frames)} frame")
                    self.save_frames_to_file(self.frames)
                    return self.frames
                else:
                    self.logger.error(f"get_frames başarısız: {response.text}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"get_frames isteği hatası: {e}")
            
            self.logger.info(f"{wait_time} saniye sonra yeniden deneniyor...")
            time.sleep(wait_time)
            wait_time *= 2
        
        self.logger.error("get_frames başarısız, dosyadan yüklenmeye çalışılıyor")
        session_name = self.get_session_name()
        if session_name:
            return self.load_frames_from_file(session_name)
        return None
    
    def save_translations_to_file(self, translations: List[Dict]):
        """Translation verilerini dosyaya kaydet"""
        try:
            translations_path = os.path.join(self.img_save_path, self.video_name, self.translations_file)
            
            with open(translations_path, 'w') as f:
                json.dump(translations, f)
            
            self.logger.info(f"Translation kaydedildi: {translations_path}")
        except Exception as e:
            self.logger.warning(f"Translation kaydetme hatası: {e}")
    
    def load_translations_from_file(self, session_name: str) -> Optional[List[Dict]]:
        """Translation verilerini dosyadan yükle"""
        base_path = os.path.join(self.img_save_path, session_name, self.translations_file)
        if os.path.exists(base_path):
            with open(base_path, 'r') as f:
                translations = json.load(f)
            self.logger.info(f"Translation yüklendi: {base_path}")
            return translations
        self.logger.warning(f"{base_path} bulunamadı")
        return None
    
    def get_translations(self, force_download: bool = False, retries: int = 5, initial_wait_time: float = 0.1) -> Optional[List[Dict]]:
        """
        Translation verilerini sunucudan al (translation_x, translation_y, translation_z)
        
        Dikkat: Bir dakika içerisinde maksimum 5 adet get_translations isteği atabilirsiniz.
        """
        if not force_download:
            try:
                if os.path.exists(self.img_save_path):
                    session_name = self.get_session_name()
                    if session_name:
                        translations = self.load_translations_from_file(session_name)
                        if translations:
                            self.logger.info("Translation dosyadan yüklendi (cache)")
                            return translations
            except Exception as e:
                self.logger.info(f"Translation dosyası bozuk: {e}")
        
        payload = {}
        headers = {'Authorization': f'Token {self.auth_token}'}
        wait_time = initial_wait_time
        
        for attempt in range(retries):
            try:
                response = requests.get(self.url_translations, headers=headers, data=payload, timeout=60)
                self.translations = json.loads(response.text)
                
                if response.status_code == 200:
                    self.logger.info(f"get_translations başarılı: {len(self.translations)} translation")
                    self.save_translations_to_file(self.translations)
                    return self.translations
                else:
                    self.logger.error(f"get_translations başarısız: {response.text}")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"get_translations isteği hatası: {e}")
            
            self.logger.info(f"{wait_time} saniye sonra yeniden deneniyor...")
            time.sleep(wait_time)
            wait_time *= 2
        
        self.logger.error("get_translations başarısız, dosyadan yüklenmeye çalışılıyor")
        session_name = self.get_session_name()
        if session_name:
            return self.load_translations_from_file(session_name)
        return None
    
    def send_prediction(self, prediction, retries: int = 5, initial_wait_time: float = 0.1) -> Optional[requests.Response]:
        """
        Tahmin sonuçlarını sunucuya gönder
        
        Dikkat: Bir dakika içerisinde maksimum 80 frame için tahmin gönderebilirsiniz.
        
        Prediction nesnesi create_payload() metoduna sahip olmalıdır.
        """
        payload = json.dumps(prediction.create_payload(self.base_url))
        files = []
        headers = {
            'Authorization': f'Token {self.auth_token}',
            'Content-Type': 'application/json',
        }
        wait_time = initial_wait_time

        self._prediction_throttle.wait()

        for attempt in range(retries):
            try:
                response = requests.post(self.url_prediction, headers=headers, data=payload, files=files, timeout=60)
                
                if response.status_code == 201:
                    self.logger.info(f"Tahmin gönderildi: {prediction}")
                    return response
                elif response.status_code == 406:
                    self.logger.error("Tahmin zaten gönderilmiş - 406 Not Acceptable")
                    return response
                else:
                    self.logger.error(f"Tahmin gönderme başarısız: {response.text}")
                    response_json = json.loads(response.text)
                    if "You do not have permission to perform this action." in response_json.get("detail", ""):
                        self.logger.info("Rate limit aşıldı: 80 frames/min")
                        return response
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Tahmin gönderme hatası: {e}")
            
            self.logger.info(f"{wait_time} saniye sonra yeniden deneniyor...")
            time.sleep(wait_time)
            wait_time *= 2
        
        self.logger.error("Tahmin gönderme başarısız (çoklu deneme)")
        return None
    
    def get_translation_by_frame(self, frame_id: int) -> Optional[Dict]:
        """Frame ID'ye göre translation verisi getir"""
        if self.translations is None:
            self.translations = self.get_translations()
        
        if self.translations:
            for trans in self.translations:
                if trans.get('frame_id') == frame_id:
                    return trans
        
        return None
    
    @property
    def is_authenticated(self) -> bool:
        """Login olunmuş mu"""
        return self.auth_token is not None
    
    @property
    def has_frames(self) -> bool:
        """Frameler yüklenmiş mi"""
        return self.frames is not None
    
    @property
    def has_translations(self) -> bool:
        """Translation verileri yüklenmiş mi"""
        return self.translations is not None
