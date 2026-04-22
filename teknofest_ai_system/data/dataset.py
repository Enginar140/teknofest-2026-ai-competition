"""
Dataset ve Dataloader Sınıfları
PyTorch tabanlı veri yükleme ve batch işleme
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict, Optional, Callable
import albumentations as A

from .augmentation import AugmentationPipeline, MixUp, MosaicAugmentation

logger = logging.getLogger(__name__)


class TeknofestDataset(Dataset):
    """
    Teknofest OTR yarışması için Dataset sınıfı
    
    YOLO formatındaki annotasyonları okur ve augmentasyon uygular
    """
    
    # Sınıf isimleri (Teknofest 2026 OTR spesifikasyonuna göre)
    CLASS_NAMES = [
        'airplane_blue',      # Mavi uçak
        'airplane_red',       # Kırmızı uçak
        'airplane_yellow',    # Sarı uçak
        'airplane_unknown',   # Bilinmeyen renk uçak
        'helicopter',         # Helikopter
        'uav',                # İHA
        'person',             # Personel
        'vehicle',            # Araç
        'ship',               # Gemi
        'building',           # Bina
        'tower',              # Kule
        'bridge',             # Köprü
        'runway',             # Pist çizgisi
        'hangar',             # Hangar
    ]
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[A.Compose] = None,
        use_mosaic: float = 0.5,
        use_mixup: float = 0.15,
        target_size: Tuple[int, int] = (640, 640),
    ):
        """
        Dataset'i başlat
        
        Args:
            images_dir: Görüntülerin olduğu dizin
            labels_dir: Annotation dosyalarının olduğu dizin
            transform: Augmentation pipeline
            use_mosaic: Mosaic kullanma olasılığı
            use_mixup: MixUp kullanma olasılığı
            target_size: Hedef görüntü boyutu
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.use_mosaic = use_mosaic
        self.use_mixup = use_mixup
        self.target_size = target_size
        
        # Görüntü listesini al
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) +
                                  list(self.images_dir.glob("*.png")) +
                                  list(self.images_dir.glob("*.jpeg")))
        
        if len(self.image_files) == 0:
            logger.warning(f"Hiç görüntü bulunamadı: {images_dir}")
        
        logger.info(f"Dataset yüklendi: {len(self.image_files)} görüntü")
        
        # Mosaic için rastgele indeks oluşturucu
        self.rng = np.random.RandomState(42)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def _load_label(self, label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        YOLO formatındaki label dosyasını oku
        
        Args:
            label_path: Label dosyası yolu
            
        Returns:
            (boxes, labels) - boxes: (N, 4) YOLO format [x_center, y_center, width, height]
                              labels: (N,) sınıf indeksleri
        """
        if not label_path.exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    boxes.append([x_center, y_center, width, height])
                    labels.append(class_id)
        
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Görüntüyü yükle"""
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Görüntü yüklenemedi: {image_path}")
            # Siyah görüntü döndür
            image = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _apply_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mosaic augmentasyon uygula
        
        Args:
            idx: Mevcut indeks
            
        Returns:
            (image, boxes, labels)
        """
        # 3 rastgele indeks seç
        indices = [idx] + list(self.rng.choice(len(self), 3, replace=False))
        
        images = []
        boxes_list = []
        labels_list = []
        
        for i in indices:
            image_path = self.image_files[i]
            label_path = self.labels_dir / (image_path.stem + ".txt")
            
            img = self._load_image(image_path)
            boxes, labels = self._load_label(label_path)
            
            images.append(img)
            boxes_list.append(boxes)
            labels_list.append(labels)
        
        # Mosaic uygula
        mosaic = MosaicAugmentation(target_size=self.target_size)
        image, boxes, labels = mosaic(images, boxes_list, labels_list)
        
        return image, boxes, labels
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Veri ögesini getir
        
        Returns:
            Dict with keys:
                - image: (3, H, W) torch tensor
                - boxes: (N, 4) bounding box'lar (YOLO format)
                - labels: (N,) sınıf etiketleri
                - image_id: görüntü ID'si
                - image_path: görüntü yolu (string)
        """
        # Mosaic veya normal yükleme
        if self.transform is not None and np.random.random() < self.use_mosaic:
            image, boxes, labels = self._apply_mosaic(idx)
        else:
            image_path = self.image_files[idx]
            label_path = self.labels_dir / (image_path.stem + ".txt")
            
            image = self._load_image(image_path)
            boxes, labels = self._load_label(label_path)
        
        # Transform uygula
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes if len(boxes) > 0 else None,
                class_labels=labels if len(labels) > 0 else None
            )
            image = transformed['image']
            
            if transformed['bboxes'] is not None and len(transformed['bboxes']) > 0:
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['class_labels'], dtype=np.int64)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
        
        # Tensor'a çevir
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                # (H, W, C) -> (C, H, W)
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            else:
                image = torch.from_numpy(image).float()
        
        boxes = torch.from_numpy(boxes).float()
        labels = torch.from_numpy(labels).long()
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': idx,
            'image_path': str(self.image_files[idx]),
        }


class YOLODataset(Dataset):
    """
    YOLO formatında dataset (klasör yapısı: images/, labels/)
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (640, 640),
    ):
        """
        YOLO dataset başlat
        
        Args:
            data_root: Veri kök dizini
            split: 'train', 'val', veya 'test'
            transform: Augmentation pipeline
            target_size: Hedef görüntü boyutu
        """
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # Klasör yapısını kontrol et
        self.images_dir = self.data_root / split / 'images'
        self.labels_dir = self.data_root / split / 'labels'
        
        # Alternatif yapı (images/train, labels/train)
        if not self.images_dir.exists():
            self.images_dir = self.data_root / 'images' / split
            self.labels_dir = self.data_root / 'labels' / split
        
        # Görüntü listesini al
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) +
                                  list(self.images_dir.glob("*.png")) +
                                  list(self.images_dir.glob("*.jpeg")))
        
        logger.info(f"YOLO {split} dataset: {len(self.image_files)} görüntü")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_files[idx]
        label_path = self.labels_dir / (image_path.stem + ".txt")
        
        # Görüntüyü yükle
        image = cv2.imread(str(image_path))
        if image is None:
            image = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Label'ı yükle
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        boxes.append([x_center, y_center, width, height])
                        labels.append(class_id)
        
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        
        # Transform uygula
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                bboxes=boxes if len(boxes) > 0 else None,
                class_labels=labels if len(labels) > 0 else None
            )
            image = transformed['image']
            
            if transformed['bboxes'] is not None and len(transformed['bboxes']) > 0:
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['class_labels'], dtype=np.int64)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
        
        # Tensor'a çevir
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            else:
                image = torch.from_numpy(image).float()
        
        boxes = torch.from_numpy(boxes).float()
        labels = torch.from_numpy(labels).long()
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': idx,
            'image_path': str(image_path),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Batch collate fonksiyonu
    
    Farklı sayıda bounding box olan örnekleri bir batch'te birleştirir
    """
    images = torch.stack([item['image'] for item in batch])
    
    # Box'ları ve label'ları list olarak tut (farklı sayıda oldukları için)
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    image_ids = torch.tensor([item['image_id'] for item in batch])
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids,
        'image_paths': image_paths,
    }


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (640, 640),
    use_mosaic: float = 0.5,
    use_mixup: float = 0.15,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Eğitim, doğrulama ve test dataloader'larını oluştur
    
    Args:
        data_root: Veri kök dizini
        batch_size: Batch boyutu
        num_workers: Worker sayısı
        target_size: Hedef görüntü boyutu
        use_mosaic: Mosaic kullanma olasılığı
        use_mixup: MixUp kullanma olasılığı
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_root = Path(data_root)
    
    # Transformları oluştur
    train_transform = AugmentationPipeline(mode='train', target_size=target_size)
    val_transform = AugmentationPipeline(mode='val', target_size=target_size)
    test_transform = AugmentationPipeline(mode='test', target_size=target_size)
    
    # Dataset'leri oluştur
    train_images = data_root / 'train' / 'images'
    train_labels = data_root / 'train' / 'labels'
    
    if train_images.exists():
        train_dataset = TeknofestDataset(
            images_dir=str(train_images),
            labels_dir=str(train_labels),
            transform=train_transform.transform,
            use_mosaic=use_mosaic,
            use_mixup=use_mixup,
            target_size=target_size,
        )
    else:
        train_dataset = YOLODataset(
            data_root=str(data_root),
            split='train',
            transform=train_transform.transform,
            target_size=target_size,
        )
    
    val_images = data_root / 'val' / 'images'
    val_labels = data_root / 'val' / 'labels'
    
    if val_images.exists():
        val_dataset = TeknofestDataset(
            images_dir=str(val_images),
            labels_dir=str(val_labels),
            transform=val_transform.transform,
            use_mosaic=0.0,
            use_mixup=0.0,
            target_size=target_size,
        )
    else:
        val_dataset = YOLODataset(
            data_root=str(data_root),
            split='val',
            transform=val_transform.transform,
            target_size=target_size,
        )
    
    # Test dataset (opsiyonel)
    test_images = data_root / 'test' / 'images'
    test_labels = data_root / 'test' / 'labels'
    
    if test_images.exists():
        if test_images.exists():
            test_dataset = TeknofestDataset(
                images_dir=str(test_images),
                labels_dir=str(test_labels),
                transform=test_transform.transform,
                use_mosaic=0.0,
                use_mixup=0.0,
                target_size=target_size,
            )
        else:
            test_dataset = YOLODataset(
                data_root=str(data_root),
                split='test',
                transform=test_transform.transform,
                target_size=target_size,
            )
    else:
        test_dataset = None
    
    # Dataloader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
    
    logger.info(f"Dataloader'lar oluşturuldu:")
    logger.info(f"  Train: {len(train_dataset)} görüntü, {len(train_loader)} batch")
    logger.info(f"  Val: {len(val_dataset)} görüntü, {len(val_loader)} batch")
    if test_loader is not None:
        logger.info(f"  Test: {len(test_dataset)} görüntü, {len(test_loader)} batch")
    
    return train_loader, val_loader, test_loader


class VideoDataset(Dataset):
    """
    Video işleme için Dataset sınıfı
    Kare kare okuma yerine frame caching kullanır
    """
    
    def __init__(
        self,
        video_path: str,
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (640, 640),
        cache_size: int = 100,
    ):
        """
        Video dataset başlat
        
        Args:
            video_path: Video dosya yolu
            transform: Augmentation pipeline
            target_size: Hedef görüntü boyutu
            cache_size: Cache boyutu (frame sayısı)
        """
        self.video_path = Path(video_path)
        self.transform = transform
        self.target_size = target_size
        self.cache_size = cache_size
        
        # Video aç
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        # Frame sayısını al
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Cache
        self._cache = {}
        
        logger.info(f"Video dataset: {video_path}, {self.total_frames} frames @ {self.fps:.2f} FPS")
    
    def __len__(self) -> int:
        return self.total_frames
    
    def _get_frame(self, idx: int) -> np.ndarray:
        """Frame'i cache'ten al veya oku"""
        if idx in self._cache:
            return self._cache[idx]
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        
        if not ret:
            frame = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Cache'e ekle
        if len(self._cache) >= self.cache_size:
            # En eski frame'i sil
            oldest_key = min(self._cache.keys())
            del self._cache[oldest_key]
        
        self._cache[idx] = frame
        return frame
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = self._get_frame(idx)
        
        # Transform uygula
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Tensor'a çevir
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            else:
                image = torch.from_numpy(image).float()
        
        return {
            'image': image,
            'frame_id': idx,
            'video_path': str(self.video_path),
        }
    
    def __del__(self):
        """Videoyu kapat"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
