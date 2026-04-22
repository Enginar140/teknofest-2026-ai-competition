"""
Veri Artırma (Augmentation) Modülü - Albumentations
"""
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """Veri artırma pipeline sınıfı"""
    
    def __init__(self, mode='train', target_size=(640, 640)):
        """
        Augmentation pipeline'ı başlat
        
        Args:
            mode: 'train', 'val', veya 'test'
            target_size: Hedef görüntü boyutu (width, height)
        """
        self.mode = mode
        self.target_size = target_size
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Augmentation transformlarını oluştur"""
        if self.mode == 'train':
            # Eğitim için agresif augmentasyon
            return A.Compose([
                # Geometrik transformlar
                A.RandomResizedCrop(
                    height=self.target_size[1],
                    width=self.target_size[0],
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ),
                
                # Renk ve aydınlatma
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3,
                        contrast_limit=0.3,
                        p=1.0
                    ),
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ], p=0.5),
                
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20,
                        p=1.0
                    ),
                ], p=0.3),
                
                # Gürültü ve blur
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.3),
                
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    A.MedianBlur(blur_limit=(3, 7), p=1.0),
                ], p=0.2),
                
                # Hava durumu efektleri
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=1.0),
                    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=200, p=1.0),
                    A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, p=1.0),
                ], p=0.1),
                
                # Kalite azaltma (jpeg artifact simülasyonu)
                A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
                
                # Normalize
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3,
                filter_bboxes_by_visibility=True
            ))
        
        elif self.mode == 'val':
            # Doğrulama için light augmentasyon
            return A.Compose([
                A.Resize(height=self.target_size[1], width=self.target_size[0]),
                A.HorizontalFlip(p=0.0),  # Hedef tespitinde flip kullanmıyoruz
                
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
            ))
        
        else:  # test
            # Test için minimal işleme
            return A.Compose([
                A.Resize(height=self.target_size[1], width=self.target_size[0]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
            ])
    
    def __call__(self, image, bboxes=None, class_labels=None):
        """
        Augmentation uygula
        
        Args:
            image: Görüntü (H, W, C)
            bboxes: YOLO formatında bounding box listesi [[x_center, y_center, width, height], ...]
            class_labels: Sınıf etiketleri
            
        Returns:
            Augmented görüntü ve bounding box'lar
        """
        if bboxes is not None and class_labels is not None:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return transformed['image'], transformed['bboxes']
        else:
            transformed = self.transform(image=image)
            return transformed['image'], None


class TestTimeAugmentation:
    """
    Test Time Augmentation (TTA)
    Inference sırasında multiple augmentasyon uygula ve sonuçları average et
    """
    
    def __init__(self, tta_transforms=None):
        """
        TTA pipeline'ı başlat
        
        Args:
            tta_transforms: Kullanılacak TTA transformları
        """
        if tta_transforms is None:
            # Varsayılan TTA transformları
            self.tta_transforms = [
                A.Compose([A.NoResize()]),  # Orijinal
                A.Compose([A.HorizontalFlip(p=1.0)]),  # Yatay flip
                A.Compose([A.VerticalFlip(p=1.0)]),  # Dikey flip
                A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),  # Her ikisi
            ]
        else:
            self.tta_transforms = tta_transforms
    
    def __call__(self, image):
        """
        TTA uygula
        
        Args:
            image: Görüntü
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        for transform in self.tta_transforms:
            transformed = transform(image=image)
            augmented_images.append(transformed['image'])
        
        return augmented_images
    
    def reverse_boxes(self, boxes, transform_idx, img_width, img_height):
        """
        Transform edilmiş bounding box'ları orijinal koordinatlara çevir
        
        Args:
            boxes: Transform edilmiş bounding box'lar
            transform_idx: Transform indeksi
            img_width: Görüntü genişliği
            img_height: Görüntü yüksekliği
            
        Returns:
            Orijinal koordinatlarda bounding box'lar
        """
        if transform_idx == 0:
            return boxes  # Orijinal, değişiklik yok
        elif transform_idx == 1:
            # Horizontal flip - x koordinatlarını ters çevir
            reversed_boxes = boxes.copy()
            reversed_boxes[:, 0] = img_width - boxes[:, 0]  # x_center
            return reversed_boxes
        elif transform_idx == 2:
            # Vertical flip - y koordinatlarını ters çevir
            reversed_boxes = boxes.copy()
            reversed_boxes[:, 1] = img_height - boxes[:, 1]  # y_center
            return reversed_boxes
        elif transform_idx == 3:
            # Her iki flip
            reversed_boxes = boxes.copy()
            reversed_boxes[:, 0] = img_width - boxes[:, 0]
            reversed_boxes[:, 1] = img_height - boxes[:, 1]
            return reversed_boxes
        else:
            return boxes


class MixUp:
    """
    MixUp veri artırma tekniği
    İki görüntüyü ve etiketlerini lineer kombinasyonla birleştirir
    """
    
    def __init__(self, alpha=0.2):
        """
        MixUp başlat
        
        Args:
            alpha: Beta dağılımı parametresi
        """
        self.alpha = alpha
    
    def __call__(self, image1, boxes1, labels1, image2, boxes2, labels2):
        """
        MixUp uygula
        
        Args:
            image1: Birinci görüntü
            boxes1: Birinci görüntünün bounding box'ları
            labels1: Birinci görüntünün etiketleri
            image2: İkinci görüntü
            boxes2: İkinci görüntünün bounding box'ları
            labels2: İkinci görüntünün etiketleri
            
        Returns:
            Mixed görüntü, boxes ve labels
        """
        # Lambda parametresi Beta dağılımından
        from scipy.stats import beta
        lam = beta(self.alpha, self.alpha).rvs()
        
        # Görüntüleri mix et
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # Box'ları ve label'ları birleştir
        mixed_boxes = np.vstack([boxes1, boxes2])
        mixed_labels = np.concatenate([labels1, labels2])
        
        return mixed_image, mixed_boxes, mixed_labels, lam


class MosaicAugmentation:
    """
    Mosaic veri artırma tekniği
    4 görüntüyü birleştirerek yeni bir görüntü oluşturur
    YOLO serisinde yaygın olarak kullanılır
    """
    
    def __init__(self, target_size=(640, 640)):
        """
        Mosaic augmentation başlat
        
        Args:
            target_size: Hedef görüntü boyutu
        """
        self.target_size = target_size
    
    def __call__(self, images, boxes_list, labels_list):
        """
        Mosaic uygula
        
        Args:
            images: 4 görüntü listesi
            boxes_list: Her görüntü için bounding box listesi
            labels_list: Her görüntü için etiket listesi
            
        Returns:
            Mosaic görüntüsü, birleştirilmiş box'lar ve label'lar
        """
        assert len(images) == 4, "Mosaic için tam 4 görüntü gerekli"
        
        # Mosaic canvas oluştur
        mosaic_img = np.zeros(
            (self.target_size[1] * 2, self.target_size[0] * 2, 3),
            dtype=np.uint8
        )
        
        # Her görüntüyü canvas'ın bir köşesine yerleştir
        centers = [
            (self.target_size[0] // 2, self.target_size[1] // 2),  # Sol üst
            (self.target_size[0] + self.target_size[0] // 2, self.target_size[1] // 2),  # Sağ üst
            (self.target_size[0] // 2, self.target_size[1] + self.target_size[1] // 2),  # Sol alt
            (self.target_size[0] + self.target_size[0] // 2, self.target_size[1] + self.target_size[1] // 2),  # Sağ alt
        ]
        
        all_boxes = []
        all_labels = []
        
        for i, (img, boxes, labels) in enumerate(zip(images, boxes_list, labels_list)):
            h, w = img.shape[:2]
            
            # Resize
            img_resized = cv2.resize(img, self.target_size)
            
            # Offset hesapla
            x_offset = centers[i][0] - self.target_size[0] // 2
            y_offset = centers[i][1] - self.target_size[1] // 2
            
            # Canvas'a yerleştir
            y1, y2 = y_offset, y_offset + self.target_size[1]
            x1, x2 = x_offset, x_offset + self.target_size[0]
            mosaic_img[y1:y2, x1:x2] = img_resized
            
            # Box'ları offset'e göre güncelle
            if len(boxes) > 0:
                boxes_updated = boxes.copy()
                boxes_updated[:, 0] = (boxes[:, 0] * w + x_offset) / (self.target_size[0] * 2)
                boxes_updated[:, 1] = (boxes[:, 1] * h + y_offset) / (self.target_size[1] * 2)
                boxes_updated[:, 2] = (boxes[:, 2] * w) / (self.target_size[0] * 2)
                boxes_updated[:, 3] = (boxes[:, 3] * h) / (self.target_size[1] * 2)
                
                all_boxes.append(boxes_updated)
                all_labels.extend(labels)
        
        # Crop to target size (rastgele offset ile)
        crop_x = np.random.randint(0, self.target_size[0])
        crop_y = np.random.randint(0, self.target_size[1])
        
        mosaic_img = mosaic_img[
            crop_y:crop_y + self.target_size[1],
            crop_x:crop_x + self.target_size[0]
        ]
        
        # Box'ları crop'a göre güncelle
        if len(all_boxes) > 0:
            all_boxes = np.vstack(all_boxes)
            all_boxes[:, 0] = (all_boxes[:, 0] * (self.target_size[0] * 2) - crop_x) / self.target_size[0]
            all_boxes[:, 1] = (all_boxes[:, 1] * (self.target_size[1] * 2) - crop_y) / self.target_size[1]
            all_boxes[:, 2] = (all_boxes[:, 2] * (self.target_size[0] * 2)) / self.target_size[0]
            all_boxes[:, 3] = (all_boxes[:, 3] * (self.target_size[1] * 2)) / self.target_size[1]
            
            # Clip to [0, 1]
            all_boxes = np.clip(all_boxes, 0, 1)
            
            # Remove boxes outside image
            valid_mask = (
                (all_boxes[:, 0] + all_boxes[:, 2] / 2 > 0) &
                (all_boxes[:, 0] - all_boxes[:, 2] / 2 < 1) &
                (all_boxes[:, 1] + all_boxes[:, 3] / 2 > 0) &
                (all_boxes[:, 1] - all_boxes[:, 3] / 2 < 1)
            )
            all_boxes = all_boxes[valid_mask]
            all_labels = np.array(all_labels)[valid_mask]
        else:
            all_boxes = np.zeros((0, 4))
            all_labels = np.array([])
        
        return mosaic_img, all_boxes, all_labels


def get_training_augmentation(target_size=(640, 640)):
    """Eğitim için augmentation pipeline'ı döndür"""
    return AugmentationPipeline(mode='train', target_size=target_size)


def get_validation_augmentation(target_size=(640, 640)):
    """Doğrulama için augmentation pipeline'ı döndür"""
    return AugmentationPipeline(mode='val', target_size=target_size)


def get_test_augmentation(target_size=(640, 640)):
    """Test için augmentation pipeline'ı döndür"""
    return AugmentationPipeline(mode='test', target_size=target_size)
