import os
import cv2
import numpy as np
from pathlib import Path
import random
import shutil
from typing import List, Tuple
import albumentations as A

class NeuronDataAugmentor:
    def __init__(self, 
                 images_dir: str = "train/images", 
                 labels_dir: str = "train/labels",
                 output_images_dir: str = "train/images",
                 output_labels_dir: str = "train/labels"):
        
        os.chdir('..')
        
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_images_dir = Path(output_images_dir)
        self.output_labels_dir = Path(output_labels_dir)
        
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=[-90, -90, 0, 90, 180, 270], p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=0, 
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        self.photometric_transform = A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
                A.ChannelShuffle(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))

    def load_yolo_labels(self, label_path: Path) -> Tuple[List[int], List[List[float]]]:
        classes = []
        bboxes = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0])
                        bbox = [float(x) for x in data[1:5]]
                        classes.append(class_id)
                        bboxes.append(bbox)
        
        return classes, bboxes

    def save_yolo_labels(self, classes: List[int], bboxes: List[List[float]], output_path: Path):
        with open(output_path, 'w') as f:
            for class_id, bbox in zip(classes, bboxes):
                bbox = [max(0, min(1, coord)) for coord in bbox]
                f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        processed = image.copy().astype(np.float32)
        
        for i in range(processed.shape[2]):
            channel = processed[:, :, i]
            channel_min, channel_max = channel.min(), channel.max()
            if channel_max > channel_min:
                processed[:, :, i] = (channel - channel_min) / (channel_max - channel_min) * 255
        
        return processed.astype(np.uint8)

    def create_patches(self, image: np.ndarray, classes: List[int], bboxes: List[List[float]], 
                      patch_size: int = 512, overlap: float = 0.2) -> List[Tuple[np.ndarray, List[int], List[List[float]]]]:
        h, w = image.shape[:2]
        step = int(patch_size * (1 - overlap))
        patches = []
        
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                patch = image[y:y+patch_size, x:x+patch_size]
                
                patch_classes = []
                patch_bboxes = []
                
                for cls, bbox in zip(classes, bboxes):
                    abs_x = bbox[0] * w
                    abs_y = bbox[1] * h
                    abs_w = bbox[2] * w
                    abs_h = bbox[3] * h
                    
                    bbox_x1 = abs_x - abs_w/2
                    bbox_y1 = abs_y - abs_h/2
                    bbox_x2 = abs_x + abs_w/2
                    bbox_y2 = abs_y + abs_h/2
                    
                    patch_x1, patch_y1 = x, y
                    patch_x2, patch_y2 = x + patch_size, y + patch_size
                    
                    inter_x1 = max(bbox_x1, patch_x1)
                    inter_y1 = max(bbox_y1, patch_y1)
                    inter_x2 = min(bbox_x2, patch_x2)
                    inter_y2 = min(bbox_y2, patch_y2)
                    
                    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                        new_x = (abs_x - x) / patch_size
                        new_y = (abs_y - y) / patch_size
                        new_w = abs_w / patch_size
                        new_h = abs_h / patch_size
                        
                        if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                            patch_classes.append(cls)
                            patch_bboxes.append([new_x, new_y, new_w, new_h])
                
                if patch_classes:
                    patches.append((patch, patch_classes, patch_bboxes))
        
        return patches

    def augment_dataset(self, multiplier: int = 3, create_patches_flag: bool = True):
        print("Copying original images...")
        for img_path in self.images_dir.glob("*.jpg"):
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            
            shutil.copy2(img_path, self.output_images_dir / img_path.name)
            
            if label_path.exists():
                shutil.copy2(label_path, self.output_labels_dir / f"{img_path.stem}.txt")
        
        print(f"Starting data augmentation with multiplier {multiplier}...")
        
        image_files = list(self.images_dir.glob("*.jpg"))
        total_generated = 0
        
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Error loading {img_path}")
                continue
            
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            classes, bboxes = self.load_yolo_labels(label_path)
            
            image = self.preprocess_image(image)
            
            if create_patches_flag and min(image.shape[:2]) > 512:
                patches = self.create_patches(image, classes, bboxes)
                for i, (patch, patch_classes, patch_bboxes) in enumerate(patches):
                    patch_name = f"{img_path.stem}_patch_{i}"
                    cv2.imwrite(str(self.output_images_dir / f"{patch_name}.jpg"), patch)
                    self.save_yolo_labels(patch_classes, patch_bboxes, 
                                        self.output_labels_dir / f"{patch_name}.txt")
                    total_generated += 1
            
            for i in range(multiplier):
                try:
                    transformed = self.transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=classes
                    )
                    
                    aug_name = f"{img_path.stem}_aug_{i}"
                    cv2.imwrite(str(self.output_images_dir / f"{aug_name}.jpg"), 
                               transformed['image'])
                    
                    if transformed['bboxes']:
                        self.save_yolo_labels(
                            transformed['class_labels'],
                            transformed['bboxes'],
                            self.output_labels_dir / f"{aug_name}.txt"
                        )
                    
                    total_generated += 1
                    
                    if random.random() < 0.5:
                        photo_transformed = self.photometric_transform(
                            image=transformed['image'],
                            bboxes=transformed['bboxes'],
                            class_labels=transformed['class_labels']
                        )
                        
                        photo_name = f"{img_path.stem}_photo_{i}"
                        cv2.imwrite(str(self.output_images_dir / f"{photo_name}.jpg"), 
                                   photo_transformed['image'])
                        
                        if photo_transformed['bboxes']:
                            self.save_yolo_labels(
                                photo_transformed['class_labels'],
                                photo_transformed['bboxes'],
                                self.output_labels_dir / f"{photo_name}.txt"
                            )
                        
                        total_generated += 1
                        
                except Exception as e:
                    print(f"Error in augmentation of {img_path.name}, iteration {i}: {str(e)}")
                    continue
        
        print(f"Data augmentation completed!")
        print(f"Original images: {len(image_files)}")
        print(f"Total images generated: {total_generated}")
        print(f"Real multiplication factor: {total_generated / len(image_files):.1f}x")

    def verify_dataset(self):
        images = list(self.output_images_dir.glob("*.jpg"))
        labels = list(self.output_labels_dir.glob("*.txt"))
        
        print(f"Images generated: {len(images)}")
        print(f"Labels generated: {len(labels)}")
        
        missing_labels = []
        for img_path in images:
            label_path = self.output_labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(img_path.name)
        
        if missing_labels:
            print(f"Warning: {len(missing_labels)} images without labels")
            for name in missing_labels[:5]:
                print(f"  - {name}")
        else:
            print("✓ All images have corresponding labels")

if __name__ == "__main__":
    augmentor = NeuronDataAugmentor(
        images_dir="train/images",
        labels_dir="train/labels",
        output_images_dir="train/augmented_images",
        output_labels_dir="train/augmented_labels"
    )
    
    augmentor.augment_dataset(multiplier=4, create_patches_flag=True)
    
    augmentor.verify_dataset()
    
    print("\nData augmentation completed!")
    print("Required dependencies:")
    print("pip install albumentations opencv-python numpy")