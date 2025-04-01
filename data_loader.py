import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import random
from typing import List, Tuple, Dict, Optional
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        """
        トリプレット学習用データセット
        
        Args:
            data_dir: データディレクトリパス
            transform: 画像変換
        """
        self.data_dir = data_dir
        self.transform = transform or self._get_default_transform()
        
        self.class_to_images = self._load_data()
        self.classes = list(self.class_to_images.keys())
        self.all_images = []
        
        for class_name, images in self.class_to_images.items():
            for img_path in images:
                self.all_images.append((img_path, class_name))
    
    def _get_default_transform(self) -> transforms.Compose:
        """デフォルトの画像変換を取得"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_data(self) -> Dict[str, List[str]]:
        """データを読み込み、クラス別に整理"""
        class_to_images = {}
        
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            images = []
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images.append(os.path.join(class_path, img_name))
            
            if images:
                class_to_images[class_name] = images
        
        return class_to_images
    
    def __len__(self) -> int:
        return len(self.all_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """トリプレット（アンカー、ポジティブ、ネガティブ）を取得"""
        anchor_path, anchor_class = self.all_images[idx]
        
        # ポジティブサンプルを選択（同じクラスから）
        positive_candidates = [img for img in self.class_to_images[anchor_class] if img != anchor_path]
        if not positive_candidates:
            positive_path = anchor_path
        else:
            positive_path = random.choice(positive_candidates)
        
        # ネガティブサンプルを選択（異なるクラスから）
        negative_classes = [cls for cls in self.classes if cls != anchor_class]
        negative_class = random.choice(negative_classes)
        negative_path = random.choice(self.class_to_images[negative_class])
        
        # 画像を読み込み、変換
        anchor = self._load_image(anchor_path)
        positive = self._load_image(positive_path)
        negative = self._load_image(negative_path)
        
        return anchor, positive, negative
    
    def _load_image(self, path: str) -> torch.Tensor:
        """画像を読み込み、変換"""
        image = Image.open(path).convert('RGB')
        return self.transform(image)

def create_data_loaders(data_dir: str, batch_size: int = 32, 
                       dataset_type: str = 'triplet', split_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    データローダーを作成
    
    Args:
        data_dir: データディレクトリパス
        batch_size: バッチサイズ
        dataset_type: データセットタイプ（'triplet', 'pair', 'classification'）
        split_ratio: 訓練データの割合
        
    Returns:
        訓練用データローダーと検証用データローダー
    """
    dataset = TripletDataset(data_dir)
    
    # データを訓練用と検証用に分割
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader