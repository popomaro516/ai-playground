import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Union, List, Optional
import logging

class DinoV2FeatureExtractor:
    def __init__(self, model_name: str = 'dinov2_vitb14', device: Optional[str] = None):
        """
        DINO v2を使用した特徴抽出器
        
        Args:
            model_name: 使用するDINO v2モデル名
            device: 使用するデバイス（CPUまたはGPU）
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.model = self._load_model()
        self.transform = self._get_transform()
        
    def _load_model(self) -> nn.Module:
        """DINO v2モデルを読み込む"""
        try:
            model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            model.eval()
            model.to(self.device)
            return model
        except Exception as e:
            logging.error(f"モデルの読み込みに失敗: {e}")
            raise
    
    def _get_transform(self) -> transforms.Compose:
        """画像の前処理用変換を取得"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        画像から特徴量を抽出
        
        Args:
            images: PIL画像、PIL画像のリスト、またはテンソル
            
        Returns:
            抽出された特徴量テンソル
        """
        with torch.no_grad():
            if isinstance(images, Image.Image):
                images = [images]
            
            if isinstance(images, list):
                # PIL画像のリストの場合
                batch = torch.stack([self.transform(img) for img in images])
            else:
                # テンソルの場合
                batch = images
            
            batch = batch.to(self.device)
            features = self.model(batch)
            
            return features
    
    def extract_patch_features(self, images: Union[Image.Image, List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        パッチレベルの特徴量を抽出
        
        Args:
            images: PIL画像、PIL画像のリスト、またはテンソル
            
        Returns:
            パッチレベルの特徴量テンソル
        """
        with torch.no_grad():
            if isinstance(images, Image.Image):
                images = [images]
            
            if isinstance(images, list):
                batch = torch.stack([self.transform(img) for img in images])
            else:
                batch = images
            
            batch = batch.to(self.device)
            
            # パッチレベルの特徴量を取得
            features = self.model.forward_features(batch)
            patch_features = features['x_norm_patchtokens']
            
            return patch_features
    
    def get_feature_dimension(self) -> int:
        """特徴量の次元数を取得"""
        if 'vitb14' in self.model_name:
            return 768
        elif 'vitl14' in self.model_name:
            return 1024
        elif 'vits14' in self.model_name:
            return 384
        else:
            return 768  # デフォルト