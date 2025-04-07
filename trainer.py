import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import datetime

from dino_feature_extractor import DinoV2FeatureExtractor
from deep_metric_learning import MetricLearningNetwork, TripletLoss

class MetricLearningTrainer:
    def __init__(self, 
                 model: MetricLearningNetwork,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda',
                 save_dir: str = './checkpoints'):
        """
        深層距離学習トレーナー
        
        Args:
            model: 学習モデル
            criterion: 損失関数
            optimizer: 最適化手法
            device: 使用デバイス
            save_dir: モデル保存ディレクトリ
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        
        self.model.to(device)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """1エポック訓練"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for anchor, positive, negative in tqdm(train_loader, desc="訓練中"):
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            self.optimizer.zero_grad()
            
            anchor_emb, _ = self.model(anchor)
            positive_emb, _ = self.model(positive)
            negative_emb, _ = self.model(negative)
            
            loss = self.criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 100) -> Dict[str, List[float]]:
        """
        モデル訓練
        
        Args:
            train_loader: 訓練データローダー
            val_loader: 検証データローダー
            num_epochs: エポック数
            
        Returns:
            訓練履歴
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"エポック {epoch + 1}/{num_epochs}")
            
            # 訓練
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            print(f"訓練損失: {train_loss:.4f}")
            
            # 定期保存
            if (epoch + 1) % 10 == 0:
                self.save_model(f'model_epoch_{epoch + 1}.pth', epoch, train_loss, 0.0)
        
        return {
            '訓練損失': self.train_losses,
            '検証損失': self.val_losses
        }
    
    def save_model(self, filename: str, epoch: int, train_loss: float, val_loss: float):
        """モデルを保存"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, filepath)

def create_trainer(model_name: str = 'dinov2_vitb14',
                  embedding_dim: int = 256,
                  learning_rate: float = 1e-4,
                  device: str = 'cuda') -> MetricLearningTrainer:
    """
    トレーナーを作成
    
    Args:
        model_name: DINO v2モデル名
        embedding_dim: 埋め込み次元数
        learning_rate: 学習率
        device: デバイス
        
    Returns:
        トレーナーオブジェクト
    """
    # 特徴抽出器とネットワークを作成
    feature_extractor = DinoV2FeatureExtractor(model_name, device)
    model = MetricLearningNetwork(feature_extractor, embedding_dim)
    
    # 損失関数とオプティマイザー
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return MetricLearningTrainer(model, criterion, optimizer, device)