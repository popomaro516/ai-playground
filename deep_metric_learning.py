import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np

class TripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        """
        トリプレット損失関数
        
        Args:
            margin: マージン値
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        トリプレット損失を計算
        
        Args:
            anchor: アンカーサンプルの特徴量
            positive: ポジティブサンプルの特徴量
            negative: ネガティブサンプルの特徴量
            
        Returns:
            トリプレット損失
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        """
        対照損失関数
        
        Args:
            margin: マージン値
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        対照損失を計算
        
        Args:
            output1: 第1サンプルの特徴量
            output2: 第2サンプルの特徴量
            label: ペアラベル（0: 異なるクラス, 1: 同じクラス）
            
        Returns:
            対照損失
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        """
        ArcFace損失関数
        
        Args:
            in_features: 入力特徴量の次元数
            out_features: クラス数
            s: スケール値
            m: マージン値
        """
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m
    
    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        ArcFace損失を計算
        
        Args:
            input: 入力特徴量
            label: ラベル
            
        Returns:
            ArcFace損失
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return F.cross_entropy(output, label)

class MetricLearningNetwork(nn.Module):
    def __init__(self, feature_extractor, embedding_dim: int = 256, num_classes: Optional[int] = None):
        """
        深層距離学習ネットワーク
        
        Args:
            feature_extractor: 特徴抽出器
            embedding_dim: 埋め込み次元数
            num_classes: クラス数（分類タスクの場合）
        """
        super(MetricLearningNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        
        # 特徴抽出器の出力次元を取得
        input_dim = feature_extractor.get_feature_dimension()
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.L2Norm(dim=1)
        )
        
        self.classifier = None
        if num_classes:
            self.classifier = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            
        Returns:
            埋め込みベクトルと分類結果（分類器がある場合）
        """
        features = self.feature_extractor.extract_features(x)
        embeddings = self.embedding_layer(features)
        
        classification = None
        if self.classifier:
            classification = self.classifier(embeddings)
        
        return embeddings, classification

class L2Norm(nn.Module):
    def __init__(self, dim: int = 1):
        super(L2Norm, self).__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=self.dim)

def compute_accuracy(embeddings: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    """
    k-NN精度を計算
    
    Args:
        embeddings: 埋め込みベクトル
        labels: ラベル
        k: k値
        
    Returns:
        精度
    """
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    correct = 0
    total = len(labels)
    
    for i in range(total):
        # 自分自身を除く最近傍のラベルを取得
        neighbor_labels = labels[indices[i][1:k+1]]
        if labels[i] in neighbor_labels:
            correct += 1
    
    return correct / total