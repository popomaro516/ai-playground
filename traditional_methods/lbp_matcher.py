import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import os
from skimage import feature


class LBPMatcher:
    def __init__(self, radius: int = 3, n_points: int = 24, method: str = 'uniform'):
        """
        LBP（Local Binary Pattern）による画像類似度検索
        
        Args:
            radius: LBPの半径
            n_points: サンプリング点数
            method: LBPの種類 ('uniform', 'default')
        """
        self.radius = radius
        self.n_points = n_points
        self.method = method
        self.features_db = {}
        self.image_paths = []
    
    def extract_lbp_features(self, image_path: str) -> np.ndarray:
        """
        画像からLBP特徴量を抽出
        
        Args:
            image_path: 画像パス
            
        Returns:
            LBPヒストグラム
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        # LBPを計算
        if self.method == 'uniform':
            lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method='uniform')
            n_bins = self.n_points + 2
        else:
            lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method='default')
            n_bins = 2 ** self.n_points
        
        # ヒストグラムを計算
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        
        # 正規化
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-8)
        
        return hist
    
    def build_database(self, image_directory: str):
        """
        画像データベースを構築
        
        Args:
            image_directory: 画像ディレクトリパス
        """
        self.features_db.clear()
        self.image_paths.clear()
        
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_directory, filename)
                try:
                    features = self.extract_lbp_features(image_path)
                    self.features_db[image_path] = features
                    self.image_paths.append(image_path)
                except Exception as e:
                    print(f"LBP特徴量抽出エラー {image_path}: {e}")
    
    def find_similar_images(self, query_image_path: str, top_k: int = 5, distance_metric: str = 'chi_square') -> List[Tuple[str, float]]:
        """
        類似画像を検索
        
        Args:
            query_image_path: クエリ画像パス
            top_k: 返す類似画像数
            distance_metric: 距離尺度 ('chi_square', 'cosine', 'euclidean')
            
        Returns:
            類似画像のパスとスコアのリスト
        """
        query_features = self.extract_lbp_features(query_image_path)
        
        similarities = []
        
        for db_path in self.image_paths:
            if db_path == query_image_path:
                continue
            
            db_features = self.features_db[db_path]
            
            if distance_metric == 'chi_square':
                chi2 = 0.5 * np.sum(((query_features - db_features) ** 2) / (query_features + db_features + 1e-8))
                score = 1.0 / (1.0 + chi2)
            elif distance_metric == 'cosine':
                score = cosine_similarity([query_features], [db_features])[0][0]
            elif distance_metric == 'euclidean':
                distance = np.linalg.norm(query_features - db_features)
                score = 1.0 / (1.0 + distance)
            else:
                score = cosine_similarity([query_features], [db_features])[0][0]
            
            similarities.append((db_path, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]