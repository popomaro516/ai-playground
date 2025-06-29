import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import os
from skimage import feature


class HOGMatcher:
    def __init__(self, orientations: int = 9, pixels_per_cell: Tuple[int, int] = (8, 8), 
                 cells_per_block: Tuple[int, int] = (2, 2), block_norm: str = 'L2-Hys'):
        """
        HOG（Histogram of Oriented Gradients）による画像類似度検索
        
        Args:
            orientations: 勾配方向のビン数
            pixels_per_cell: セルあたりのピクセル数
            cells_per_block: ブロックあたりのセル数
            block_norm: ブロック正規化方法
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.features_db = {}
        self.image_paths = []
    
    def extract_hog_features(self, image_path: str, resize_shape: Tuple[int, int] = (128, 128)) -> np.ndarray:
        """
        画像からHOG特徴量を抽出
        
        Args:
            image_path: 画像パス
            resize_shape: リサイズ後の画像サイズ
            
        Returns:
            HOG特徴量ベクトル
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        # 画像をリサイズ
        image = cv2.resize(image, resize_shape)
        
        # HOG特徴量を抽出
        hog_features = feature.hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            visualize=False,
            feature_vector=True
        )
        
        return hog_features
    
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
                    features = self.extract_hog_features(image_path)
                    self.features_db[image_path] = features
                    self.image_paths.append(image_path)
                except Exception as e:
                    print(f"HOG特徴量抽出エラー {image_path}: {e}")
    
    def find_similar_images(self, query_image_path: str, top_k: int = 5, distance_metric: str = 'cosine') -> List[Tuple[str, float]]:
        """
        類似画像を検索
        
        Args:
            query_image_path: クエリ画像パス
            top_k: 返す類似画像数
            distance_metric: 距離尺度 ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            類似画像のパスとスコアのリスト
        """
        query_features = self.extract_hog_features(query_image_path)
        
        similarities = []
        
        for db_path in self.image_paths:
            if db_path == query_image_path:
                continue
            
            db_features = self.features_db[db_path]
            
            if distance_metric == 'cosine':
                score = cosine_similarity([query_features], [db_features])[0][0]
            elif distance_metric == 'euclidean':
                distance = np.linalg.norm(query_features - db_features)
                score = 1.0 / (1.0 + distance)
            elif distance_metric == 'manhattan':
                distance = np.sum(np.abs(query_features - db_features))
                score = 1.0 / (1.0 + distance)
            else:
                score = cosine_similarity([query_features], [db_features])[0][0]
            
            similarities.append((db_path, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]