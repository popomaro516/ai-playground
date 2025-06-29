import cv2
import numpy as np
from typing import List, Tuple
import os


class ORBMatcher:
    def __init__(self, n_features: int = 500):
        """
        ORB特徴量による画像類似度検索
        
        Args:
            n_features: 検出する特徴点の最大数
        """
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.features_db = {}
        self.image_paths = []
    
    def extract_features(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像からORB特徴量を抽出
        
        Args:
            image_path: 画像パス
            
        Returns:
            キーポイントと記述子
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        if descriptors is None:
            descriptors = np.array([]).reshape(0, 32)
        
        return keypoints, descriptors
    
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
                    keypoints, descriptors = self.extract_features(image_path)
                    self.features_db[image_path] = {
                        'keypoints': keypoints,
                        'descriptors': descriptors
                    }
                    self.image_paths.append(image_path)
                except Exception as e:
                    print(f"特徴量抽出エラー {image_path}: {e}")
    
    def find_similar_images(self, query_image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        類似画像を検索
        
        Args:
            query_image_path: クエリ画像パス
            top_k: 返す類似画像数
            
        Returns:
            類似画像のパスとスコアのリスト
        """
        query_kp, query_desc = self.extract_features(query_image_path)
        
        if query_desc.shape[0] == 0:
            return []
        
        similarities = []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        for db_path in self.image_paths:
            if db_path == query_image_path:
                continue
                
            db_desc = self.features_db[db_path]['descriptors']
            
            if db_desc.shape[0] == 0:
                similarities.append((db_path, 0.0))
                continue
            
            try:
                matches = bf.match(query_desc, db_desc)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # 距離の逆数をスコアとして使用
                if matches:
                    avg_distance = np.mean([m.distance for m in matches])
                    score = 1.0 / (1.0 + avg_distance)
                else:
                    score = 0.0
                
                similarities.append((db_path, score))
                
            except Exception:
                similarities.append((db_path, 0.0))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]