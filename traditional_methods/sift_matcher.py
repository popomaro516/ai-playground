import cv2
import numpy as np
from typing import List, Tuple
import os


class SIFTMatcher:
    def __init__(self, n_features: int = 500, n_octave_layers: int = 3, contrast_threshold: float = 0.04):
        """
        SIFT特徴量による画像類似度検索
        
        Args:
            n_features: 検出する特徴点の最大数
            n_octave_layers: オクターブあたりのレイヤー数
            contrast_threshold: コントラスト閾値
        """
        self.sift = cv2.SIFT_create(
            nfeatures=n_features, 
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold
        )
        self.features_db = {}
        self.image_paths = []
        
    def extract_features(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像からSIFT特徴量を抽出
        
        Args:
            image_path: 画像パス
            
        Returns:
            キーポイントと記述子
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        
        if descriptors is None:
            descriptors = np.array([]).reshape(0, 128)
        
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
    
    def find_similar_images(self, query_image_path: str, top_k: int = 5, ratio_threshold: float = 0.75) -> List[Tuple[str, float]]:
        """
        類似画像を検索
        
        Args:
            query_image_path: クエリ画像パス
            top_k: 返す類似画像数
            ratio_threshold: Lowe's ratio test の閾値
            
        Returns:
            類似画像のパスとスコアのリスト
        """
        query_kp, query_desc = self.extract_features(query_image_path)
        
        if query_desc.shape[0] == 0:
            return []
        
        similarities = []
        
        # FLANN matcher for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        for db_path in self.image_paths:
            if db_path == query_image_path:
                continue
                
            db_desc = self.features_db[db_path]['descriptors']
            
            if db_desc.shape[0] < 2:
                similarities.append((db_path, 0.0))
                continue
            
            try:
                matches = flann.knnMatch(query_desc, db_desc, k=2)
                
                # Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_threshold * n.distance:
                            good_matches.append(m)
                
                # マッチング数を正規化
                score = len(good_matches) / max(len(query_desc), len(db_desc))
                similarities.append((db_path, score))
                
            except Exception:
                similarities.append((db_path, 0.0))
        
        # スコア順にソート
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]