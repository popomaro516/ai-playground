import cv2
import numpy as np
from typing import List, Tuple
import os


class ColorHistogramMatcher:
    def __init__(self, bins: Tuple[int, int, int] = (50, 60, 60)):
        """
        色ヒストグラムによる画像類似度検索
        
        Args:
            bins: HSVチャンネルのビン数
        """
        self.bins = bins
        self.histograms_db = {}
        self.image_paths = []
    
    def extract_histogram(self, image_path: str) -> np.ndarray:
        """
        画像から色ヒストグラムを抽出
        
        Args:
            image_path: 画像パス
            
        Returns:
            正規化されたヒストグラム
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        # BGRからHSVに変換
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 3次元ヒストグラムを計算
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins, [0, 180, 0, 256, 0, 256])
        
        # 正規化
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    
    def build_database(self, image_directory: str):
        """
        画像データベースを構築
        
        Args:
            image_directory: 画像ディレクトリパス
        """
        self.histograms_db.clear()
        self.image_paths.clear()
        
        for filename in os.listdir(image_directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_directory, filename)
                try:
                    hist = self.extract_histogram(image_path)
                    self.histograms_db[image_path] = hist
                    self.image_paths.append(image_path)
                except Exception as e:
                    print(f"ヒストグラム抽出エラー {image_path}: {e}")
    
    def find_similar_images(self, query_image_path: str, top_k: int = 5, method: str = 'correlation') -> List[Tuple[str, float]]:
        """
        類似画像を検索
        
        Args:
            query_image_path: クエリ画像パス
            top_k: 返す類似画像数
            method: 比較方法 ('correlation', 'chi_square', 'intersection', 'bhattacharyya')
            
        Returns:
            類似画像のパスとスコアのリスト
        """
        query_hist = self.extract_histogram(query_image_path)
        
        similarities = []
        
        for db_path in self.image_paths:
            if db_path == query_image_path:
                continue
            
            db_hist = self.histograms_db[db_path]
            
            if method == 'correlation':
                score = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_CORREL)
            elif method == 'chi_square':
                score = 1.0 / (1.0 + cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_CHISQR))
            elif method == 'intersection':
                score = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_INTERSECT)
            elif method == 'bhattacharyya':
                score = 1.0 - cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_BHATTACHARYYA)
            else:
                score = cv2.compareHist(query_hist, db_hist, cv2.HISTCMP_CORREL)
            
            similarities.append((db_path, score))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]