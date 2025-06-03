import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Tuple, Dict, Optional, Union
import os
from PIL import Image
from skimage import feature, color
from scipy import ndimage
import matplotlib.pyplot as plt

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