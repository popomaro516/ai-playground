import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from traditional_methods import SIFTMatcher, ORBMatcher, ColorHistogramMatcher, LBPMatcher, HOGMatcher

class TraditionalMethodsEvaluator:
    def __init__(self, data_directory: str):
        """
        従来手法の統合評価システム
        
        Args:
            data_directory: 評価用データディレクトリ
        """
        self.data_directory = data_directory
        self.methods = {}
        self.results = {}
        self.ground_truth = {}
        
        # 各手法を初期化
        self._initialize_methods()
        self._load_ground_truth()
    
    def _initialize_methods(self):
        """各手法を初期化"""
        self.methods = {
            'SIFT': SIFTMatcher(n_features=500),
            'ORB': ORBMatcher(n_features=500),
            'Color_Histogram': ColorHistogramMatcher(bins=(50, 60, 60)),
            'LBP': LBPMatcher(radius=3, n_points=24, method='uniform'),
            'HOG': HOGMatcher(orientations=9, pixels_per_cell=(8, 8))
        }
    
    def _load_ground_truth(self):
        """グランドトゥルースを読み込み（クラス情報から生成）"""
        self.ground_truth.clear()
        
        # ディレクトリ構造からクラス情報を取得
        class_to_images = {}
        
        for class_name in os.listdir(self.data_directory):
            class_path = os.path.join(self.data_directory, class_name)
            if not os.path.isdir(class_path):
                continue
            
            images = []
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    images.append(img_path)
            
            if images:
                class_to_images[class_name] = images
        
        # 各画像に対する正解クラスを設定
        for class_name, images in class_to_images.items():
            for img_path in images:
                # 同じクラスの他の画像が正解
                self.ground_truth[img_path] = [img for img in images if img != img_path]
    
    def build_databases(self):
        """全手法のデータベースを構築"""
        print("データベース構築中...")
        
        for method_name, method in self.methods.items():
            print(f"  {method_name}...")
            start_time = time.time()
            method.build_database(self.data_directory)
            build_time = time.time() - start_time
            print(f"    構築時間: {build_time:.2f}秒")
    
    def evaluate_method(self, method_name: str, method, test_images: List[str], 
                       top_k: int = 5) -> Dict[str, Any]:
        """
        単一手法の評価
        
        Args:
            method_name: 手法名
            method: 手法オブジェクト
            test_images: テスト画像リスト
            top_k: 検索する上位k件
            
        Returns:
            評価結果辞書
        """
        print(f"  {method_name} 評価中...")
        
        total_queries = 0
        total_recall = 0.0
        total_precision = 0.0
        query_times = []
        
        for query_img in test_images:
            if query_img not in self.ground_truth:
                continue
            
            try:
                # 検索実行
                start_time = time.time()
                results = method.find_similar_images(query_img, top_k)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # 正解画像
                relevant_images = set(self.ground_truth[query_img])
                
                # 検索結果
                retrieved_images = set([result[0] for result in results])
                
                # Precision と Recall を計算
                if retrieved_images:
                    precision = len(relevant_images & retrieved_images) / len(retrieved_images)
                    total_precision += precision
                
                if relevant_images:
                    recall = len(relevant_images & retrieved_images) / len(relevant_images)
                    total_recall += recall
                
                total_queries += 1
                
            except Exception as e:
                print(f"    エラー {query_img}: {e}")
        
        # 平均値を計算
        avg_precision = total_precision / total_queries if total_queries > 0 else 0.0
        avg_recall = total_recall / total_queries if total_queries > 0 else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        avg_query_time = np.mean(query_times) if query_times else 0.0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score,
            'avg_query_time': avg_query_time,
            'total_queries': total_queries
        }
    
    def evaluate_all_methods(self, top_k: int = 5, sample_ratio: float = 0.2) -> Dict[str, Dict[str, Any]]:
        """
        全手法の評価
        
        Args:
            top_k: 検索する上位k件
            sample_ratio: テストに使用する画像の割合
            
        Returns:
            全手法の評価結果
        """
        print("=== 従来手法の評価開始 ===")
        
        # テスト画像をサンプリング
        all_images = []
        for gt_img in self.ground_truth.keys():
            all_images.append(gt_img)
        
        # ランダムサンプリング
        np.random.seed(42)
        n_samples = max(1, int(len(all_images) * sample_ratio))
        test_images = np.random.choice(all_images, n_samples, replace=False).tolist()
        
        print(f"テスト画像数: {len(test_images)}")
        print(f"Top-{top_k} での評価")
        
        results = {}
        
        for method_name, method in self.methods.items():
            result = self.evaluate_method(method_name, method, test_images, top_k)
            results[method_name] = result
            
            print(f"  {method_name}:")
            print(f"    Precision: {result['precision']:.4f}")
            print(f"    Recall: {result['recall']:.4f}")
            print(f"    F1-Score: {result['f1_score']:.4f}")
            print(f"    平均クエリ時間: {result['avg_query_time']:.4f}秒")
        
        self.results = results
        return results
    
    def create_comparison_chart(self, save_path: str = 'method_comparison.png'):
        """
        手法比較チャートを作成
        
        Args:
            save_path: 保存パス
        """
        if not self.results:
            print("評価結果がありません。先に evaluate_all_methods() を実行してください。")
            return
        
        methods = list(self.results.keys())
        precision_scores = [self.results[method]['precision'] for method in methods]
        recall_scores = [self.results[method]['recall'] for method in methods]
        f1_scores = [self.results[method]['f1_score'] for method in methods]
        query_times = [self.results[method]['avg_query_time'] for method in methods]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision比較
        axes[0, 0].bar(methods, precision_scores, color='skyblue')
        axes[0, 0].set_title('Precision 比較')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Recall比較
        axes[0, 1].bar(methods, recall_scores, color='lightgreen')
        axes[0, 1].set_title('Recall 比較')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # F1-Score比較
        axes[1, 0].bar(methods, f1_scores, color='lightcoral')
        axes[1, 0].set_title('F1-Score 比較')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # クエリ時間比較
        axes[1, 1].bar(methods, query_times, color='lightyellow')
        axes[1, 1].set_title('平均クエリ時間 比較')
        axes[1, 1].set_ylabel('時間 (秒)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"比較チャートを保存: {save_path}")
    
    def create_detailed_report(self, save_path: str = 'detailed_evaluation_report.txt'):
        """
        詳細な評価レポートを作成
        
        Args:
            save_path: 保存パス
        """
        if not self.results:
            print("評価結果がありません。先に evaluate_all_methods() を実行してください。")
            return
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=== 従来手法による画像類似度検索 評価レポート ===\n\n")
            
            # 総合結果
            f.write("== 総合結果 ==\n")
            df_results = pd.DataFrame(self.results).T
            f.write(df_results.to_string())
            f.write("\n\n")
            
            # 手法別詳細
            f.write("== 手法別詳細 ==\n\n")
            
            for method_name, result in self.results.items():
                f.write(f"### {method_name} ###\n")
                f.write(f"Precision: {result['precision']:.4f}\n")
                f.write(f"Recall: {result['recall']:.4f}\n")
                f.write(f"F1-Score: {result['f1_score']:.4f}\n")
                f.write(f"平均クエリ時間: {result['avg_query_time']:.4f}秒\n")
                f.write(f"処理クエリ数: {result['total_queries']}\n")
                
                # 特徴量の説明
                if method_name == 'SIFT':
                    f.write("特徴: スケール不変特徴変換、キーポイント検出\n")
                elif method_name == 'ORB':
                    f.write("特徴: 高速バイナリ特徴量、リアルタイム処理\n")
                elif method_name == 'Color_Histogram':
                    f.write("特徴: 色分布情報、シンプルで高速\n")
                elif method_name == 'LBP':
                    f.write("特徴: テクスチャパターン、照明変化に頑健\n")
                elif method_name == 'HOG':
                    f.write("特徴: 勾配方向ヒストグラム、形状情報\n")
                
                f.write("\n")
            
            # ランキング
            f.write("== 性能ランキング ==\n\n")
            
            # F1-Scoreでランキング
            f1_ranking = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            f.write("F1-Score ランキング:\n")
            for i, (method, result) in enumerate(f1_ranking, 1):
                f.write(f"{i}. {method}: {result['f1_score']:.4f}\n")
            f.write("\n")
            
            # 速度ランキング
            speed_ranking = sorted(self.results.items(), key=lambda x: x[1]['avg_query_time'])
            f.write("速度ランキング (高速順):\n")
            for i, (method, result) in enumerate(speed_ranking, 1):
                f.write(f"{i}. {method}: {result['avg_query_time']:.4f}秒\n")
            f.write("\n")
            
            # 推奨用途
            f.write("== 推奨用途 ==\n\n")
            f.write("• 高精度が必要: SIFT, HOG\n")
            f.write("• 高速処理が必要: Color_Histogram, ORB\n")
            f.write("• テクスチャ重視: LBP\n")
            f.write("• 形状重視: HOG\n")
            f.write("• 色情報重視: Color_Histogram\n")
        
        print(f"詳細レポートを保存: {save_path}")
    
    def save_results_csv(self, save_path: str = 'evaluation_results.csv'):
        """
        結果をCSVファイルに保存
        
        Args:
            save_path: 保存パス
        """
        if not self.results:
            print("評価結果がありません。")
            return
        
        df = pd.DataFrame(self.results).T
        df.to_csv(save_path, encoding='utf-8')
        print(f"結果をCSVに保存: {save_path}")

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='従来手法による画像類似度検索の評価')
    parser.add_argument('--data_dir', type=str, required=True, help='データディレクトリパス')
    parser.add_argument('--top_k', type=int, default=5, help='検索する上位k件')
    parser.add_argument('--sample_ratio', type=float, default=0.2, help='テスト画像の割合')
    parser.add_argument('--output_dir', type=str, default='./traditional_evaluation_results', help='結果出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 評価実行
    evaluator = TraditionalMethodsEvaluator(args.data_dir)
    evaluator.build_databases()
    results = evaluator.evaluate_all_methods(top_k=args.top_k, sample_ratio=args.sample_ratio)
    
    # 結果保存
    chart_path = os.path.join(args.output_dir, 'method_comparison.png')
    report_path = os.path.join(args.output_dir, 'detailed_evaluation_report.txt')
    csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
    
    evaluator.create_comparison_chart(chart_path)
    evaluator.create_detailed_report(report_path)
    evaluator.save_results_csv(csv_path)
    
    print("=== 評価完了 ===")

if __name__ == "__main__":
    main()