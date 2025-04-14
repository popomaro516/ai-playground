#!/usr/bin/env python3
import argparse
import os
import torch
from data_loader import create_data_loaders
from trainer import create_trainer

def train_model(args):
    """モデル訓練"""
    print("=== 深層距離学習モデルの訓練開始 ===")
    
    # データローダー作成
    train_loader, val_loader = create_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        dataset_type='triplet',
        split_ratio=0.8
    )
    
    print(f"訓練データ数: {len(train_loader.dataset)}")
    print(f"検証データ数: {len(val_loader.dataset)}")
    
    # トレーナー作成
    trainer = create_trainer(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # 訓練実行
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs
    )
    
    print("=== 訓練完了 ===")
    return history

def main():
    parser = argparse.ArgumentParser(description='DINO v2を使用した深層距離学習')
    
    parser.add_argument('--data_dir', type=str, required=True, help='データディレクトリパス')
    parser.add_argument('--model_name', type=str, default='dinov2_vitb14', help='DINO v2モデル名')
    parser.add_argument('--embedding_dim', type=int, default=256, help='埋め込み次元数')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=100, help='エポック数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学習率')
    parser.add_argument('--device', type=str, default='cuda', help='使用デバイス')
    
    args = parser.parse_args()
    
    # デバイス設定
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA が利用できません。CPUを使用します。")
        args.device = 'cpu'
    
    print(f"使用デバイス: {args.device}")
    
    if not os.path.exists(args.data_dir):
        raise ValueError(f"データディレクトリが存在しません: {args.data_dir}")
    
    history = train_model(args)

if __name__ == "__main__":
    main()