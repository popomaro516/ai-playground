# 開発サマリー（.mat画像の遅延読み込み + SimSiam事前学習 + FastAP微調整）

## やったこと
- プロジェクトの雛形を作成し、以下を実装しました。
  - 遅延読み込み対応の `.mat` 画像データセット（v7.3/HDF5想定）
  - ResNet50バックボーンのSimSiamモデルと学習スクリプト
  - FastAP損失を用いた距離学習（微調整）スクリプト
  - 付随ユーティリティ、README、依存関係ファイル
- 画像配列のレイアウト（`[N,H,W,C]`, `[N,C,H,W]`, `[H,W,C,N]`, `[C,H,W,N]`）を自動判別してCHWに揃える実装。
- SimSiam用の強いデータ拡張（2ビュー生成）を組み込み。
- Fine-tuneでは `pytorch-metric-learning` の `FastAPLoss` を利用。

ディレクトリ構成（主なもの）:
- `mat_ssl/datasets/lazy_mat.py`: 遅延読み込みデータセット
- `mat_ssl/models/simsiam.py`: SimSiam本体、埋め込みヘッド等
- `mat_ssl/train/train_simsiam.py`: 自己教師ありの事前学習
- `mat_ssl/train/finetune_fastap.py`: FastAPによる微調整
- `mat_ssl/utils/mat_utils.py`: HDF5（v7.3）点検用ヘルパ
- `requirements.txt`, `mat_ssl/README.md`

## 結果
- コード一式を作成・配置済み（この環境では実行は行っていません）。
- 真の遅延読み込みを行うため、`MATLAB v7.3`（HDF5）形式の `.mat` を前提にしています。
  - 1サンプル単位で `h5py` によるスライス読み出しを行い、全読み込みを回避します。
- SimSiam事前学習スクリプトは、ResNet50+Projector+Predictor構成で負のコサイン類似度損失を採用。
- 微調整スクリプトは、SimSiamで学習済みバックボーンを読み込み、`FastAPLoss`で埋め込みを最適化。
- 想定コマンド例（参考）:
  - 事前学習（自己教師あり）:
    ```bash
    python -m mat_ssl.train.train_simsiam \
      --mat_glob "data/*.mat" \
      --image_key images \
      --epochs 100 \
      --batch_size 256 \
      --output_dir runs/simsiam_r50
    ```
  - 微調整（教師あり・ラベル必要）:
    ```bash
    python -m mat_ssl.train.finetune_fastap \
      --mat_glob "data/*.mat" \
      --image_key images \
      --label_key labels \
      --pretrained runs/simsiam_r50/checkpoints/last.pth \
      --epochs 20 \
      --batch_size 256 \
      --output_dir runs/fastap_r50
    ```

## これから必要かもしれないこと
- データ検証:
  - `.mat` が v7.3 であるかの確認（必要なら変換: `save('out.mat','var','-v7.3')`）。
  - 実データ内のキー名（`images`, `labels` など）の点検と整備。
  - 画素値レンジの統一（[0,1] or [0,255]）。
- 実行/学習周り:
  - LARS/AdamWなど最適化手法・ハイパラのチューニング（バッチサイズ、学習率、温度/スケジュール等）。
  - マルチGPU・分散学習対応（DDP）と混合精度（`--fp16`）の利用検証。
  - オーグメントの強度調整（SimSiamは学習安定性に影響大）。
  - チェックポイントの世代管理・ベストモデル保存。
- 評価/分析:
  - FastAPに加え、Recall@K、mAP、NMI等の評価スクリプト追加。
  - 埋め込みの可視化（t-SNE/UMAP）。
  - クラス不均衡対策、サンプラーの導入。
- 品質/運用:
  - ユニットテスト（形状変換、遅延読み込み、ラベル整合性）。
  - ログ（TensorBoardやW&B）の導入。
  - 設定ファイル化（YAML/CLI統合）と再現性（seed固定）。

---
依存関係は `requirements.txt` に記載済みです。必要に応じてGPU版PyTorchのインストール手順（CUDAバージョン）を調整してください。
