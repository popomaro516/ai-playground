# SimSiam + FastAP ノートブックのデータ配置ガイド

対象: `notebook/simsiam_fastap_ultrasound.ipynb`

このノートブックをそのまま動かすための、データ/ラベル/画像の配置規約と実行手順をまとめます。パスはノートブックからの相対指定（`../...`）になっているため、以下のディレクトリ構成に揃えてください。

## ディレクトリ構成

```
(repo root)/
├─ data/
│  ├─ dataset.mat               # 超音波データ (MAT v7.3 / HDF5)
│  ├─ dataset_labels.csv        # フレーム擬似ラベル CSV
│  ├─ invivo/                   # in vivo 参照画像（任意、下記参照）
│  │  └─ invivo.jpg             # Google Drive 配布物（拡張子は実ファイルに合わせる）
│  └─ breath_hold.png           # 息止め時の外部画像 (任意、PNG/JPG 等)
└─ notebook/
   └─ simsiam_fastap_ultrasound.ipynb
```

- ノートブック内の既定パス
  - `DATA_MAT = '../data/dataset.mat'`
  - `LABEL_CSV = '../data/dataset_labels.csv'`
  - `BREATH_HOLD_IMAGE = ''  # 実行時に '../data/...' へ設定`
  - `CHECKPOINT_DIR = '../checkpoints'`（未存在なら自動作成）

必要に応じてノートブックの該当セルでパスを変更しても構いませんが、上記構成に合わせると手戻りが少ないです。

### in vivo 参照画像を使う場合

- Google Drive 配布物: <https://drive.google.com/file/d/1IyBjzMmJhZ4JRQu90OOl0BeZ1O3N5laf/view?usp=sharing>
  - ダウンロードすると JPEG (`*.jpg`) が得られます。`data/invivo/` ディレクトリを作成し、元ファイル名が異なる場合は `invivo.jpg`（拡張子は `.jpg` のまま）にリネームして配置してください。
- ノートブック内で類似フレーム検索に利用する際は、該当セルを以下のように設定します。
  - `BREATH_HOLD_IMAGE = '../data/invivo/invivo.jpg'`
- 既定の `breath_hold.png` などと併用する場合は、評価したい画像に応じてセル内のパスを切り替えてください。

## MAT ファイルの前提

- 期待キー: `Acq/Amp`（必要に応じて `Acq/Data` でも可）
- 形状想定: `(frame, lateral, depth)`
- ノートブックでは下記の軸設定で解釈します:
  - `IMAGE_KEY = 'Acq/Amp'`
  - `IMAGE_AXES = (0, 2, 1)`  # (frame, depth, lateral) として扱う

MAT/HDF5 構造や `image_axes` の詳細は `docs/ultrasound_mat_preprocessing.md` も参照してください。

## 擬似ラベル CSV（dataset_labels.csv）

- 必須列（ヘッダ行を含む）
  - `frame,label,score,smooth_score`
- 行はフレーム 0..N-1 に対応し、`label` は 0/1 の整数（安定/不安定など）です。
- 生成元：以下のいずれかで作成し、`data/dataset_labels.csv` に配置してください。
  1) スクリプト: `scripts/generate_pseudo_labels.py`
     - 例:
       ```bash
       python scripts/generate_pseudo_labels.py \
         --mat_path data/dataset.mat \
         --image_key Acq/Amp \
         --image_axes 0 2 1 \
         --output_csv annotations/dataset_labels.csv \
         --percentile 20
       cp annotations/dataset_labels.csv data/dataset_labels.csv
       ```
  2) ノートブック: `notebook/pseudo_labeling.ipynb`
     - 生成先が `annotations/dataset_labels.csv` なので、上と同様に `data/` へコピーしてください。

CSV の中身例:

```
frame,label,score,smooth_score
0,0,0.0,0.0
1,1,0.0123,0.0101
2,1,0.0118,0.0110
...
```

## 息止め画像（breath_hold.png など）

- 任意の外部画像（PNG/JPG）を `data/` に置き、ノートブックの変数にパスを設定します。
  - 例: `BREATH_HOLD_IMAGE = '../data/breath_hold.png'`
- ノートブック内で 224×224 中心クロップ + 正規化を行い、学習済み埋め込みとのコサイン類似度で Top-K フレームが表示されます。

## 出力（チェックポイント・埋め込み）

- ノートブック実行中に `checkpoints/` が作成され、以下が保存されます。
  - `checkpoints/simsiam_latest.pth`（学習済みモデル）
  - `checkpoints/simsiam_embeddings.npy`（評価用埋め込み）

## すぐに動かす手順（クイックスタート）

1. `(repo root)/data/` を作成し、`dataset.mat` を配置。
2. 擬似ラベルを作成し、`data/dataset_labels.csv` として配置（上記のどちらかの方法）。
3. （任意）息止め画像を `data/breath_hold.png` などとして保存。
4. `notebook/simsiam_fastap_ultrasound.ipynb` を開き、必要なら `DATA_MAT`/`LABEL_CSV`/`BREATH_HOLD_IMAGE` を調整。
5. 上から順に実行。mAP が表示され、`BREATH_HOLD_IMAGE` を設定すると Top-K 類似フレームが出力されます。

---
トラブルシュート:
- `KeyError: image_key ... not found` → `IMAGE_KEY` が MAT 内のパスと一致しているか確認 (`Acq/Amp` など)。
- 形状解釈エラー → `IMAGE_AXES` がデータの軸順に合っているか確認（本ノート既定は `(0,2,1)`）。
- CSV 列不足 → ヘッダ行と上記4列が揃っているか確認。
