# Ultrasound `.mat` Frame Handling

このドキュメントでは、`data/dataset.mat` に保存された超音波計測データを学習用フレームとして扱う手順をまとめます。特に以下をカバーします。

- HDF5 構造の確認方法
- 遅延ロード時の軸指定 (`image_axes`) と B-mode 変換
- 物理座標に基づいた縦横比補正（resampling）
- 表示時の上下方向（浅部を上に）

## データ構造の概要

MATLAB v7.3 (HDF5) 形式です。主要データセットは `Acq` グループ直下にまとめられています。

```text
Acq/
  Amp            (445, 256, 3319) float64  # 各フレームのアンプリチュード
  Data           (445, 256, 3319) float64  # 生データ（Amp と同形状）
  x              (3, 1) object             # 座標参照 (object reference)
  ... (メタデータ多数)
```

`Acq/x` には HDF5 object reference が格納されており、以下の順序で参照先が保存されています。

| インデックス | 含まれる座標 | 長さ | 概要 |
|--------------|--------------|------|------|
| `x[0,0]` | 深さ方向 (Depth) | 3319 | 約 −7.4 mm → 164 mm |
| `x[1,0]` | ラテラル方向 (Lateral) | 256 | 約 −110.6 mm → 110.6 mm |
| `x[2,0]` | フレーム時刻 (Time) | 445 | 約 0.035 s → 15.55 s |

このうち `Depth` と `Lateral` を使って縦横比を補正します。

## 遅延ロード (`LazyMatImageDataset`)

`mat_ssl/datasets/lazy_mat.py` に `LazyMatImageDataset` が用意されています。バージョンアップ後は 3D `(N, H, W)` データにも対応でき、`image_axes` で軸順をオーバーライドできます。

### 例: `Acq/Amp` の1フレーム取得

```python
from mat_ssl.datasets.lazy_mat import LazyMatImageDataset

ds = LazyMatImageDataset(
    ["data/dataset.mat"],
    image_key="Acq/Amp",      # or "Acq/Data"
    normalize_255=False,
    image_axes=(0, 2, 1),      # (frame, depth, lateral)
)
frame = ds[0]  # shape: [1, depth_samples, lateral_samples]
```

- `image_axes=(0,2,1)` により、0番目がフレーム、2番目がラテラル、1番目が深さであることを明示しています。
- これにより `frame` の形状は `[1, 3319, 256]` （[C,H,W]）となります。

## B-mode 変換と縦横比補正

超音波 B-mode 表示では振幅の絶対値を取り、ログ圧縮後にダイナミックレンジを調整します。また、深さサンプル数と横方向サンプル数が大きく異なるため、座標情報に基づいて縦横比を補正（resample）すると自然な見た目になります。

### 手順サマリー

1. **振幅 → ログ圧縮**: `abs` → 最大値で正規化 → `20*log10` → [-60, 0] dB にクリップ → [0,1] へ線形変換。
2. **縦横比補正**: `Depth` と `Lateral` の実長さから比率 `ratio = depth_range / lateral_range` を算出し、`height = round(width * ratio)` で縦方向だけリサイズ。
3. **保存**: `PIL.Image.fromarray` 等で PNG に変換。
   - そのまま保存すれば、浅部（深度が小さい側）が画像の上部に来る構成です。必要に応じて `np.flipud` で反転してください。

### コード例

```python
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import h5py

from mat_ssl.datasets.lazy_mat import LazyMatImageDataset

# 1. 遅延ロード
loader = LazyMatImageDataset(
    ["data/dataset.mat"],
    image_key="Acq/Amp",
    normalize_255=False,
    image_axes=(0, 2, 1),
)
frame = loader[0].unsqueeze(0).float()  # shape: [1, 1, depth, lateral]

# 2. B-mode 変換
frame = torch.abs(frame)
frame /= frame.max() + 1e-12
frame = 20 * torch.log10(frame + 1e-12)
frame = torch.clamp(frame, min=-60.0, max=0.0)
frame = (frame + 60.0) / 60.0  # [0,1]

# 3. 縦横比補正に必要な座標を取得
with h5py.File("data/dataset.mat", "r") as f:
    depth = f[f['Acq']['x'][0, 0]][...].flatten()
    lateral = f[f['Acq']['x'][1, 0]][...].flatten()
ratio = float((depth.max() - depth.min()) / (lateral.max() - lateral.min()))

_, _, h, w = frame.shape
new_h = max(1, int(round(w * ratio)))
frame_resampled = F.interpolate(frame, size=(new_h, w), mode="bilinear", align_corners=False)

# 4. 保存（必要に応じて上下反転を追加）
img = frame_resampled.squeeze().numpy()
Image.fromarray((img * 255).astype(np.uint8)).save("data/sample_bmode_resampled.png")
```

上記の `sample_bmode_resampled.png` が、見栄えのよい縦横比を持つ B-mode フレームです。データの向きに応じて浅部と深部を入れ替えたい場合は `np.flipud` を追加してください。

## 学習用への組み込み

- `LazyMatImageDataset` に渡す `transform` 内で、上記 B-mode 変換とリサイズ（必要に応じて反転）を含めると、SimSiam 等で扱いやすい正規化画像を得られます。
- 物理座標のメタデータに基づいているので、他ファイルでも `Acq/x` を読み取るロジックを組み込めば同じ手順で処理できます。

## 注意点

- 振幅の絶対値がゼロや極端に小さいと `log10` で発散するので、`1e-12` 程度を加えて防いでいます。
- 深さ側の範囲が負から始まる場合、必要に応じて原点（0）基準へ平行移動するなど調整してください。
- `Acq/Data` と `Acq/Amp` は同形状ですが、どちらを使用するかは計測条件に応じて選択してください。

---
このドキュメントでは単一フレームの例を示しました。複数フレームを処理する場合も同様に遅延ロード＋逐次処理でメモリ 2GB 制限内に収めることができます。
