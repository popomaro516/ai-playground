# FastAP Stage Report

- generated_at: 2026-01-29T09:00:59.334181Z
- device: cuda
- checkpoint_path: /content/save/latest.pth
- embeddings_path: /content/save/inference_results/fastap_embeddings.npy
- metrics_path: /content/save/inference_results/fastap_metrics.json

## Config

- backbone: resnet18
- projector_dim: 2048
- embedding_dim: 256
- simsiam_epochs: 10
- fastap_epochs: 10
- fastap_bins: 100
- fastap_sigma: 0.05
- freeze_backbone: False
- freeze_projector: False
- use_labels: False

## SimSiam Training History

- epochs_trained: 10
- final_loss: -0.2170
- best_loss: -0.2170
- total_time_sec: 784.6

| Epoch | Loss | LR | Time (s) |
| ----- | ---- | -- | -------- |
| 1 | -0.0013 | 4.999e-02 | 95.2 |
| 2 | -0.0035 | 4.995e-02 | 73.2 |
| 3 | -0.0066 | 4.989e-02 | 76.6 |
| 4 | -0.0114 | 4.980e-02 | 76.9 |
| 5 | -0.0198 | 4.969e-02 | 77.8 |
| 6 | -0.0307 | 4.956e-02 | 77.0 |
| 7 | -0.0517 | 4.940e-02 | 78.1 |
| 8 | -0.0868 | 4.921e-02 | 76.9 |
| 9 | -0.1360 | 4.901e-02 | 77.3 |
| 10 | -0.2170 | 4.878e-02 | 75.5 |

## FastAP Training History

- epochs_trained: 10
- final_loss: -0.0050
- best_loss: -0.0053
- total_time_sec: 387.7

| Epoch | Loss | LR | Time (s) |
| ----- | ---- | -- | -------- |
| 1 | -0.0050 | 1.000e-03 | 41.6 |
| 2 | -0.0052 | 1.000e-03 | 38.0 |
| 3 | -0.0051 | 1.000e-03 | 38.6 |
| 4 | -0.0052 | 1.000e-03 | 38.2 |
| 5 | -0.0050 | 1.000e-03 | 44.1 |
| 6 | -0.0052 | 1.000e-03 | 38.0 |
| 7 | -0.0053 | 1.000e-03 | 38.7 |
| 8 | -0.0050 | 1.000e-03 | 36.3 |
| 9 | -0.0052 | 1.000e-03 | 37.1 |
| 10 | -0.0050 | 1.000e-03 | 37.1 |

## Evaluation Metrics

- mAP: 0.7498

| K | Precision | Recall |
| - | --------- | ------ |
| 1 | 0.8517 | 0.0036 |
| 5 | 0.8103 | 0.0166 |
| 10 | 0.7796 | 0.0306 |

## Retrieval Examples

### Internal query (index=0)
| Rank | Frame | Score | Label |
| ---- | ----- | ----- | ----- |
| 1 | 0 | 1.0000 | 1 |
| 2 | 1 | 0.7687 | 1 |
| 3 | 10 | 0.7338 | 0 |
| 4 | 11 | 0.7294 | 0 |
| 5 | 9 | 0.7291 | 0 |
| 6 | 2 | 0.7287 | 1 |
| 7 | 22 | 0.7283 | 0 |
| 8 | 20 | 0.7236 | 1 |
| 9 | 17 | 0.7152 | 1 |
| 10 | 180 | 0.7091 | 0 |

### External query (/content/data/invivo_normalized.png)
| Rank | Frame | Score | Label |
| ---- | ----- | ----- | ----- |
| 1 | 73 | 0.5323 | 0 |
| 2 | 405 | 0.4992 | 0 |
| 3 | 75 | 0.4949 | 0 |
| 4 | 406 | 0.4856 | 0 |
| 5 | 227 | 0.4814 | 0 |
| 6 | 395 | 0.4772 | 0 |
| 7 | 382 | 0.4668 | 0 |
| 8 | 355 | 0.4668 | 0 |
| 9 | 390 | 0.4650 | 0 |
| 10 | 336 | 0.4593 | 0 |



## Dataset
- dataset_path: /content/data/dataset.mat
- dataset_shape: (445, 1, 3319, 256)
- image_key: Acq/Amp
- image_axes: (0, 2, 1)
- labels_path: /content/annotations/dataset_labels.csv
- label_positive_ratio: 0.1955

## Preprocessing & Augmentation
- B-mode: x=|A|/Amax, I=20log10(x), clip[-60,0], normalize to [0,1]
- Eval: Resize(256) + CenterCrop(224) + ImageNet normalize
- SimSiam aug: RandomResizedCrop(224, scale=0.2-1.0), HFlip, ColorJitter(0.4), Gray(0.2), Blur, ImageNet normalize

## Temporal Consistency Loss
- L = L_simsiam + w * L_temporal
- L_temporal = -cos(p(x_t), z(x_{t+Δ}))
## Pseudo Labels
- motion_smooth_window: 5
- motion_percentile: 20.0
- min_stable_length: 3
- reference_image: /content/data/invivo_normalized.png
- reference_metric: ssim
- reference_percentile: 30.0

## Temporal Consistency (SimSiam)
- temporal_enabled: True
- temporal_delta: 1
- temporal_weight: 0.2
