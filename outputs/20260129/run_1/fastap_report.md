# FastAP Stage Report

- generated_at: 2026-01-29T08:09:45.129584Z
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
- final_loss: -0.2783
- best_loss: -0.2783
- total_time_sec: 728.1

| Epoch | Loss | LR | Time (s) |
| ----- | ---- | -- | -------- |
| 1 | -0.0012 | 4.999e-02 | 88.3 |
| 2 | -0.0042 | 4.995e-02 | 68.2 |
| 3 | -0.0076 | 4.989e-02 | 70.7 |
| 4 | -0.0141 | 4.980e-02 | 70.5 |
| 5 | -0.0249 | 4.969e-02 | 71.7 |
| 6 | -0.0386 | 4.956e-02 | 72.5 |
| 7 | -0.0713 | 4.940e-02 | 70.7 |
| 8 | -0.1169 | 4.921e-02 | 69.9 |
| 9 | -0.1852 | 4.901e-02 | 71.8 |
| 10 | -0.2783 | 4.878e-02 | 73.8 |

## FastAP Training History

- epochs_trained: 10
- final_loss: -0.0047
- best_loss: -0.0049
- total_time_sec: 355.9

| Epoch | Loss | LR | Time (s) |
| ----- | ---- | -- | -------- |
| 1 | -0.0047 | 1.000e-03 | 38.2 |
| 2 | -0.0048 | 1.000e-03 | 35.0 |
| 3 | -0.0047 | 1.000e-03 | 36.6 |
| 4 | -0.0047 | 1.000e-03 | 36.5 |
| 5 | -0.0047 | 1.000e-03 | 36.5 |
| 6 | -0.0048 | 1.000e-03 | 34.2 |
| 7 | -0.0049 | 1.000e-03 | 36.1 |
| 8 | -0.0048 | 1.000e-03 | 33.9 |
| 9 | -0.0048 | 1.000e-03 | 34.0 |
| 10 | -0.0047 | 1.000e-03 | 35.0 |

## Evaluation Metrics

- mAP: skipped (no labels)

### Top-k SSIM (avg)
| K | SSIM |
| - | ---- |
| 1 | 0.9518 |
| 5 | 0.9417 |
| 10 | 0.9350 |

## Retrieval Examples

### Internal query (index=0)
| Rank | Frame | Score | Label |
| ---- | ----- | ----- | ----- |
| 1 | 0 | 1.0000 | -1 |
| 2 | 23 | 0.8380 | -1 |
| 3 | 15 | 0.8377 | -1 |
| 4 | 1 | 0.8198 | -1 |
| 5 | 114 | 0.8161 | -1 |
| 6 | 2 | 0.8039 | -1 |
| 7 | 24 | 0.8039 | -1 |
| 8 | 4 | 0.8029 | -1 |
| 9 | 97 | 0.8013 | -1 |
| 10 | 3 | 0.7947 | -1 |

### External query (/content/data/invivo_normalized.png)
| Rank | Frame | Score | Label |
| ---- | ----- | ----- | ----- |
| 1 | 25 | 0.4460 | -1 |
| 2 | 412 | 0.4459 | -1 |
| 3 | 101 | 0.4448 | -1 |
| 4 | 180 | 0.4443 | -1 |
| 5 | 157 | 0.4412 | -1 |
| 6 | 360 | 0.4410 | -1 |
| 7 | 141 | 0.4396 | -1 |
| 8 | 143 | 0.4331 | -1 |
| 9 | 104 | 0.4329 | -1 |
| 10 | 221 | 0.4326 | -1 |



## Dataset
- dataset_path: /content/data/dataset.mat
- dataset_shape: (445, 1, 3319, 256)
- image_key: Acq/Amp
- image_axes: (0, 2, 1)
- labels_path: /content/annotations/dataset_labels.csv
- label_positive_ratio: nan

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
