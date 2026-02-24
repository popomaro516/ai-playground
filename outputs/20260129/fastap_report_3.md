# FastAP Stage Report

- generated_at: 2026-01-29T09:56:22.167376Z
- device: cuda
- checkpoint_path: /content/save/latest.pth
- embeddings_path: /content/save/inference_results/fastap_embeddings.npy
- metrics_path: /content/save/inference_results/fastap_metrics.json

## Config

- backbone: resnet18
- projector_dim: 2048
- embedding_dim: 256
- simsiam_epochs: 20
- fastap_epochs: 10
- fastap_bins: 100
- fastap_sigma: 0.05
- freeze_backbone: False
- freeze_projector: False
- use_labels: False

## SimSiam Training History

- epochs_trained: 20
- final_loss: -0.9363
- best_loss: -0.9363
- total_time_sec: 1471.0

| Epoch | Loss | LR | Time (s) |
| ----- | ---- | -- | -------- |
| 1 | -0.0013 | 4.999e-02 | 73.8 |
| 2 | -0.0035 | 4.995e-02 | 71.6 |
| 3 | -0.0066 | 4.989e-02 | 78.8 |
| 4 | -0.0114 | 4.980e-02 | 74.0 |
| 5 | -0.0198 | 4.969e-02 | 75.4 |
| 6 | -0.0307 | 4.956e-02 | 76.2 |
| 7 | -0.0517 | 4.940e-02 | 79.1 |
| 8 | -0.0868 | 4.921e-02 | 74.1 |
| 9 | -0.1360 | 4.901e-02 | 74.0 |
| 10 | -0.2170 | 4.878e-02 | 76.4 |
| 11 | -0.3098 | 4.852e-02 | 74.4 |
| 12 | -0.4301 | 4.824e-02 | 71.0 |
| 13 | -0.5540 | 4.794e-02 | 73.5 |
| 14 | -0.6594 | 4.762e-02 | 69.8 |
| 15 | -0.7444 | 4.728e-02 | 73.1 |
| 16 | -0.8090 | 4.691e-02 | 70.8 |
| 17 | -0.8364 | 4.652e-02 | 70.9 |
| 18 | -0.8760 | 4.611e-02 | 72.3 |
| 19 | -0.9099 | 4.568e-02 | 71.3 |
| 20 | -0.9363 | 4.523e-02 | 70.6 |

## FastAP Training History

- epochs_trained: 10
- final_loss: -0.0041
- best_loss: -0.0041
- total_time_sec: 362.8

| Epoch | Loss | LR | Time (s) |
| ----- | ---- | -- | -------- |
| 1 | -0.0041 | 1.000e-03 | 36.5 |
| 2 | -0.0041 | 1.000e-03 | 37.9 |
| 3 | -0.0041 | 1.000e-03 | 35.2 |
| 4 | -0.0041 | 1.000e-03 | 36.8 |
| 5 | -0.0041 | 1.000e-03 | 37.9 |
| 6 | -0.0041 | 1.000e-03 | 36.6 |
| 7 | -0.0041 | 1.000e-03 | 35.7 |
| 8 | -0.0041 | 1.000e-03 | 34.8 |
| 9 | -0.0041 | 1.000e-03 | 35.5 |
| 10 | -0.0041 | 1.000e-03 | 35.9 |

## Evaluation Metrics

- mAP: 0.7761

| K | Precision | Recall |
| - | --------- | ------ |
| 1 | 0.8854 | 0.0038 |
| 5 | 0.8369 | 0.0178 |
| 10 | 0.8157 | 0.0335 |

## Retrieval Examples

### Internal query (index=0)
| Rank | Frame | Score | Label |
| ---- | ----- | ----- | ----- |
| 1 | 0 | 1.0000 | 1 |
| 2 | 11 | 0.8643 | 0 |
| 3 | 10 | 0.8492 | 0 |
| 4 | 14 | 0.8443 | 1 |
| 5 | 22 | 0.8409 | 0 |
| 6 | 1 | 0.8387 | 1 |
| 7 | 13 | 0.8341 | 0 |
| 8 | 20 | 0.8245 | 1 |
| 9 | 17 | 0.8208 | 1 |
| 10 | 19 | 0.8139 | 1 |

### External query (/content/data/invivo_normalized.png)
| Rank | Frame | Score | Label |
| ---- | ----- | ----- | ----- |
| 1 | 192 | 0.5178 | 0 |
| 2 | 376 | 0.5104 | 0 |
| 3 | 188 | 0.5066 | 0 |
| 4 | 38 | 0.4999 | 0 |
| 5 | 52 | 0.4933 | 0 |
| 6 | 336 | 0.4920 | 0 |
| 7 | 53 | 0.4911 | 0 |
| 8 | 54 | 0.4910 | 0 |
| 9 | 55 | 0.4908 | 0 |
| 10 | 351 | 0.4881 | 0 |



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
