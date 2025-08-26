mat_ssl: SimSiam pretraining + FastAP fine-tuning on .mat images

This project provides a minimal PyTorch setup to:

- Lazily read images stored in MATLAB `.mat` files (v7.3/HDF5) without loading everything into memory.
- Pretrain a `ResNet50` backbone using SimSiam (self-supervised) with strong augmentations.
- Fine-tune the embedding with the FastAP loss (deep metric learning to rank).

Note: No execution required here; code is scaffolded and ready.

Data Assumptions

- `.mat` files are v7.3 (HDF5-based) so we can index lazily with `h5py`.
- Each `.mat` file contains at least one image dataset variable, e.g. `images`.
  - Supported shapes per file: `[N, H, W, C]`, `[N, C, H, W]`, `[H, W, C, N]`, `[C, H, W, N]`.
  - Images are expected to be uint8 or float in range `[0, 255]` or `[0., 1.]`.
- For FastAP fine-tuning you also need labels. You can store them in the same `.mat` file
  under a variable (e.g. `labels`) of shape `[N]` (int) or `[N, 1]`.

If your files are not v7.3, consider converting them in MATLAB: `save('out.mat','var','-v7.3')`.

Structure

- `mat_ssl/datasets/lazy_mat.py`: Lazy dataset across multiple `.mat` files.
- `mat_ssl/models/simsiam.py`: SimSiam model with ResNet50 backbone.
- `mat_ssl/train/train_simsiam.py`: Self-supervised pretraining script.
- `mat_ssl/train/finetune_fastap.py`: Supervised fine-tuning script with FastAP loss.
- `mat_ssl/utils/mat_utils.py`: Helpers for reading shapes and axis handling.

Installation

Create a virtual environment and install dependencies:

pip install -r requirements.txt

Usage

SimSiam pretraining (unsupervised):

python -m mat_ssl.train.train_simsiam \
  --mat_glob "data/*.mat" \
  --image_key images \
  --epochs 100 \
  --batch_size 256 \
  --output_dir runs/simsiam_r50

FastAP fine-tuning (supervised):

python -m mat_ssl.train.finetune_fastap \
  --mat_glob "data/*.mat" \
  --image_key images \
  --label_key labels \
  --pretrained runs/simsiam_r50/checkpoints/last.pth \
  --epochs 20 \
  --batch_size 256 \
  --output_dir runs/fastap_r50

Key flags:

- `--image_key`: Dataset name inside the `.mat` file containing images.
- `--label_key`: Dataset name inside the `.mat` file containing labels (for FastAP).
- `--mat_glob`: Glob for multiple `.mat` files, merged lazily.

Notes

- For very large datasets, increase `num_workers` in the `DataLoader`.
- If your `.mat` file contains images channel-first or last, axis is detected automatically.
- If reading pre-v7.3 `.mat` is required, lazy loading is not feasible with `scipy.io.loadmat`. Convert to v7.3.
