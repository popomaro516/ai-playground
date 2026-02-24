#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/workspaces/ai-playground/data"
OUT_FILE="$DATA_DIR/dataset.mat"
URL="https://drive.google.com/file/d/1fa0gaEmbtGmqZ92L0EqzhH5LiMUAztix/view?usp=sharing"

mkdir -p "$DATA_DIR"

if [[ -f "$OUT_FILE" ]]; then
  echo "Dataset already exists: $OUT_FILE"
  exit 0
fi

if ! command -v gdown >/dev/null 2>&1; then
  echo "gdown not found. Install with: python -m pip install gdown"
  exit 1
fi

gdown --fuzzy "$URL" -O "$OUT_FILE"

echo "Downloaded to $OUT_FILE"
