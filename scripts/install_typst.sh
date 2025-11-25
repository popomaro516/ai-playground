#!/usr/bin/env bash
set -euo pipefail

# Install Typst locally into .bin/ without requiring system-wide privileges.
# Robust variant: resolves latest tag, tries gnu/musl assets, then falls back to official install script.

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BIN_DIR="$ROOT_DIR/.bin"
mkdir -p "$BIN_DIR"

OS=$(uname -s)
ARCH=$(uname -m)

if [[ "$OS" != "Linux" ]]; then
  echo "This helper currently targets Linux." >&2
  exit 1
fi

case "$ARCH" in
  x86_64|amd64)
    TRIPLE_GNU="x86_64-unknown-linux-gnu"
    TRIPLE_MUSL="x86_64-unknown-linux-musl"
    ;;
  *)
    echo "Unsupported arch: $ARCH (script only handles x86_64)." >&2
    exit 1
    ;;
esac

cd "$BIN_DIR"

# Detect the latest release tag (e.g., v0.12.0)
if command -v curl >/dev/null 2>&1; then
  TAG=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/typst/typst/releases/latest" | sed -E 's#.*/tag/##')
elif command -v wget >/dev/null 2>&1; then
  TAG=$(wget --max-redirect=0 -S -O /dev/null "https://github.com/typst/typst/releases/latest" 2>&1 | awk '/^  Location: /{print $2}' | sed -E 's#.*/tag/##')
else
  echo "Neither curl nor wget is available." >&2
  exit 1
fi

if [[ -z "${TAG:-}" ]]; then
  echo "Failed to detect latest Typst tag; falling back to install script." >&2
  TAG=""
fi

download_and_extract() {
  local url="$1"
  local out="typst.tar.xz"
  echo "Downloading: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL -o "$out" "$url" || return 1
  else
    wget -O "$out" "$url" || return 1
  fi
  # quick sanity check
  if ! xz -t "$out" >/dev/null 2>&1; then
    echo "Archive test failed (possibly 404 HTML)." >&2
    return 1
  fi
  echo "Extracting ..."
  tar -xf "$out"
  return 0
}

FOUND=0
if [[ -n "$TAG" ]]; then
  URL_GNU="https://github.com/typst/typst/releases/download/${TAG}/typst-${TRIPLE_GNU}.tar.xz"
  URL_MUSL="https://github.com/typst/typst/releases/download/${TAG}/typst-${TRIPLE_MUSL}.tar.xz"
  download_and_extract "$URL_GNU" || download_and_extract "$URL_MUSL" || true
  if [[ -d . ]]; then
    if BIN_PATH=$(find . -type f -name typst -perm -u+x | head -n1); then
      FOUND=1
    fi
  fi
fi

if [[ "$FOUND" -ne 1 ]]; then
  echo "Falling back to official install script (typst.app)." >&2
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL https://typst.app/install.sh | bash
  else
    wget -qO- https://typst.app/install.sh | bash
  fi
  if command -v typst >/dev/null 2>&1; then
    cp "$(command -v typst)" ./typst
    chmod +x ./typst
    FOUND=1
  fi
fi

if [[ "$FOUND" -ne 1 ]]; then
  echo "Failed to install Typst automatically. Consider: conda install -c conda-forge typst OR cargo install typst-cli" >&2
  exit 1
fi

./typst --version || true
echo
echo "Typst installed to $BIN_DIR/typst"
echo "Build with: $BIN_DIR/typst compile $ROOT_DIR/paper/main.typ $ROOT_DIR/paper/main.pdf"
