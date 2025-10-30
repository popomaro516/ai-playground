#!/usr/bin/env bash
set -euo pipefail

echo "[codex setup] Starting environment setup..."

if command -v npm >/dev/null 2>&1; then
  echo "[codex setup] npm already installed: $(command -v npm)"
else
  echo "[codex setup] npm not found. Attempting to install..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y npm
  elif command -v yum >/dev/null 2>&1; then
    sudo yum install -y npm
  elif command -v brew >/dev/null 2>&1; then
    brew install npm
  else
    echo "[codex setup] ERROR: Unable to install npm automatically. Install npm manually and rerun." >&2
    exit 1
  fi
fi

echo "[codex setup] Installing codex via npm..."
npm install -g codex

echo "[codex setup] Done."
