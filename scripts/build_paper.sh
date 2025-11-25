#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BIN_LOCAL="$ROOT_DIR/.bin/typst"
SRC="$ROOT_DIR/paper/main.typ"
OUT="$ROOT_DIR/paper/main.pdf"

if [[ -x "$BIN_LOCAL" ]]; then
  TYPST_BIN="$BIN_LOCAL"
elif command -v typst >/dev/null 2>&1; then
  TYPST_BIN="$(command -v typst)"
else
  echo "Typst not found. Install with: bash scripts/install_typst.sh" >&2
  exit 1
fi

"$TYPST_BIN" compile "$SRC" "$OUT"
echo "Built: $OUT"

