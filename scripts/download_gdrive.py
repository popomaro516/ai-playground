#!/usr/bin/env python3
"""Download files from Google Drive using gdown.

Usage examples:
  python scripts/download_gdrive.py --id 1AbCdEFgHIjkLMnOP --out data/dataset.zip
  python scripts/download_gdrive.py --url https://drive.google.com/uc?id=1AbCdEFgHIjkLMnOP --out data/dataset.zip

Requires: pip install gdown
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError as exc:  # pragma: no cover
    raise SystemExit("gdown がインストールされていません。`pip install gdown` を実行してください。") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download files from Google Drive via gdown")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", dest="file_id", type=str, help="Google Drive file ID")
    group.add_argument("--url", dest="url", type=str, help="Google Drive shareable URL")
    parser.add_argument("--out", dest="out", type=str, required=True, help="Download destination path")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching (gdown option)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.file_id:
        url = f"https://drive.google.com/uc?id={args.file_id}"
    else:
        url = args.url

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gdown.download(url=url, output=str(out_path), quiet=args.quiet, fuzzy=args.fuzzy)
    except Exception as exc:
        print(f"[ERROR] ダウンロードに失敗しました: {exc}", file=sys.stderr)
        return 1

    if not out_path.exists():
        print("[ERROR] ダウンロードは完了しませんでした。出力ファイルが確認できません。", file=sys.stderr)
        return 1

    print(f"[INFO] ダウンロード完了: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
