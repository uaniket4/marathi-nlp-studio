"""Download the L3Cube-MahaNER IOB dataset into ``data/``.

Usage:
    python scripts/download_data.py

Only downloads the three IOB files plus the dataset README. No dataset is
bundled in the repo; this script fetches it from the official source.
"""

from __future__ import annotations

import os
import sys
import urllib.request

RAW_BASE = (
    "https://raw.githubusercontent.com/l3cube-pune/MarathiNLP/main/L3Cube-MahaNER/"
)

FILES = {
    "IOB/train_iob.txt": "IOB/train_iob.txt",
    "IOB/valid_iob.txt": "IOB/valid_iob.txt",
    "IOB/test_iob.txt": "IOB/test_iob.txt",
    "README.md": "SOURCE_README.md",
}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(here, "..", "data"))
    os.makedirs(os.path.join(data_dir, "IOB"), exist_ok=True)

    for remote, local in FILES.items():
        dest = os.path.join(data_dir, local)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = RAW_BASE + remote
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:  # pragma: no cover - network failure path
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 1
        size = os.path.getsize(dest)
        print(f"  -> {dest} ({size:,} bytes)")

    print("Done. Dataset ready in", data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
