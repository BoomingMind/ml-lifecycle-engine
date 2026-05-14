from __future__ import annotations

"""
Set a reference baseline for drift monitoring.

The API appends each prediction to `data/predictions.jsonl`.
To compare "current" vs "baseline" behavior, we snapshot that file into a
separate reference file.
"""

import argparse
import shutil
from pathlib import Path


def main() -> None:
    """Copy current prediction log to the reference log path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="data/predictions.jsonl")
    parser.add_argument("--reference", default="data/reference_predictions.jsonl")
    args = parser.parse_args()

    src = Path(args.current)
    dst = Path(args.reference)
    if not src.exists():
        raise SystemExit(f"Current predictions not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


if __name__ == "__main__":
    main()
