from __future__ import annotations

import argparse
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()

    def report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        print(f"\rDownloading: {pct:5.1f}% ({downloaded/1024/1024:.1f} MB)", end="")

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    print()
    tmp.replace(dst)


def _extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:xz") as tar:
        tar.extractall(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("MVTEC_AD_URL", DEFAULT_URL))
    parser.add_argument("--archive", default=os.getenv("MVTEC_AD_ARCHIVE", "data/raw/mvtec_anomaly_detection.tar.xz"))
    parser.add_argument("--out", default=os.getenv("MVTEC_AD_OUT", "data/raw/mvtec_ad"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    out_dir = Path(args.out)

    if not args.skip_download:
        if archive_path.exists():
            print(f"Archive already exists: {archive_path}")
        else:
            print(f"Downloading MVTec AD archive to: {archive_path}")
            print("License: CC BY-NC-SA 4.0 (non-commercial). Review terms on the official MVTec website.")
            _download(args.url, archive_path)
            print(f"SHA256: {_sha256(archive_path)}")

    if not args.skip_extract:
        if not archive_path.exists():
            raise SystemExit(f"Archive not found: {archive_path}")
        print(f"Extracting to: {out_dir}")
        _extract(archive_path, out_dir)


if __name__ == "__main__":
    main()
