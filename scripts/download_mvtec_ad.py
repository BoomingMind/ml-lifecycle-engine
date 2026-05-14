from __future__ import annotations

"""
Download and extract the MVTec AD dataset for local development.

Why this exists:
Most CV/MLOps repos assume you already have a dataset. In real projects, the
first reproducibility problem is: "How does a new teammate get the same data?"

This script makes that step explicit:
- downloads the official MVTec AD archive (tar.xz)
- prints a SHA256 checksum so you can verify integrity
- extracts into a folder under data/raw/

Important:
MVTec AD is released under CC BY-NC-SA 4.0 (non-commercial).
Review license terms on the official MVTec website before using it.
"""

import argparse
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"


def _sha256(path: Path) -> str:
    """Compute SHA256 for a file in a streaming way (doesn't load whole file into memory)."""
    # Create a SHA256 hasher object. We'll "feed" it bytes chunk-by-chunk.
    h = hashlib.sha256()
    # Open the file in binary mode so we read raw bytes (not text).
    with path.open("rb") as f:
        # Read the file in 1MB chunks. This keeps memory usage constant even for large files.
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            # Update the hash with the next chunk of bytes.
            h.update(chunk)
    # Return the final hex digest (human-readable checksum).
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    """
    Download a URL to a destination path.

    We download to a temporary file first and then rename, so a partial download
    doesn't look like a valid archive.
    """
    # Ensure the destination directory exists (e.g. data/raw/).
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Download into a temporary file first:
    # - if the download fails midway, you do not end up with a corrupted "final" archive
    # - once completed, we rename temp -> final in one step
    tmp = dst.with_suffix(dst.suffix + ".partial")
    if tmp.exists():
        # If a previous partial download exists, delete it so we start clean.
        tmp.unlink()

    def report(block_num: int, block_size: int, total_size: int) -> None:
        """Progress callback used by urlretrieve."""
        # total_size can be -1/0 if the server doesn't provide Content-Length.
        if total_size <= 0:
            return
        # urlretrieve downloads in blocks. Estimate how many bytes we have so far.
        downloaded = block_num * block_size
        # Convert to a percentage but clamp to 100%.
        pct = min(100.0, downloaded * 100.0 / total_size)
        # Print a "live updating" progress line. end="" keeps it on one line.
        print(f"\rDownloading: {pct:5.1f}% ({downloaded/1024/1024:.1f} MB)", end="")

    # Perform the actual download.
    # - reporthook calls our report() function repeatedly so the user sees progress.
    urllib.request.urlretrieve(url, tmp, reporthook=report)
    # Finish the progress line with a newline.
    print()
    # Rename the finished temp file to the final destination.
    tmp.replace(dst)


def _extract(archive: Path, out_dir: Path) -> None:
    """
    Extract a tar.xz archive into a folder.

    For MVTec AD, extraction produces a directory per category (bottle, cable, ...).
    """
    # Ensure output directory exists before extraction.
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open a .tar.xz archive. "r:xz" means: read mode + xz compression.
    with tarfile.open(archive, mode="r:xz") as tar:
        # Extract everything into out_dir.
        # Resulting structure (for MVTec AD) is a folder per category.
        tar.extractall(out_dir)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    # Allow overriding via environment variables so this works well in CI/Docker.
    parser.add_argument("--url", default=os.getenv("MVTEC_AD_URL", DEFAULT_URL))
    parser.add_argument("--archive", default=os.getenv("MVTEC_AD_ARCHIVE", "data/raw/mvtec_anomaly_detection.tar.xz"))
    parser.add_argument("--out", default=os.getenv("MVTEC_AD_OUT", "data/raw/mvtec_ad"))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    # Convert string paths into Path objects for safer path operations.
    archive_path = Path(args.archive)
    out_dir = Path(args.out)

    if not args.skip_download:
        if archive_path.exists():
            # If the archive is already present, we do not re-download.
            print(f"Archive already exists: {archive_path}")
        else:
            print(f"Downloading MVTec AD archive to: {archive_path}")
            print("License: CC BY-NC-SA 4.0 (non-commercial). Review terms on the official MVTec website.")
            _download(args.url, archive_path)
            # Printing the checksum helps detect corrupt downloads or mismatched files.
            print(f"SHA256: {_sha256(archive_path)}")

    if not args.skip_extract:
        if not archive_path.exists():
            raise SystemExit(f"Archive not found: {archive_path}")
        print(f"Extracting to: {out_dir}")
        _extract(archive_path, out_dir)


if __name__ == "__main__":
    main()
