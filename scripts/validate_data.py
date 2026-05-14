from __future__ import annotations

"""
Data validation gate for training.

This script checks the input manifest (CSV) before training:
- schema expectations (Great Expectations)
- referenced files exist on disk

If validation succeeds, it writes an "ok" marker file (reports/validation.ok).
That marker can be used by pipeline tools to treat validation as a formal stage.
"""

import argparse
from pathlib import Path

import pandas as pd


def _file_exists(path: str) -> bool:
    """Best-effort existence check that never raises."""
    try:
        return Path(path).expanduser().exists()
    except Exception:
        return False


def _validate_with_ge(df: pd.DataFrame) -> None:
    """Run Great Expectations checks on the manifest dataframe."""
    import great_expectations as gx

    validator = gx.from_pandas(df)
    validator.expect_column_to_exist("image_path")
    validator.expect_column_values_to_not_be_null("image_path")
    validator.expect_column_values_to_be_of_type("image_path", "str")
    result = validator.validate()
    if not result.success:
        raise SystemExit("Great Expectations validation failed")


def main() -> None:
    """Validate manifest schema + file paths, then write a success marker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--ok-out", default="reports/validation.ok")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "image_path" not in df.columns:
        raise SystemExit("manifest.csv must include an 'image_path' column")

    _validate_with_ge(df)

    missing_images = df.loc[~df["image_path"].astype(str).map(_file_exists), "image_path"].tolist()
    if missing_images:
        raise SystemExit(f"Missing image files: {missing_images[:10]}")

    if "label_path" in df.columns:
        missing_labels = df.loc[~df["label_path"].astype(str).map(_file_exists), "label_path"].tolist()
        if missing_labels:
            raise SystemExit(f"Missing label files: {missing_labels[:10]}")

    ok_path = Path(args.ok_out)
    ok_path.parent.mkdir(parents=True, exist_ok=True)
    ok_path.write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
