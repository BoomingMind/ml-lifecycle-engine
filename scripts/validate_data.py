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
        # Expand "~" so user home paths work, then check file existence.
        return Path(path).expanduser().exists()
    except Exception:
        # Never raise from validation helper: treat unknown paths as "missing".
        return False


def _validate_with_ge(df: pd.DataFrame) -> None:
    """Run Great Expectations checks on the manifest dataframe."""
    import great_expectations as gx

    # Create a Great Expectations validator from a pandas DataFrame.
    validator = gx.from_pandas(df)
    # Minimum schema checks. You can add more expectations as the project matures
    # (image size distribution, allowed formats, label cardinality, etc.).
    validator.expect_column_to_exist("image_path")
    validator.expect_column_values_to_not_be_null("image_path")
    validator.expect_column_values_to_be_of_type("image_path", "str")
    # Execute the expectations.
    result = validator.validate()
    if not result.success:
        raise SystemExit("Great Expectations validation failed")


def main() -> None:
    """Validate manifest schema + file paths, then write a success marker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--ok-out", default="reports/validation.ok")
    args = parser.parse_args()

    # Load the manifest CSV describing where images (and optionally labels) live.
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    # Read CSV into a DataFrame for schema validation and path checks.
    df = pd.read_csv(manifest_path)
    if "image_path" not in df.columns:
        raise SystemExit("manifest.csv must include an 'image_path' column")

    # First: schema validation (fast and catches common mistakes).
    _validate_with_ge(df)

    # Second: file existence validation (catches broken paths early).
    missing_images = df.loc[~df["image_path"].astype(str).map(_file_exists), "image_path"].tolist()
    if missing_images:
        raise SystemExit(f"Missing image files: {missing_images[:10]}")

    if "label_path" in df.columns:
        # Label paths are optional because some workflows are unsupervised or weakly supervised.
        missing_labels = df.loc[~df["label_path"].astype(str).map(_file_exists), "label_path"].tolist()
        if missing_labels:
            raise SystemExit(f"Missing label files: {missing_labels[:10]}")

    # Success marker file so pipelines can treat validation as a formal stage.
    ok_path = Path(args.ok_out)
    ok_path.parent.mkdir(parents=True, exist_ok=True)
    ok_path.write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
