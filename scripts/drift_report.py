from __future__ import annotations

"""
Prediction drift report.

This script compares two JSONL prediction logs:
- reference: "what normal looked like" (baseline period)
- current: recent predictions

Because we may not have ground-truth labels in production, this focuses on simple
behavioral features derived from predictions (counts and confidence stats).
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_jsonl(path: Path) -> pd.DataFrame:
    """Read a JSONL file (one JSON object per line) into a pandas DataFrame."""
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _explode_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive numeric monitoring features from raw prediction records.

    Each prediction contains variable-length arrays (boxes/scores). Drift tools
    work best on fixed columns, so we compute a few simple aggregates.
    """

    if df.empty:
        return df
    df = df.copy()
    df["n_boxes"] = df["boxes"].map(lambda v: len(v) if isinstance(v, list) else 0)
    df["max_score"] = df["scores"].map(lambda v: max(v) if isinstance(v, list) and v else 0.0)
    df["mean_score"] = df["scores"].map(lambda v: float(sum(v) / len(v)) if isinstance(v, list) and v else 0.0)
    return df[["ts", "image_sha256", "n_boxes", "max_score", "mean_score"]]


def main() -> None:
    """Generate an Evidently HTML drift report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/reference_predictions.jsonl")
    parser.add_argument("--current", default="data/predictions.jsonl")
    parser.add_argument("--out", default="reports/drift_report.html")
    args = parser.parse_args()

    ref_df = _explode_predictions(_read_jsonl(Path(args.reference)))
    cur_df = _explode_predictions(_read_jsonl(Path(args.current)))

    if ref_df.empty or cur_df.empty:
        raise SystemExit("Need both reference and current prediction logs to compute drift")

    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))


if __name__ == "__main__":
    main()
