from __future__ import annotations

"""
Evaluate a YOLOv8 model and (optionally) enforce an evaluation gate.

Why this exists:
Training produces weights (best.pt). Before you "ship" a model, you want an
objective check that it meets a minimum quality bar. In production MLOps, this
step is often called a "gate".

This script:
- runs YOLOv8 validation (`YOLO(...).val(...)`)
- extracts mAP@0.5 (map50) and other metrics if available
- exits with a non-zero code if the gate is enabled and the threshold is not met

Notes:
- The gate threshold is `--min-map50` (default 0.85).
- If metrics cannot be extracted (dataset not labeled / val fails), the script
  exits non-zero when `--enforce-gate` is set.
"""

import argparse
from dataclasses import asdict, dataclass
from typing import Any

from ultralytics import YOLO


@dataclass(frozen=True)
class EvalMetrics:
    map50: float | None
    map50_95: float | None


def _extract_metrics(result: Any) -> EvalMetrics:
    """
    Best-effort extraction of common YOLOv8 validation metrics.

    Ultralytics has changed result object shapes across versions. This function
    tries a few known locations for mAP metrics and returns None when not found.
    """

    map50 = None
    map50_95 = None

    # Newer Ultralytics versions typically expose `results_dict`.
    results_dict = getattr(result, "results_dict", None)
    if isinstance(results_dict, dict):
        for key in ("metrics/mAP50(B)", "metrics/mAP50", "map50"):
            v = results_dict.get(key)
            if isinstance(v, (int, float)):
                map50 = float(v)
                break
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map"):
            v = results_dict.get(key)
            if isinstance(v, (int, float)):
                map50_95 = float(v)
                break

    # Some versions expose metrics under `box`.
    box = getattr(result, "box", None)
    if box is not None:
        v = getattr(box, "map50", None)
        if isinstance(v, (int, float)):
            map50 = float(v)
        v = getattr(box, "map", None)
        if isinstance(v, (int, float)):
            map50_95 = float(v)

    return EvalMetrics(map50=map50, map50_95=map50_95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 weights (e.g. best.pt)")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML path")
    parser.add_argument("--min-map50", type=float, default=0.85)
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = model.val(data=args.data, verbose=False)

    metrics = _extract_metrics(result)
    print(asdict(metrics))

    if args.enforce_gate:
        if metrics.map50 is None:
            raise SystemExit("Gate enabled but map50 could not be extracted")
        if metrics.map50 < args.min_map50:
            raise SystemExit(f"Gate failed: map50 {metrics.map50:.4f} < {args.min_map50:.4f}")


if __name__ == "__main__":
    main()

