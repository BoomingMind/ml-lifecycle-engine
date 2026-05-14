from __future__ import annotations

"""
YOLOv8 training entrypoint with MLflow tracking.

What it does:
- Reads training settings from params.yaml (or CLI overrides)
- Trains Ultralytics YOLOv8
- Logs metrics/params/artifacts to MLflow
- Logs a PyFunc model so serving can load the model by URI (optional registry)

Optional (production-grade) behavior:
- Evaluate the trained model on the validation split (YOLOv8 `val`)
- Enforce a quality gate: only promote if mAP@0.5 (map50) meets a threshold
- Champion vs challenger: only replace Production if the new model is better
"""

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

from defect_detection.mlflow_models import YOLOv8PyFuncModel
from defect_detection.mlflow_utils import configure_mlflow


@dataclass(frozen=True)
class EvalMetrics:
    """
    Minimal evaluation metrics we care about for gating.

    - map50: mAP@0.5 (commonly used quick quality signal)
    - map50_95: mAP averaged across IoU thresholds 0.5..0.95 (stricter metric)
    """

    map50: float | None
    map50_95: float | None


def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict."""
    # Path() allows using the same function for string paths and Path objects.
    p = Path(path)
    # safe_load prevents executing arbitrary YAML tags.
    return yaml.safe_load(p.read_text()) or {}


def _extract_eval_metrics(result: Any) -> EvalMetrics:
    """
    Best-effort extraction of mAP metrics from Ultralytics validation results.

    Ultralytics result objects have changed across versions; this code tries
    multiple locations and returns None values when metrics are not found.
    """

    # We default to None because evaluation might not be possible (missing labels/val split).
    map50 = None
    map50_95 = None

    results_dict = getattr(result, "results_dict", None)
    if isinstance(results_dict, dict):
        # Try multiple key names to stay compatible across Ultralytics versions.
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

    box = getattr(result, "box", None)
    if box is not None:
        # Some versions store metrics on a `.box` object.
        v = getattr(box, "map50", None)
        if isinstance(v, (int, float)):
            map50 = float(v)
        v = getattr(box, "map", None)
        if isinstance(v, (int, float)):
            map50_95 = float(v)

    return EvalMetrics(map50=map50, map50_95=map50_95)


def _get_production_map50(client: MlflowClient, model_name: str) -> float | None:
    """
    Read the current Production model's map50 from the registry (if present).

    We store map50 as a model version tag so we can compare champion vs challenger
    without having to query run metrics in a specific experiment.
    """

    # Ask the registry for the most recent Production model version.
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        # No model in Production yet.
        return None
    # We store "val_map50" as a model version tag during registration.
    tag = versions[0].tags.get("val_map50")
    try:
        return float(tag) if tag is not None else None
    except Exception:
        return None


def main() -> None:
    """Train YOLOv8 and log outputs to MLflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "defect-detection"))
    parser.add_argument("--register-name", default=os.getenv("MLFLOW_MODEL_NAME"))
    parser.add_argument("--min-map50", type=float, default=float(os.getenv("MIN_MAP50", "0.85")))
    parser.add_argument("--enforce-gate", action="store_true", default=os.getenv("ENFORCE_GATE", "0") == "1")
    parser.add_argument("--promote", action="store_true", default=os.getenv("PROMOTE_MODEL", "1") == "1")
    args = parser.parse_args()

    # Load training configuration from YAML so hyperparameters are versioned in Git.
    params = _load_yaml(args.params)
    train_cfg = params.get("train", {})

    # Choose which dataset YAML to use:
    # - CLI overrides YAML (so you can test quickly without editing files)
    data_yaml = args.data or train_cfg.get("data")
    if not data_yaml:
        raise SystemExit("Missing dataset config. Provide --data or set train.data in params.yaml")

    # Configure MLflow so this run logs to the correct tracking server.
    configure_mlflow(os.getenv("MLFLOW_TRACKING_URI"))
    # Ensure experiments are grouped under a named experiment.
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        # Keep hyperparameters/data pointers with the run to make it reproducible.
        mlflow.log_params(
            {
                "model": train_cfg.get("model", "yolov8n.pt"),
                "data": data_yaml,
                "epochs": train_cfg.get("epochs", 10),
                "imgsz": train_cfg.get("imgsz", 640),
                "batch": train_cfg.get("batch", 16),
                "lr0": train_cfg.get("lr0", None),
            }
        )

        yolo = YOLO(train_cfg.get("model", "yolov8n.pt"))
        # Start training. Ultralytics writes training outputs under runs/.
        results = yolo.train(
            data=data_yaml,
            epochs=int(train_cfg.get("epochs", 10)),
            imgsz=int(train_cfg.get("imgsz", 640)),
            batch=int(train_cfg.get("batch", 16)),
            lr0=train_cfg.get("lr0", None),
        )

        # Ultralytics returns a results object that may include a dict of metrics.
        metrics = getattr(results, "results_dict", None) or {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        # Ultralytics stores outputs (including best.pt) under a run directory.
        save_dir = Path(getattr(results, "save_dir", "runs/detect/train"))
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.exists():
            raise SystemExit(f"Expected weights not found at {best_weights}")

        # Evaluate the trained weights. This is the key step that enables gating.
        # If your dataset YAML has a val split with labels, YOLO will compute mAP.
        evaluator = YOLO(str(best_weights))
        # Run validation to compute metrics like mAP. Requires labels in the dataset.
        val_result = evaluator.val(data=data_yaml, verbose=False)
        val_metrics = _extract_eval_metrics(val_result)
        # Log evaluation metrics to the MLflow run so you can compare experiments.
        mlflow.log_metrics({k: v for k, v in asdict(val_metrics).items() if isinstance(v, (int, float))})

        # Log weights file for debugging/inspection or direct weights-based serving.
        mlflow.log_artifact(str(best_weights), artifact_path="artifacts/weights")

        # Log a portable MLflow model that internally loads the weights artifact.
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=YOLOv8PyFuncModel(),
            artifacts={"weights": str(best_weights)},
            pip_requirements=[
                "mlflow==2.22.1",
                "ultralytics==8.3.157",
                "pillow==11.2.1",
                "numpy==2.2.6",
                "pandas==2.2.3",
            ],
        )

        if args.register_name:
            # Register the model so it becomes addressable as models:/name/<stage>.
            client = MlflowClient()
            # Register the PyFunc model we just logged under this run.
            # This creates a new "model version" in the MLflow Model Registry.
            # After this, you will have:
            # - a model name (args.register_name)
            # - a version number (registered.version)
            # - an optional stage assignment (Staging/Production/etc.)
            registered = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=args.register_name)

            # Store evaluation metrics on the model version for easy retrieval later.
            # Why store as tags?
            # - Tags live with the model version in the registry
            # - They are easy to query later without knowing which experiment/run produced the model
            # - They work well for simple champion-vs-challenger comparisons
            if val_metrics.map50 is not None:
                client.set_model_version_tag(args.register_name, registered.version, "val_map50", str(val_metrics.map50))
            if val_metrics.map50_95 is not None:
                client.set_model_version_tag(
                    args.register_name, registered.version, "val_map50_95", str(val_metrics.map50_95)
                )

            # Gate logic:
            # - If enforce-gate is on: fail the run if we cannot compute map50 or it's below threshold.
            # - If promote is off: stop after registering (no stage transition).
            if args.enforce_gate:
                # Gate enabled means: "do not allow low-quality models to be promoted".
                # In CI/CD, this would prevent deployment automatically.
                if val_metrics.map50 is None:
                    # If we cannot compute map50, we cannot prove quality.
                    # Common reasons:
                    # - dataset has no labels
                    # - data YAML missing a val split
                    # - evaluation crashed
                    raise SystemExit("Evaluation gate enabled but map50 could not be extracted")
                if val_metrics.map50 < args.min_map50:
                    # Failing here causes a non-zero exit, which makes pipelines fail fast.
                    raise SystemExit(f"Evaluation gate failed: map50 {val_metrics.map50:.4f} < {args.min_map50:.4f}")

            if args.promote:
                # Champion vs challenger:
                # - If Production doesn't exist yet -> promote.
                # - If we cannot compute map50 -> promote (policy choice; you can tighten this).
                # - Otherwise promote only if challenger map50 is higher than champion map50.
                #
                # Terms:
                # - Champion = current Production model
                # - Challenger = newly trained and registered model version
                prod_map50 = _get_production_map50(client, args.register_name)
                # Decision rule:
                # - If there is no current Production model: promote the challenger.
                # - If challenger has no map50: promote anyway (looser policy).
                # - Else promote only if challenger beats champion on map50.
                should_promote = prod_map50 is None or val_metrics.map50 is None or val_metrics.map50 > prod_map50
                if should_promote:
                    # Transition this version to Production.
                    # archive_existing_versions=True keeps history but ensures only one active Production model.
                    client.transition_model_version_stage(
                        name=args.register_name,
                        version=registered.version,
                        stage="Production",
                        archive_existing_versions=True,
                    )


if __name__ == "__main__":
    main()
