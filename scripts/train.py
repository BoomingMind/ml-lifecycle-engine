from __future__ import annotations

"""
YOLOv8 training entrypoint with MLflow tracking.

What it does:
- Reads training settings from params.yaml (or CLI overrides)
- Trains Ultralytics YOLOv8
- Logs metrics/params/artifacts to MLflow
- Logs a PyFunc model so serving can load the model by URI (optional registry)
"""

import argparse
import os
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO

from defect_detection.mlflow_models import YOLOv8PyFuncModel
from defect_detection.mlflow_utils import configure_mlflow


def _load_yaml(path: str | Path) -> dict:
    """Load a YAML file into a dict."""
    p = Path(path)
    return yaml.safe_load(p.read_text()) or {}


def main() -> None:
    """Train YOLOv8 and log outputs to MLflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "defect-detection"))
    parser.add_argument("--register-name", default=os.getenv("MLFLOW_MODEL_NAME"))
    args = parser.parse_args()

    params = _load_yaml(args.params)
    train_cfg = params.get("train", {})

    data_yaml = args.data or train_cfg.get("data")
    if not data_yaml:
        raise SystemExit("Missing dataset config. Provide --data or set train.data in params.yaml")

    configure_mlflow(os.getenv("MLFLOW_TRACKING_URI"))
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

        save_dir = Path(getattr(results, "save_dir", "runs/detect/train"))
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.exists():
            raise SystemExit(f"Expected weights not found at {best_weights}")

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
            mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=args.register_name)


if __name__ == "__main__":
    main()
