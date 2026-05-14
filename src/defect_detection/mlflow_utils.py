from __future__ import annotations

"""
Small MLflow helpers used by both training and serving.

MLflow has a few concepts that appear repeatedly in an MLOps codebase:
- tracking URI: where metrics/params/artifacts are logged
- model URIs: how a model is referenced (runs:/..., models:/...)
- model registry stages: e.g. Staging, Production
"""

import os

import mlflow
from mlflow.tracking import MlflowClient


def configure_mlflow(tracking_uri: str | None) -> None:
    """Set the MLflow tracking URI for the current process (if provided)."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def resolve_model_uri(explicit_model_uri: str | None = None) -> str | None:
    """
    Decide which MLflow model URI to serve.

    Priority:
    1) explicit_model_uri argument
    2) MLFLOW_MODEL_URI environment variable
    3) MLFLOW_MODEL_NAME + MLFLOW_MODEL_STAGE (defaults to Production)
    """

    if explicit_model_uri:
        return explicit_model_uri
    env_uri = os.getenv("MLFLOW_MODEL_URI")
    if env_uri:
        return env_uri
    name = os.getenv("MLFLOW_MODEL_NAME")
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    if not name:
        return None
    return f"models:/{name}/{stage}"


def promote_best_to_production(model_name: str, run_id: str, model_artifact_path: str = "model") -> str:
    """
    Register a run’s model and promote it to Production.

    This is a convenience helper for a typical workflow:
    - train a run
    - register the logged MLflow model from that run
    - set that version to Production (archiving older ones)
    """

    client = MlflowClient()
    registered = mlflow.register_model(model_uri=f"runs:/{run_id}/{model_artifact_path}", name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    return f"models:/{model_name}/Production"
