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
        # This makes all subsequent mlflow.* calls log to this server.
        mlflow.set_tracking_uri(tracking_uri)


def resolve_model_uri(explicit_model_uri: str | None = None) -> str | None:
    """
    Decide which MLflow model URI to serve.

    Priority:
    1) explicit_model_uri argument
    2) MLFLOW_MODEL_URI environment variable
    3) MLFLOW_MODEL_NAME + MLFLOW_MODEL_STAGE (defaults to Production)
    """

    # 1) Caller provided an explicit URI (highest priority).
    if explicit_model_uri:
        return explicit_model_uri
    # 2) Environment variable can directly point at a run or registry model.
    env_uri = os.getenv("MLFLOW_MODEL_URI")
    if env_uri:
        return env_uri
    # 3) If only a name is provided, assume registry usage and choose a stage.
    name = os.getenv("MLFLOW_MODEL_NAME")
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    if not name:
        # No way to resolve a model, caller must fall back to weights-based serving.
        return None
    # Registry URI format: models:/<name>/<stage>
    return f"models:/{name}/{stage}"


def promote_best_to_production(model_name: str, run_id: str, model_artifact_path: str = "model") -> str:
    """
    Register a run’s model and promote it to Production.

    This is a convenience helper for a typical workflow:
    - train a run
    - register the logged MLflow model from that run
    - set that version to Production (archiving older ones)
    """

    # MlflowClient is the registry API (stages, versions, transitions, etc.).
    client = MlflowClient()
    # Register the model artifact that was logged under the run.
    registered = mlflow.register_model(model_uri=f"runs:/{run_id}/{model_artifact_path}", name=model_name)
    # Move that registered version to Production, and archive any previous Production versions.
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )
    # Return a convenient URI that can be used for serving.
    return f"models:/{model_name}/Production"
