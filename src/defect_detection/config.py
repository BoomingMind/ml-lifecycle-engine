from __future__ import annotations

"""
Configuration loading.

This repo intentionally uses environment variables for runtime configuration:
- works well in Docker/Kubernetes
- keeps secrets out of code

Training configuration is kept in params.yaml and read as plain YAML.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ApiConfig:
    """Runtime configuration needed by the FastAPI service."""

    mlflow_tracking_uri: str | None
    mlflow_model_uri: str | None
    model_path: str | None
    prediction_log_path: Path


def load_api_config() -> ApiConfig:
    """
    Load API configuration from environment variables.

    The API logs each prediction to a JSONL file for monitoring later; this also
    ensures the log directory exists before the first request arrives.
    """

    prediction_log_path = Path(os.getenv("PREDICTION_LOG_PATH", "data/predictions.jsonl"))
    prediction_log_path.parent.mkdir(parents=True, exist_ok=True)

    return ApiConfig(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        mlflow_model_uri=os.getenv("MLFLOW_MODEL_URI"),
        model_path=os.getenv("MODEL_PATH"),
        prediction_log_path=prediction_log_path,
    )


def load_params(params_path: str | Path = "params.yaml") -> dict:
    """Load params.yaml (returns {} if missing)."""
    path = Path(params_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}
