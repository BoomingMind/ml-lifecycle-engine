# Walkthrough: Learn The Codebase By Reading

This doc explains the repository like you’re new and want to understand "why" each thing exists.

If you want a separate beginner-first MLOps explanation of the tools, files, and important functions used in this repo, read `docs/MLOPS_FOR_BEGINNERS.md`.

If you want the full book-style explanation from Phase 1 foundation through training,
serving, monitoring, deployment, and operations, read `docs/PROJECT_BOOK.md` first.

## Big Picture

The whole system is built around one simple loop:

1. Collect images (and labels if training).
2. Validate the dataset format before training.
3. Train a model and track the run (metrics, params, artifacts).
4. Serve that model via an API.
5. Log predictions so we can monitor drift later.
6. Generate monitoring reports from those logs.

Everything in the repo supports one of those steps.

## Folder Layout

- `src/defect_detection/`: reusable Python code (library code)
- `api/`: FastAPI application (serving)
- `scripts/`: runnable scripts (train, validate, drift report)
- `pipelines/`: orchestration (Prefect flows)
- `data/`: placeholder dataset configs/manifests
- `docker/` + `docker-compose.yml`: local infrastructure (API + MLflow)
- `dvc.yaml` + `params.yaml`: “pipeline as code” and parameters (DVC)

## How A Prediction Flows Through The System

1. You send an image to `POST /predict`.
2. The API reads bytes and passes them into the model wrapper (`YoloPredictor`).
3. The model returns bounding boxes and scores.
4. The API writes one JSON line to `data/predictions.jsonl`.
5. Monitoring scripts can later compare `predictions.jsonl` against a saved “reference” file.

## Core Library Code (`src/defect_detection`)

### `config.py`

File: `src/defect_detection/config.py`

Goal: keep configuration in one place and load it from environment variables.

- `ApiConfig`: a small container for the API config values.
- `load_api_config()`:
  - Reads environment variables like `MODEL_PATH`, `MLFLOW_TRACKING_URI`, etc.
  - Creates the parent directory for `PREDICTION_LOG_PATH` so the API can write logs.
- `load_params()`:
  - Reads `params.yaml` for training configuration.
  - Returns `{}` if the file doesn’t exist, which keeps scripts simple.

### `mlflow_utils.py`

File: `src/defect_detection/mlflow_utils.py`

Goal: small helpers so the rest of the code doesn’t repeat MLflow boilerplate.

- `configure_mlflow(tracking_uri)`:
  - If you pass a tracking URI, it sets it globally for the process.
- `resolve_model_uri(explicit_model_uri=None)`:
  - Serving needs a “model location”.
  - Priority:
    1) explicit argument
    2) `MLFLOW_MODEL_URI`
    3) `MLFLOW_MODEL_NAME` + `MLFLOW_MODEL_STAGE` (defaults to `Production`)
  - Produces a URI like: `models:/defect-yolo/Production`
- `promote_best_to_production(...)`:
  - Registers a model from a run and moves that version to the Production stage.

### `mlflow_models.py`

File: `src/defect_detection/mlflow_models.py`

Goal: wrap YOLOv8 weights into a standard MLflow “PyFunc” model so it can be loaded consistently.

What is a PyFunc model?

- Think of it as a standard interface: it takes a table-like input (a pandas DataFrame)
  and returns JSON-like output.

Key parts:

- `YOLOv8PyFuncModel.load_context(...)`:
  - MLflow gives you artifacts (files) attached to the model.
  - This code loads YOLO weights from `context.artifacts["weights"]`.
- `predict(...)`:
  - Accepts a DataFrame with either:
    - `image_b64` (base64 encoded bytes), or
    - `image_path` (path on disk)
  - Runs YOLO, then formats results into a clean dict:
    - `boxes`: [x1, y1, x2, y2] per detection
    - `scores`: confidence per detection
    - `class_ids`: integer label per detection
    - `class_names`: readable label names per detection

### `yolo.py`

File: `src/defect_detection/yolo.py`

Goal: the API needs ONE simple interface for prediction regardless of whether the model
comes from a local weights file or MLflow.

Key types:

- `Prediction`:
  - A structured representation of YOLO output.

- `YoloPredictor`:
  - Constructor chooses how to load:
    - If `mlflow_model_uri` is provided → load model using `mlflow.pyfunc.load_model`
    - Else if `model_path` is provided → load weights using `ultralytics.YOLO(model_path)`
  - `predict_image_bytes(image_bytes)`:
    - MLflow branch:
      - Base64-encode the bytes, build a DataFrame, call the PyFunc model
      - Convert returned dict into `Prediction`
    - Weights branch:
      - Decode bytes to an RGB image (PIL), convert to numpy, run YOLO
      - Format output into `Prediction`

This design keeps the API code simple: it doesn’t care how the model was loaded.

## Serving (`api/main.py`)

File: `api/main.py`

This file is the REST service.

Startup:

- Loads config (`load_api_config`)
- Configures MLflow tracking URI
- Resolves which model to use:
  - If you set `MLFLOW_MODEL_URI`, it loads from MLflow
  - Otherwise it can load from `MODEL_PATH`
- Creates a single `YoloPredictor` and stores it on `app.state`

Endpoints:

- `GET /health`:
  - Simple readiness check.
- `GET /metrics`:
  - Prometheus scrape endpoint.
  - Exposes counters/histograms for basic observability.
- `POST /predict`:
  - Reads image bytes
  - Calls predictor
  - Logs JSONL record (timestamp + hash + boxes/scores/classes)
  - Returns the same record as the response

Why JSONL?

- It’s easy to append one line per request.
- It’s easy to parse later for monitoring jobs.

## Scripts (`scripts/`)

### `validate_data.py`

Validates `data/manifest.csv`:

- Requires `image_path` column and checks it isn’t null.
- Checks file paths exist on disk.
- Writes `reports/validation.ok` when validation passes.

This is a “gate” step so you don’t waste GPU time on broken input data.

### `train.py`

Trains YOLOv8 and logs to MLflow:

- Reads config from `params.yaml` and/or `--data`
- Starts an MLflow run
- Logs key params (epochs, imgsz, etc.)
- Runs `ultralytics.YOLO(...).train(...)`
- Locates the resulting `best.pt`
- Logs:
  - weights as an artifact
  - a PyFunc model so serving can be uniform
- Optionally registers the model if `MLFLOW_MODEL_NAME` is set

### `set_reference_predictions.py`

Copies `data/predictions.jsonl` to `data/reference_predictions.jsonl`.

This is how you define “what normal looks like” after a stable period.

### `drift_report.py`

Builds a simple drift report using prediction logs:

- Reads JSONL logs into DataFrames
- Derives simple features:
  - number of detected boxes
  - max confidence
  - mean confidence
- Runs Evidently’s `DataDriftPreset` and writes an HTML report.

Important note:

This is monitoring for “behavior drift” (output statistics), not ground-truth accuracy.

## Orchestration (`pipelines/prefect_flow.py`)

Prefect gives you a way to schedule and track jobs:

- `retraining_flow`: validate → train
- `monitoring_flow`: set_reference → drift_report

The flow runs scripts as subprocesses with `PYTHONPATH=src` so imports work consistently.

## DVC (`dvc.yaml`, `params.yaml`)

DVC is “makefiles for data + ML”.

- `dvc.yaml` describes pipeline stages and dependencies.
- `params.yaml` is the tunable configuration tracked by DVC.

When you later add real data versioning, DVC can track datasets and reproduce runs.

## Docker Compose

`docker-compose.yml` starts:

- `mlflow`: MLflow tracking server (SQLite backend for local use)
- `api`: FastAPI server

You can choose whether the API loads a model from:

- `MODEL_PATH` (weights on disk)
- `MLFLOW_MODEL_URI` (registry/run reference)
