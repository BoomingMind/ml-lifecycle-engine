# DefectGuard: Manufacturing Defect Detection MLOps Platform

DefectGuard is a computer vision MLOps system for automated manufacturing defect detection.

It combines model training, experiment tracking, model packaging, API serving, browser-based inspection, monitoring, orchestration, and local deployment into one end-to-end system.


## Platform Simulation

 capture -> validate -> train -> promote -> serve -> inspect -> monitor.

![DefectGuard workflow simulation](docs/assets/defectguard-workflow.gif)

## Executive Summary

- Uses YOLOv8 for visual defect detection on manufacturing-style image data
- Tracks experiments, artifacts, and model versions with MLflow
- Validates input data before training with Great Expectations
- Serves predictions through a FastAPI inference API and lightweight frontend
- Logs predictions for monitoring and drift analysis
- Generates drift reports with Evidently
- Orchestrates workflows with Prefect
- Reproduces pipeline stages with DVC
- Ships with Docker Compose, Nginx, Prometheus, and Grafana for local platform operations
- Includes automated tests and GitHub Actions CI

## Why This Project Matters

In real manufacturing environments, visual inspection systems need more than a trained model.

They also need:

- reproducible training
- model versioning
- safe promotion rules
- reliable serving
- operational visibility
- monitoring for changing behavior over time

DefectGuard demonstrates that full lifecycle.

It is a strong portfolio project for roles involving:

- MLOps
- machine learning engineering
- computer vision platforms
- production AI systems

## System Scope

This repository covers the following lifecycle:

1. Download or prepare dataset inputs
2. Validate dataset structure before training
3. Train YOLOv8 and log experiments to MLflow
4. Evaluate and optionally gate model promotion
5. Register models in MLflow Model Registry
6. Serve predictions through FastAPI
7. Visualize results in a browser UI
8. Log predictions to JSONL for monitoring
9. Generate drift reports from prediction behavior
10. Run services locally with Docker Compose
11. Monitor runtime metrics with Prometheus and Grafana
12. Protect quality with pytest and GitHub Actions CI

## Architecture

At a high level, the platform is organized into five layers:

- Data layer: dataset download, manifest validation, dataset config, DVC stages
- Training layer: YOLOv8 training, MLflow tracking, evaluation, registry packaging
- Serving layer: FastAPI API, prediction abstraction, browser frontend
- Monitoring layer: JSONL prediction logs, Evidently drift reports, Prometheus metrics, Grafana dashboards
- Operations layer: Prefect flows, Docker images, Docker Compose stack, Nginx reverse proxy, CI

## Core Capabilities

### Training And Registry

- YOLOv8 training via `scripts/train.py`
- MLflow experiment logging for params, metrics, and artifacts
- MLflow PyFunc packaging for standardized model serving
- Optional registry promotion using model version tags
- Quality gate based on `mAP@0.5`
- Champion-vs-challenger promotion logic for Production stage decisions

### Serving And Product UX

- FastAPI inference service with `/predict`, `/health`, `/ready`, `/version`, and `/metrics`
- Optional API key protection using `X-API-Key`
- Request IDs, structured logging, security headers, CORS, trusted hosts, and gzip support
- Browser UI for image upload, prediction visualization, and raw JSON inspection

### Monitoring And Reliability

- Prediction log capture in JSONL format
- Reference-vs-current drift reporting with Evidently
- Prometheus service metrics
- Grafana datasource provisioning
- Local restart policies and healthchecks in Docker Compose
- Automated API tests with lightweight dummy predictor mode

## Technology Stack

### ML And Data

- YOLOv8 / Ultralytics
- NumPy
- Pandas
- Pillow
- Great Expectations
- DVC

### MLOps And Model Lifecycle

- MLflow Tracking
- MLflow PyFunc
- MLflow Model Registry
- Prefect
- Evidently

### Serving And Platform

- FastAPI
- Uvicorn
- Nginx
- Docker
- Docker Compose

### Monitoring And Quality

- Prometheus
- Grafana
- pytest
- Ruff
- GitHub Actions

## Repository Layout

- `api/`: FastAPI service and frontend UI
- `src/defect_detection/`: reusable application and model helper modules
- `scripts/`: training, validation, dataset, and monitoring scripts
- `pipelines/`: Prefect workflow definitions
- `monitoring/`: Prometheus and Grafana provisioning config
- `docker/`: Docker build files
- `tests/`: automated API test coverage
- `docs/`: deep documentation and codebase walkthroughs
- `data/`: dataset config, manifests, and runtime log files

## Documentation Map

For different reading styles:

- [MLOPS_FOR_BEGINNERS.md](docs/MLOPS_FOR_BEGINNERS.md): dedicated beginner guide to MLOps concepts, tools, files, and important functions used in this project
- [PROJECT_BOOK.md](docs/PROJECT_BOOK.md): full end-to-end handbook, architecture deep dives, and 7-day study plan
- [WALKTHROUGH.md](docs/WALKTHROUGH.md): shorter guided codebase tour
- [README.md](README.md): professional project overview and operational quickstart

The GIF asset in this README is generated from:

- `scripts/generate_demo_gif.py`

## Getting Started

### Prerequisites

- Python 3.11
- `pip`
- Docker and Docker Compose for containerized local runs

### Install Dependencies

```bash
pip3 install -r requirements.txt -r requirements-mlops.txt -r requirements-dev.txt
```

## Quickstart: Local API Run

Set the project import path and point the API to a local YOLO weights file:

```bash
export PYTHONPATH=src
export MODEL_PATH=/absolute/path/to/best.pt
uvicorn api.main:app --reload
```

Open:

- UI: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

Useful checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/metrics
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/image.jpg"
```

## Quickstart: Full Local Platform

Start the full stack:

```bash
docker compose up --build
```

Service endpoints:

- Nginx entrypoint and UI: `http://127.0.0.1:8080`
- API direct access: `http://127.0.0.1:8000`
- MLflow UI: `http://127.0.0.1:5000`
- Prometheus UI: `http://127.0.0.1:9090`
- Grafana UI: `http://127.0.0.1:3000` using `admin/admin`

## Dataset Setup

Download and extract the dataset:

```bash
python3 scripts/download_mvtec_ad.py --out data/raw/mvtec_ad
```

Important note:

- MVTec AD is released under `CC BY-NC-SA 4.0`
- review license terms before any non-demo use

## Training

Training expects a YOLO dataset YAML. A placeholder example is available at `data/dataset.yaml`.

Run training with MLflow tracking:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=defect-detection
export MLFLOW_MODEL_NAME=defect-yolo
PYTHONPATH=src python3 scripts/train.py --data data/dataset.yaml
```

Enable a production-style quality gate:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_MODEL_NAME=defect-yolo
export ENFORCE_GATE=1
export MIN_MAP50=0.85
export PROMOTE_MODEL=1
PYTHONPATH=src python3 scripts/train.py --data data/dataset.yaml
```

## Data Validation

Validate the training manifest before model training:

```bash
PYTHONPATH=src python3 scripts/validate_data.py --manifest data/manifest.csv
```

What this checks:

- required manifest schema
- null safety for image paths
- file existence on disk
- optional label file presence

## DVC Pipeline

The repository defines a simple reproducible DVC pipeline in `dvc.yaml`.

Current stages:

- `download_mvtec_ad`
- `validate`
- `train`

Run the pipeline:

```bash
dvc repro
```

## Monitoring And Drift

Runtime and model monitoring are both included.

### Service Monitoring

- Prometheus scrapes `/metrics`
- Grafana reads from Prometheus
- FastAPI exposes request count and latency metrics

### Behavior Monitoring

- the API logs predictions to `data/predictions.jsonl`
- `scripts/set_reference_predictions.py` creates a baseline snapshot
- `scripts/drift_report.py` compares baseline vs current behavior

Generate a drift report:

```bash
python3 scripts/set_reference_predictions.py
python3 scripts/drift_report.py
```

## Workflow Orchestration

Prefect flows are provided for:

- retraining flow: validation -> training
- monitoring flow: reference snapshot -> drift report

Run the default flow entrypoint:

```bash
python3 pipelines/prefect_flow.py
```

## Testing And CI

Run local quality checks:

```bash
ruff check .
pytest
```

CI is configured in [ci.yml](.github/workflows/ci.yml) and runs on:

- `push`
- `pull_request`

The test suite uses `DISABLE_MODEL_LOAD=1` so CI stays fast and does not require real model weights.

## Key Environment Variables

### Serving

- `MODEL_PATH`: local YOLO weights path
- `MLFLOW_TRACKING_URI`: MLflow tracking backend
- `MLFLOW_MODEL_URI`: explicit model URI for serving
- `MLFLOW_MODEL_NAME`: registry model name for stage-based serving
- `MLFLOW_MODEL_STAGE`: registry stage, default `Production`
- `PREDICTION_LOG_PATH`: JSONL prediction log output path
- `API_KEY`: enables `X-API-Key` protection for `/predict` and `/metrics`
- `MAX_UPLOAD_MB`: upload size limit
- `LOG_LEVEL`: service logging level
- `DEBUG`: detailed internal errors when set to `1`
- `CORS_ORIGINS`: comma-separated CORS origins
- `ALLOWED_HOSTS`: comma-separated trusted hosts
- `DISABLE_MODEL_LOAD`: enables dummy predictor mode for tests and CI

### Training And Gating

- `MLFLOW_EXPERIMENT_NAME`
- `MIN_MAP50`
- `ENFORCE_GATE`
- `PROMOTE_MODEL`

### Dataset Helper

- `MVTEC_AD_URL`
- `MVTEC_AD_ARCHIVE`
- `MVTEC_AD_OUT`

## Production-Shaped Design Choices

This repository intentionally includes patterns commonly expected in real ML systems:

- structured JSON logging
- request tracing with request IDs
- readiness and health endpoints
- model registry integration
- evaluation gates
- promotion policies
- reverse proxy entrypoint
- metrics and dashboards
- automated tests and CI
- documented configuration

## Current Outputs

During normal usage, the platform creates outputs such as:

- `runs/` from YOLO training
- `data/predictions.jsonl`
- `data/reference_predictions.jsonl`
- `reports/validation.ok`
- `reports/drift_report.html`
- MLflow run and model artifacts

## Limitations And Next Steps

This is a strong production-shaped portfolio project, but a real industrial rollout may also require:

- persistent external databases and object storage
- role-based access control
- secret management
- alerting policies
- larger-scale dataset pipelines
- label-aware production monitoring
- GPU scheduling and infrastructure tuning
- Kubernetes or infrastructure-as-code deployment layers

## Usage Notes

If you use the MVTec AD helper flow, make sure dataset usage follows the original dataset license.
