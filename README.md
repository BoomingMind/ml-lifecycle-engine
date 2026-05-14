# Manufacturing Defect Detection Pipeline

This repository is a minimal, production-shaped skeleton for a manufacturing defect detection system:

- Train a YOLOv8 object detector and track everything in MLflow
- Validate input data before training (Great Expectations)
- Serve predictions over a REST API (FastAPI)
- Log predictions for monitoring
- Produce a drift report comparing “reference” vs “current” prediction behavior (Evidently)
- Run the above as an orchestrated flow (Prefect)
- Run MLflow + API locally with Docker Compose

If you’re new to these tools, start with: `docs/WALKTHROUGH.md`.

## Quickstart (Local)

Install dependencies:

```bash
pip3 install -r requirements.txt -r requirements-mlops.txt
```

Start the API locally (without Docker):

```bash
export PYTHONPATH=src
export MODEL_PATH=/absolute/path/to/your/best.pt
uvicorn api.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction (image upload):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/image.jpg"
```

Metrics (Prometheus format):

```bash
curl http://127.0.0.1:8000/metrics
```

## Quickstart (Docker Compose)

```bash
docker compose up --build
```

Services:

- API: http://127.0.0.1:8000
- MLflow: http://127.0.0.1:5000

## Training

Training needs a YOLO dataset YAML (example placeholder exists at `data/dataset.yaml`).

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=defect-detection
export MLFLOW_MODEL_NAME=defect-yolo
PYTHONPATH=src python3 scripts/train.py --data data/dataset.yaml
```

## DVC Pipeline

`dvc.yaml` wires two stages:

- `validate`: validate `data/manifest.csv`
- `train`: train YOLOv8 using `params.yaml`

## Environment Variables

API:

- `MODEL_PATH`: path to YOLOv8 weights (best.pt)
- `MLFLOW_TRACKING_URI`: MLflow tracking server
- `MLFLOW_MODEL_URI`: model URI for serving (e.g. `models:/defect-yolo/Production`)
- `PREDICTION_LOG_PATH`: where to write JSONL prediction logs

