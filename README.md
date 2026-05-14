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
pip3 install -r requirements-dev.txt
```

Start the API locally (without Docker):

```bash
export PYTHONPATH=src
export MODEL_PATH=/absolute/path/to/your/best.pt
uvicorn api.main:app --reload
```

UI (frontend):

- Open: http://127.0.0.1:8000/ (upload image and visualize boxes)
- API docs: http://127.0.0.1:8000/docs

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Readiness (checks model loaded):

```bash
curl http://127.0.0.1:8000/ready
```

Prediction (image upload):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/image.jpg"
```

Metrics (Prometheus format):

```bash
curl http://127.0.0.1:8000/metrics
```

## Dataset (MVTec AD)

Download and extract the MVTec AD dataset into `data/raw/mvtec_ad`:

```bash
python3 scripts/download_mvtec_ad.py --out data/raw/mvtec_ad
```

## Quickstart (Docker Compose)

```bash
docker compose up --build
```

Services:

- Nginx (entrypoint + UI): http://127.0.0.1:8080
- API (direct): http://127.0.0.1:8000
- MLflow UI: http://127.0.0.1:5000
- Prometheus UI: http://127.0.0.1:9090
- Grafana UI: http://127.0.0.1:3000 (admin/admin)

## Training

Training needs a YOLO dataset YAML (example placeholder exists at `data/dataset.yaml`).

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=defect-detection
export MLFLOW_MODEL_NAME=defect-yolo
PYTHONPATH=src python3 scripts/train.py --data data/dataset.yaml
```

Quality gate (production-style):

```bash
export MLFLOW_MODEL_NAME=defect-yolo
export ENFORCE_GATE=1
export MIN_MAP50=0.85
export PROMOTE_MODEL=1
PYTHONPATH=src python3 scripts/train.py --data data/dataset.yaml
```

## DVC Pipeline

`dvc.yaml` wires stages:

- `download_mvtec_ad`: download/extract MVTec AD
- `validate`: validate `data/manifest.csv`
- `train`: train YOLOv8 using `params.yaml`

## Environment Variables

API:

- `MODEL_PATH`: path to YOLOv8 weights (best.pt)
- `MLFLOW_TRACKING_URI`: MLflow tracking server
- `MLFLOW_MODEL_URI`: model URI for serving (e.g. `models:/defect-yolo/Production`)
- `PREDICTION_LOG_PATH`: where to write JSONL prediction logs
- `DISABLE_MODEL_LOAD`: set to `1` to skip loading a real model (used by CI tests)
- `API_KEY`: if set, requires `X-API-Key` header for `/predict` and `/metrics`
- `MAX_UPLOAD_MB`: max upload size for `/predict` (default: 10)
- `LOG_LEVEL`: logging level (default: INFO)
- `DEBUG`: set to `1` to return detailed 500 errors (default: 0)
- `CORS_ORIGINS`: comma-separated origins for CORS (default: `*`)
- `ALLOWED_HOSTS`: comma-separated trusted hostnames (default: `*`)
