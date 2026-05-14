from __future__ import annotations

"""
FastAPI inference service.

Responsibilities:
- Load a YOLOv8 detector once at startup (either from MLflow or local weights)
- Expose an HTTP API:
  - /health: readiness
  - /predict: run inference on an uploaded image
  - /metrics: Prometheus scrape endpoint
- Append prediction logs as JSONL for monitoring/drift analysis
"""

import hashlib
import json
import time
from datetime import datetime, timezone

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from defect_detection.config import load_api_config
from defect_detection.mlflow_utils import configure_mlflow, resolve_model_uri
from defect_detection.yolo import YoloPredictor

app = FastAPI(title="Manufacturing Defect Detection API")

# Prometheus metrics are process-wide singletons in a typical FastAPI deployment.
REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])
PRED_COUNT = Counter("predictions_total", "Total predictions served")


@app.on_event("startup")
def _startup() -> None:
    """Initialize config and load the model once per process."""
    cfg = load_api_config()
    configure_mlflow(cfg.mlflow_tracking_uri)

    model_uri = resolve_model_uri(cfg.mlflow_model_uri)
    app.state.cfg = cfg
    app.state.predictor = YoloPredictor(model_path=cfg.model_path, mlflow_model_uri=model_uri)


@app.get("/health")
def health() -> dict:
    """Basic service health check."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Run object detection on an uploaded image.

    Returns a JSON response including:
    - image hash (sha256)
    - boxes/scores/classes from YOLO
    Also appends the same payload as one line in the JSONL prediction log.
    """

    start = time.perf_counter()
    status = "200"
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        pred = app.state.predictor.predict_image_bytes(content)
        PRED_COUNT.inc()

        # Hashing provides a stable identifier without storing the raw image.
        image_hash = hashlib.sha256(content).hexdigest()
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "image_sha256": image_hash,
            "boxes": pred.boxes,
            "scores": pred.scores,
            "class_ids": pred.class_ids,
            "class_names": pred.class_names,
        }
        with app.state.cfg.prediction_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record
    except HTTPException:
        status = "400"
        raise
    except Exception as e:
        status = "500"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        elapsed = time.perf_counter() - start
        REQ_COUNT.labels(method="POST", path="/predict", status=status).inc()
        REQ_LATENCY.labels(path="/predict").observe(elapsed)
