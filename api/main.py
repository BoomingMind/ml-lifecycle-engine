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
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.staticfiles import StaticFiles

from defect_detection.config import load_api_config
from defect_detection.mlflow_utils import configure_mlflow, resolve_model_uri
from defect_detection.yolo import YoloPredictor

app = FastAPI(title="Manufacturing Defect Detection API")

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="ui-static")

# Prometheus metrics are process-wide singletons in a typical FastAPI deployment.
# We create them once at import time so they exist for the lifetime of the process.
REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])
PRED_COUNT = Counter("predictions_total", "Total predictions served")


@app.on_event("startup")
def _startup() -> None:
    """Initialize config and load the model once per process."""
    # Read environment-driven configuration for this process.
    cfg = load_api_config()
    # Point MLflow at the correct tracking server (if configured).
    configure_mlflow(cfg.mlflow_tracking_uri)

    # Decide which model to serve:
    # - explicit MLFLOW_MODEL_URI / MLFLOW_MODEL_NAME+stage, or
    # - fallback to MODEL_PATH weights.
    model_uri = resolve_model_uri(cfg.mlflow_model_uri)

    # Store objects on app.state so request handlers can access them without
    # re-creating them for every request.
    app.state.cfg = cfg
    if os.getenv("DISABLE_MODEL_LOAD") == "1":
        # Testing mode: keep CI fast and deterministic.
        # Instead of downloading weights or loading MLflow models, return empty predictions.
        from defect_detection.yolo import Prediction

        class _DummyPredictor:
            def predict_image_bytes(self, image_bytes: bytes) -> Prediction:
                return Prediction(boxes=[], scores=[], class_ids=[], class_names=[])

        app.state.predictor = _DummyPredictor()
    else:
        # Production mode: load the real predictor once during startup.
        app.state.predictor = YoloPredictor(model_path=cfg.model_path, mlflow_model_uri=model_uri)

@app.get("/")
def ui() -> Response:
    index = _FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not available")
    return FileResponse(index)


@app.get("/ui")
def ui_alias() -> Response:
    return ui()



@app.get("/health")
def health() -> dict:
    """Basic service health check."""


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

    # Track request latency accurately (perf_counter is monotonic).
    start = time.perf_counter()
    # We'll set this string depending on how the request finishes, so metrics match reality.
    status = "200"
    try:
        # Read the entire uploaded file into memory (works for typical image sizes).
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Run inference. This returns a structured Prediction object.
        pred = app.state.predictor.predict_image_bytes(content)
        # Count successful prediction calls.
        PRED_COUNT.inc()

        # Hashing provides a stable identifier without storing the raw image.
        image_hash = hashlib.sha256(content).hexdigest()
        # JSON payload we return to the client and also log to disk.
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "image_sha256": image_hash,
            "boxes": pred.boxes,
            "scores": pred.scores,
            "class_ids": pred.class_ids,
            "class_names": pred.class_names,
        }
        with app.state.cfg.prediction_log_path.open("a", encoding="utf-8") as f:
            # JSONL = one JSON object per line. Easy to append and easy to process later.
            f.write(json.dumps(record) + "\n")

        return record
    except HTTPException:
        # Client error (usually invalid input). Keep a 400 status label.
        status = "400"
        raise
    except Exception as e:
        # Any unexpected error becomes 500 to the client.
        status = "500"
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always record metrics, even on failures.
        elapsed = time.perf_counter() - start
        REQ_COUNT.labels(method="POST", path="/predict", status=status).inc()
        REQ_LATENCY.labels(path="/predict").observe(elapsed)
