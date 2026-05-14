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
import logging
from pathlib import Path

import uuid
import contextvars
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from fastapi import Header
from starlette.staticfiles import StaticFiles

from defect_detection.config import load_api_config
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from defect_detection.mlflow_utils import configure_mlflow, resolve_model_uri
from defect_detection.yolo import YoloPredictor

app = FastAPI(title="Manufacturing Defect Detection API")

REQUEST_ID = contextvars.ContextVar("request_id", default=None)

_FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="ui-static")

# Prometheus metrics are process-wide singletons in a typical FastAPI deployment.
# We create them once at import time so they exist for the lifetime of the process.
REQ_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "path", "status"])
# Prometheus metrics are process-wide singletons in a typical FastAPI deployment.
# We create them once at import time so they exist for the lifetime of the process.
REQ_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["path"])
PRED_COUNT = Counter("predictions_total", "Total predictions served")


LOG = logging.getLogger("defectguard")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default).strip()
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _configure_http_middleware() -> None:
    """Register request middlewares before the ASGI app starts."""
    allowed_hosts = _env_csv("ALLOWED_HOSTS", "*")
    if allowed_hosts and allowed_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    cors_origins = _env_csv("CORS_ORIGINS", "*")
    if cors_origins:
        allow_all = cors_origins == ["*"]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins if not allow_all else ["*"],
            allow_credentials=not allow_all,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(GZipMiddleware, minimum_size=1024)


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(message)s")


def _json_log(event: str, **fields: object) -> None:
    payload = {"event": event, "ts": datetime.now(timezone.utc).isoformat(), **fields}
    LOG.info(json.dumps(payload, default=str))


@app.middleware("http")
async def _request_context(request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    REQUEST_ID.set(request_id)
    start = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response
    finally:
        elapsed = time.perf_counter() - start
        status_code = response.status_code if response is not None else 500
        _json_log(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=str(request.url.path),
            status=int(status_code),
            duration_ms=round(elapsed * 1000, 2),
        )


@app.exception_handler(HTTPException)
async def _http_exception_handler(request, exc: HTTPException):
    request_id = REQUEST_ID.get()
    resp = Response(content=json.dumps({"detail": exc.detail}), status_code=exc.status_code, media_type="application/json")
    if request_id:
        resp.headers["X-Request-ID"] = request_id
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc: Exception):
    request_id = REQUEST_ID.get()
    _json_log("unhandled_error", request_id=request_id, error=str(exc))
    debug = os.getenv("DEBUG", "0") == "1"
    detail = str(exc) if debug else "Internal server error"
    resp = Response(content=json.dumps({"detail": detail}), status_code=500, media_type="application/json")
    if request_id:
        resp.headers["X-Request-ID"] = request_id
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


def _require_api_key(x_api_key: str | None) -> None:
    configured = os.getenv("API_KEY")
    if not configured:
        return
    if not x_api_key or x_api_key != configured:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _max_upload_bytes() -> int:
    return _env_int("MAX_UPLOAD_MB", 10) * 1024 * 1024


_configure_http_middleware()


@app.on_event("startup")
def _startup() -> None:
    """Initialize config and load the model once per process."""
    _configure_logging()

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
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version() -> dict:
    return {
        "service": "defectguard",
        "git_sha": os.getenv("GIT_SHA"),
        "build_time": os.getenv("BUILD_TIME"),
    }


@app.get("/metrics")
def metrics(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> Response:
    """Prometheus scrape endpoint."""
    _require_api_key(x_api_key)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
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
        _require_api_key(x_api_key)

        # Read the entire uploaded file into memory (works for typical image sizes).
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(content) > _max_upload_bytes():
            raise HTTPException(status_code=413, detail="File too large")

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
    except HTTPException as e:
        status = str(e.status_code)
        raise
    except Exception as e:
        # Any unexpected error becomes 500 to the client.
        status = "500"
        _json_log("predict_error", error=str(e))
        debug = os.getenv("DEBUG", "0") == "1"
        raise HTTPException(status_code=500, detail=str(e) if debug else "Internal server error")
    finally:
        # Always record metrics, even on failures.
        elapsed = time.perf_counter() - start
        REQ_COUNT.labels(method="POST", path="/predict", status=status).inc()
        REQ_LATENCY.labels(path="/predict").observe(elapsed)
