from __future__ import annotations

import io
import os

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def _make_test_image_bytes() -> bytes:
    # Create a small deterministic RGB image so tests don't depend on external files.
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    # Save the image to an in-memory buffer (BytesIO) instead of writing to disk.
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint() -> None:
    # Disable real model loading so CI does not download weights or require MLflow.
    os.environ.setdefault("DISABLE_MODEL_LOAD", "1")
    # Import the FastAPI app after setting env so startup uses the dummy predictor.
    from api.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_ready_endpoint() -> None:
    os.environ.setdefault("DISABLE_MODEL_LOAD", "1")
    from api.main import app

    with TestClient(app) as client:
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


def test_ui_root_serves_html() -> None:
    os.environ.setdefault("DISABLE_MODEL_LOAD", "1")
    from api.main import app

    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


def test_predict_endpoint_returns_schema() -> None:
    # Same testing strategy as above: avoid heavy model loading.
    os.environ.setdefault("DISABLE_MODEL_LOAD", "1")
    from api.main import app

    with TestClient(app) as client:
        image_bytes = _make_test_image_bytes()
        # FastAPI expects file uploads under the "files" parameter (multipart/form-data).
        resp = client.post("/predict", files={"file": ("image.png", image_bytes, "image/png")})
        assert resp.status_code == 200
        payload = resp.json()
        # Verify output schema is stable (clients rely on these keys).
        assert "ts" in payload
        assert "image_sha256" in payload
        assert "boxes" in payload
        assert "scores" in payload
        assert "class_ids" in payload
        assert "class_names" in payload


def test_api_key_auth_for_predict() -> None:
    os.environ.setdefault("DISABLE_MODEL_LOAD", "1")
    os.environ["API_KEY"] = "secret"
    try:
        from api.main import app

        with TestClient(app) as client:
            image_bytes = _make_test_image_bytes()
            resp = client.post("/predict", files={"file": ("image.png", image_bytes, "image/png")})
            assert resp.status_code == 401

            resp = client.post(
                "/predict",
                files={"file": ("image.png", image_bytes, "image/png")},
                headers={"X-API-Key": "secret"},
            )
            assert resp.status_code == 200
    finally:
        del os.environ["API_KEY"]
