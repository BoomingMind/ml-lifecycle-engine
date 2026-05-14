from __future__ import annotations

import io
import os

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


def _make_test_image_bytes() -> bytes:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint() -> None:
    os.environ.setdefault("DISABLE_MODEL_LOAD", "1")
    from api.main import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_predict_endpoint_returns_schema() -> None:
    os.environ.setdefault("DISABLE_MODEL_LOAD", "1")
    from api.main import app

    with TestClient(app) as client:
        image_bytes = _make_test_image_bytes()
        resp = client.post("/predict", files={"file": ("image.png", image_bytes, "image/png")})
        assert resp.status_code == 200
        payload = resp.json()
        assert "ts" in payload
        assert "image_sha256" in payload
        assert "boxes" in payload
        assert "scores" in payload
        assert "class_ids" in payload
        assert "class_names" in payload
