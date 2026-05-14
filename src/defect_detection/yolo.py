from __future__ import annotations

"""
Unified YOLOv8 prediction interface.

The API should not care *where* the model comes from:
- a local YOLOv8 weights file (best.pt), or
- an MLflow PyFunc model URI (models:/..., runs:/...).

This module provides a single `YoloPredictor` that hides those details and returns
a stable, JSON-friendly `Prediction` schema.
"""

import base64
import io
from dataclasses import dataclass
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class Prediction:
    """
    Normalized object-detection output.

    - boxes: list of [x1, y1, x2, y2] in pixel coordinates
    - scores: confidence per box (0..1)
    - class_ids: integer class index per box
    - class_names: human-readable class label per box
    """

    boxes: list[list[float]]
    scores: list[float]
    class_ids: list[int]
    class_names: list[str]


class YoloPredictor:
    def __init__(self, model_path: str | None = None, mlflow_model_uri: str | None = None) -> None:
        """
        Load a predictor from either MLflow or local weights.

        - If `mlflow_model_uri` is set, this loads an MLflow PyFunc model.
        - Otherwise, `model_path` must point to YOLOv8 weights (e.g. best.pt).
        """

        # We support two serving modes:
        # 1) MLflow mode: load a model by URI from an MLflow tracking/registry backend
        # 2) Weights mode: load a local YOLOv8 .pt file directly
        if mlflow_model_uri:
            self._kind = "mlflow"
            # mlflow.pyfunc.load_model returns a "PyFunc" model with a standardized predict() API.
            self._model = mlflow.pyfunc.load_model(mlflow_model_uri)
        elif model_path:
            self._kind = "weights"
            # ultralytics.YOLO loads and runs weights directly (no MLflow involved).
            self._model = YOLO(model_path)
        else:
            raise ValueError("Either model_path or mlflow_model_uri must be provided")

    def predict_image_bytes(self, image_bytes: bytes) -> Prediction:
        """
        Run object detection on raw image bytes.

        Returns a `Prediction` that is safe to JSON-serialize and log.
        """

        if self._kind == "mlflow":
            # MLflow PyFunc models expect tabular inputs (DataFrames). We pass the image
            # as base64 to keep the input JSON-friendly.
            # Step 1: encode bytes -> base64 string (safe to put into JSON/CSV/DataFrame).
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            # Step 2: build a 1-row DataFrame because PyFunc predict() expects tabular input.
            df = pd.DataFrame([{"image_b64": b64}])
            # Step 3: call the model. Our wrapper returns list-of-dicts (one dict per input image).
            out = self._model.predict(df)
            # Step 4: normalize output to a plain dict no matter what type MLflow returns.
            record = out[0] if isinstance(out, list) else out.iloc[0].to_dict()
            return Prediction(
                boxes=record.get("boxes", []),
                scores=record.get("scores", []),
                class_ids=record.get("class_ids", []),
                class_names=record.get("class_names", []),
            )

        # For local weights, we decode bytes into an RGB image and run Ultralytics YOLO.
        # Step 1: decode bytes -> PIL image (common image representation in Python).
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Step 2: convert to numpy array because Ultralytics accepts numpy images.
        arr = np.array(image)
        # Step 3: run YOLO prediction. It returns a list of results (one per input image).
        results = self._model.predict(arr, verbose=False)
        # Step 4: convert the rich result object into simple JSON-friendly lists.
        record = self._format_ultralytics(results[0])
        return Prediction(**record)

    @staticmethod
    def _format_ultralytics(result: Any) -> dict:
        """
        Convert Ultralytics result objects into plain Python lists.

        Ultralytics returns tensors and rich objects; the API/logging layer needs
        simple JSON-compatible structures.
        """

        boxes = result.boxes
        names = result.names

        if boxes is None or boxes.xyxy is None:
            return {"boxes": [], "scores": [], "class_ids": [], "class_names": []}

        # boxes.xyxy is a tensor shaped [N, 4] where N = number of detections.
        # We move it to CPU, convert to numpy, then to a Python list for JSON serialization.
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        # Confidence scores per detection (tensor [N]).
        conf = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
        # Class IDs per detection (tensor [N]). Cast to int so JSON consumers don't get floats.
        cls = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
        # Map class IDs to human-readable names if available.
        class_names = [names.get(i, str(i)) for i in cls] if isinstance(names, dict) else [str(i) for i in cls]
        return {"boxes": xyxy, "scores": conf, "class_ids": cls, "class_names": class_names}
