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

        if mlflow_model_uri:
            self._kind = "mlflow"
            self._model = mlflow.pyfunc.load_model(mlflow_model_uri)
        elif model_path:
            self._kind = "weights"
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
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            df = pd.DataFrame([{"image_b64": b64}])
            out = self._model.predict(df)
            record = out[0] if isinstance(out, list) else out.iloc[0].to_dict()
            return Prediction(
                boxes=record.get("boxes", []),
                scores=record.get("scores", []),
                class_ids=record.get("class_ids", []),
                class_names=record.get("class_names", []),
            )

        # For local weights, we decode bytes into an RGB image and run Ultralytics YOLO.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(image)
        results = self._model.predict(arr, verbose=False)
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

        xyxy = boxes.xyxy.cpu().numpy().tolist()
        conf = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
        cls = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
        class_names = [names.get(i, str(i)) for i in cls] if isinstance(names, dict) else [str(i) for i in cls]
        return {"boxes": xyxy, "scores": conf, "class_ids": cls, "class_names": class_names}
