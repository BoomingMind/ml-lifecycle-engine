from __future__ import annotations

"""
MLflow model wrappers.

Ultralytics YOLO models load from a weights file (best.pt). MLflow Model Registry
and MLflow Serving work best with a standardized interface, so we wrap YOLOv8 in
an MLflow PyFunc model:

- input: pandas DataFrame (either `image_b64` or `image_path`)
- output: list of JSON-friendly dicts (boxes/scores/class metadata)
"""

import base64
import io
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


class YOLOv8PyFuncModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for YOLOv8.

    This is the bridge between training artifacts (weights) and serving (a stable
    `predict(DataFrame)` contract).
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Called by MLflow when the model is loaded.

        `context.artifacts` contains files bundled with the model. We expect a key
        named "weights" pointing to a YOLOv8 .pt file.
        """

        # MLflow packages "artifacts" together with the model. During logging we stored:
        # artifacts={"weights": "<path-to-best.pt>"}.
        weights_path = context.artifacts["weights"]
        # Load YOLO weights so predict() is fast and does not reload every call.
        self._model = YOLO(weights_path)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> Any:
        """
        Run YOLO prediction.

        Supported input schemas:
        - image_b64: base64-encoded image bytes (best for APIs)
        - image_path: file paths readable by this process (best for batch jobs)
        """

        # This function must accept a pandas DataFrame (PyFunc contract).
        # We support two common ways of providing images:
        # - base64 bytes (good for APIs)
        # - file paths (good for batch jobs)
        if "image_b64" in model_input.columns:
            # Convert each base64 string into a numpy RGB image.
            images = [self._decode_b64_image(v) for v in model_input["image_b64"].tolist()]
        elif "image_path" in model_input.columns:
            # Load each file path into a numpy RGB image.
            images = [np.array(Image.open(p).convert("RGB")) for p in model_input["image_path"].tolist()]
        else:
            raise ValueError("model_input must contain 'image_b64' or 'image_path'")

        # Ultralytics can accept a list of numpy arrays and returns one result per image.
        results = self._model.predict(images, verbose=False)
        # Convert each result to plain dicts so MLflow can serialize and clients can consume.
        return [self._format_result(r) for r in results]

    @staticmethod
    def _decode_b64_image(value: str) -> np.ndarray:
        """Decode base64 string into an RGB numpy array."""
        # base64 -> raw bytes
        raw = base64.b64decode(value)
        # bytes -> PIL image -> RGB
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        # PIL -> numpy array
        return np.array(image)

    @staticmethod
    def _format_result(result: Any) -> dict:
        """Normalize an Ultralytics result into a JSON-friendly dict."""
        boxes = result.boxes
        names = result.names

        if boxes is None or boxes.xyxy is None:
            return {"boxes": [], "scores": [], "class_ids": [], "class_names": []}

        # xyxy boxes are [x1, y1, x2, y2] for each detected object.
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        # Confidence score per detection.
        conf = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else []
        # Class ID per detection.
        cls = boxes.cls.cpu().numpy().astype(int).tolist() if boxes.cls is not None else []
        # Map class IDs to readable names when possible.
        class_names = [names.get(i, str(i)) for i in cls] if isinstance(names, dict) else [str(i) for i in cls]

        return {"boxes": xyxy, "scores": conf, "class_ids": cls, "class_names": class_names}
