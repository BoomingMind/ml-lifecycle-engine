"""
Defect detection library package.

This package contains the reusable building blocks used by:
- the FastAPI service (serving)
- training and monitoring scripts (offline jobs)
"""

# __all__ controls what gets exported when another file does:
# from defect_detection import *
# We keep this explicit so the public surface of the package is easy to understand.
__all__ = ["config", "mlflow_models", "mlflow_utils", "yolo"]
