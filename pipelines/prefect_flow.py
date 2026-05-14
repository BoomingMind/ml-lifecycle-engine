from __future__ import annotations

"""
Prefect orchestration flows.

Prefect is used here to schedule and observe multi-step jobs such as:
- dataset validation + retraining
- monitoring jobs (baseline snapshot + drift report)

We run the repo's scripts as subprocesses to keep each step simple and explicit.
"""

import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, task


def _run(cmd: list[str]) -> None:
    """
    Run a subprocess with PYTHONPATH pointing at src/ so imports work.

    Using `check=True` makes the task fail fast if any script exits non-zero.
    """

    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(Path("src").resolve())})

def _py(args: list[str]) -> list[str]:
    """
    Build a command that uses the current Python interpreter.

    This avoids relying on a `python` binary being present on PATH, which differs
    across OSes and environments.
    """

    return [sys.executable, *args]


@task
def validate_data() -> None:
    """Validate manifest schema and file paths."""
    _run(_py(["scripts/validate_data.py", "--manifest", os.getenv("MANIFEST_PATH", "data/manifest.csv")]))


@task
def train() -> None:
    """Train YOLO using params.yaml and optional YOLO_DATA_YAML override."""
    data_yaml = os.getenv("YOLO_DATA_YAML")
    cmd = _py(["scripts/train.py", "--params", os.getenv("PARAMS_PATH", "params.yaml")])
    if data_yaml:
        cmd += ["--data", data_yaml]
    _run(cmd)


@task
def set_reference_predictions() -> None:
    """Snapshot the current prediction log as a reference baseline."""
    _run(_py(["scripts/set_reference_predictions.py"]))


@task
def drift_report() -> None:
    """Generate an Evidently drift report comparing reference vs current logs."""
    _run(_py(["scripts/drift_report.py"]))


@flow(name="defect-detection-retraining")
def retraining_flow() -> None:
    """Offline training flow: validate → train."""
    validate_data()
    train()


@flow(name="defect-detection-monitoring")
def monitoring_flow() -> None:
    """Monitoring flow: snapshot baseline → compute drift report."""
    set_reference_predictions()
    drift_report()


if __name__ == "__main__":
    retraining_flow()
