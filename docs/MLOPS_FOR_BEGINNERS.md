# MLOps For Beginners: DefectGuard Explained Slowly

This document explains the project from the MLOps point of view.

It is written for someone who may be:

- new to MLOps
- new to backend engineering
- new to deployment
- new to model serving
- new to tools like MLflow, DVC, Prefect, Prometheus, or Grafana

The goal is simple:

- explain what each major tool does
- explain why it exists in this project
- explain which files and functions use it
- explain how all the parts connect together

If you read this guide from start to finish, you should understand not only "what the project does", but also "why this is called an MLOps project".

## 1. What Is MLOps?

MLOps means "Machine Learning Operations".

In plain language:

- machine learning builds the model
- operations keeps the system usable, repeatable, and reliable

A normal beginner ML project often stops at:

- training a model
- saving a weights file
- maybe showing accuracy

An MLOps project goes further:

- how do you validate data?
- how do you track experiments?
- how do you version models?
- how do you serve predictions?
- how do you monitor what happens after deployment?
- how do you test and automate the system?

That is what this repository is trying to teach.

## 2. What Makes This Repo An MLOps Project?

This project is not only:

- YOLO training

It also includes:

- data validation
- experiment tracking
- model packaging
- model registry support
- quality gates
- API serving
- browser UI
- prediction logging
- drift monitoring
- Docker deployment
- metrics and dashboards
- workflow orchestration
- automated tests
- CI

That full lifecycle is what makes it MLOps-focused.

## 3. The Big System Story

Here is the simplest story of the whole repository:

1. Images are collected or downloaded.
2. A manifest or dataset YAML describes them.
3. Validation checks run before training.
4. YOLOv8 trains a defect detection model.
5. MLflow logs the run and packages the model.
6. The API loads the model and serves predictions.
7. The browser UI lets a human upload an image and see defect boxes.
8. The API writes prediction logs for monitoring.
9. Prometheus and Grafana monitor service behavior.
10. Evidently compares current prediction behavior against a baseline.
11. Prefect and DVC help organize repeatable workflows.

Every tool in the repo supports one part of this story.

## 4. The MLOps Lifecycle In This Repo

Think of the project in layers.

### Layer 1: Data Preparation

Goal:

- get data into a usable format
- verify the data is not broken

Main files:

- `scripts/download_mvtec_ad.py`
- `scripts/validate_data.py`
- `data/dataset.yaml`
- `data/manifest.csv`
- `dvc.yaml`

### Layer 2: Training And Experiment Tracking

Goal:

- train a model
- record what happened

Main files:

- `scripts/train.py`
- `params.yaml`
- `src/defect_detection/mlflow_models.py`
- `src/defect_detection/mlflow_utils.py`

### Layer 3: Serving

Goal:

- make the model usable by other software and people

Main files:

- `api/main.py`
- `src/defect_detection/yolo.py`
- `api/frontend/index.html`
- `api/frontend/app.js`
- `api/frontend/styles.css`

### Layer 4: Monitoring

Goal:

- observe the service
- observe the model behavior after deployment

Main files:

- `scripts/set_reference_predictions.py`
- `scripts/drift_report.py`
- `monitoring/prometheus.yml`
- `monitoring/grafana/provisioning/datasources/datasource.yml`

### Layer 5: Operations And Automation

Goal:

- run the platform reliably
- automate checks and workflows

Main files:

- `pipelines/prefect_flow.py`
- `docker-compose.yml`
- `nginx/nginx.conf`
- `.github/workflows/ci.yml`
- `tests/test_api.py`

## 5. Tool-By-Tool Explanation

This section explains each major tool in beginner language.

## 6. YOLOv8

What it is:

- an object detection framework

What object detection means:

- find where something is in an image
- draw a box around it
- assign a class
- provide a confidence score

Why this project uses it:

- defect detection is a visual detection problem
- YOLO is popular, practical, and fast

Where it appears:

- `scripts/train.py`
- `src/defect_detection/yolo.py`
- `src/defect_detection/mlflow_models.py`

Important functions and code usage:

- `YOLO(...).train(...)` in `scripts/train.py`
- `YOLO(...).val(...)` in `scripts/train.py`
- `YOLO(model_path)` in `src/defect_detection/yolo.py`
- `YOLO(weights_path)` in `src/defect_detection/mlflow_models.py`

What it produces:

- training outputs under `runs/`
- `best.pt` weights
- prediction boxes, class IDs, and confidence scores

## 7. MLflow

What it is:

- an experiment tracking and model management system

Why it matters:

- training many models without tracking is confusing
- teams need to know which run created which model
- production systems need a consistent way to package and register models

This project uses MLflow for:

- experiment tracking
- artifact storage
- model packaging
- model registry support

Where it appears:

- `scripts/train.py`
- `src/defect_detection/mlflow_utils.py`
- `src/defect_detection/mlflow_models.py`
- `api/main.py`

Important functions:

### `configure_mlflow(tracking_uri)`

File:

- `src/defect_detection/mlflow_utils.py`

What it does:

- points the current process to the right MLflow server

Used by:

- training
- serving

### `resolve_model_uri(explicit_model_uri=None)`

File:

- `src/defect_detection/mlflow_utils.py`

What it does:

- decides which MLflow model URI should be served

Examples:

- `runs:/<run_id>/model`
- `models:/defect-yolo/Production`

### `mlflow.start_run()`

File:

- `scripts/train.py`

What it does:

- opens an MLflow run so parameters, metrics, and artifacts are recorded together

### `mlflow.log_params(...)`

File:

- `scripts/train.py`

What it does:

- stores training configuration with the run

### `mlflow.log_metric(...)` and `mlflow.log_metrics(...)`

File:

- `scripts/train.py`

What they do:

- store numeric results like training metrics and validation metrics

### `mlflow.log_artifact(...)`

File:

- `scripts/train.py`

What it does:

- stores files such as `best.pt`

### `mlflow.pyfunc.log_model(...)`

File:

- `scripts/train.py`

What it does:

- logs a standardized MLflow model that serving can load consistently

### `mlflow.register_model(...)`

Files:

- `scripts/train.py`
- `src/defect_detection/mlflow_utils.py`

What it does:

- creates a versioned registry entry for the model

## 8. MLflow PyFunc

What it is:

- MLflow's standardized model interface

Why it matters:

- raw YOLO weights are not enough for a clean serving contract
- PyFunc gives a predictable `predict(DataFrame)` interface

Where it appears:

- `src/defect_detection/mlflow_models.py`
- `scripts/train.py`
- `src/defect_detection/yolo.py`

Main class:

- `YOLOv8PyFuncModel`

Important functions:

### `load_context(...)`

What it does:

- loads the model weights artifact when MLflow loads the model

### `predict(context, model_input)`

What it does:

- accepts a DataFrame
- reads `image_b64` or `image_path`
- runs YOLO
- returns JSON-friendly output

Why this is useful:

- FastAPI serving can load from MLflow without caring about raw YOLO internals

## 9. FastAPI

What it is:

- the web framework used for the inference API

Why it matters:

- a trained model needs a way to receive requests and return predictions

Where it appears:

- `api/main.py`

Main endpoints:

- `GET /`
- `GET /ui`
- `GET /health`
- `GET /ready`
- `GET /version`
- `GET /metrics`
- `POST /predict`

Important functions:

### `_startup()`

What it does:

- loads config
- configures middleware
- loads the predictor
- stores shared objects on `app.state`

### `health()`

What it does:

- says the process is alive

### `ready()`

What it does:

- says the model is loaded and ready

### `metrics(...)`

What it does:

- exposes Prometheus-format metrics

### `predict(...)`

What it does:

- reads an uploaded image
- runs prediction
- logs the result
- returns JSON

## 10. The Predictor Abstraction

One of the smartest design choices in the repo is the predictor abstraction.

File:

- `src/defect_detection/yolo.py`

Main class:

- `YoloPredictor`

Why this exists:

- the API should not care whether the model comes from MLflow or local weights

Important pieces:

### `Prediction`

What it is:

- a dataclass that stores normalized detection results

Fields:

- `boxes`
- `scores`
- `class_ids`
- `class_names`

### `YoloPredictor.__init__(...)`

What it does:

- loads either:
  - MLflow model, or
  - raw YOLO weights

### `predict_image_bytes(image_bytes)`

What it does:

- converts raw uploaded bytes into model input
- runs prediction
- returns normalized results

### `_format_ultralytics(result)`

What it does:

- converts YOLO tensors and rich objects into plain Python lists

Why this matters for MLOps:

- serving becomes stable
- logging becomes easier
- the API contract stays consistent

## 11. Great Expectations

What it is:

- a data validation framework

Why it matters:

- training should not start with obviously broken data

Where it appears:

- `scripts/validate_data.py`

Important function:

### `_validate_with_ge(df)`

What it does:

- checks the manifest DataFrame for required structure

Current expectations include:

- `image_path` column exists
- `image_path` is not null
- `image_path` values are strings

Why this matters:

- it catches errors early
- it protects the training stage from bad inputs

## 12. DVC

What it is:

- a tool for defining reproducible data and ML pipeline stages

Why it matters:

- ML projects often need repeatable pipelines, not only one-off commands

Where it appears:

- `dvc.yaml`
- `params.yaml`

What DVC tracks here:

- commands
- dependencies
- outputs
- parameters

Current stages:

- `download_mvtec_ad`
- `validate`
- `train`

Why this matters:

- if something changes, DVC knows what should rerun

## 13. Prefect

What it is:

- a workflow orchestration tool

Why it matters:

- MLOps needs repeatable, trackable multi-step jobs

Where it appears:

- `pipelines/prefect_flow.py`

Important functions:

### `_run(cmd)`

What it does:

- runs subprocesses with correct `PYTHONPATH`

### `_py(args)`

What it does:

- uses the current Python interpreter instead of assuming `python` exists on PATH

### `validate_data()`

- task that runs the validation script

### `train()`

- task that runs the training script

### `set_reference_predictions()`

- task that creates the baseline snapshot

### `drift_report()`

- task that generates the Evidently report

### `retraining_flow()`

- validate -> train

### `monitoring_flow()`

- set reference -> drift report

## 14. Evidently

What it is:

- a monitoring and reporting library for ML systems

Why it matters:

- after deployment, model behavior may drift

Where it appears:

- `scripts/drift_report.py`

Important functions:

### `_read_jsonl(path)`

What it does:

- loads prediction logs into a DataFrame

### `_explode_predictions(df)`

What it does:

- converts variable-length prediction lists into fixed numeric features:
  - `n_boxes`
  - `max_score`
  - `mean_score`

### `main()`

What it does:

- loads reference and current prediction logs
- builds an Evidently report
- saves an HTML file

Important beginner note:

- this is behavior drift monitoring
- it is not exactly the same as measuring true production accuracy

## 15. Prometheus

What it is:

- a metrics collection system

Why it matters:

- services need operational visibility

Where it appears:

- `api/main.py`
- `monitoring/prometheus.yml`
- `docker-compose.yml`

Important metric objects in `api/main.py`:

### `REQ_COUNT`

What it measures:

- total HTTP requests by method, path, and status

### `REQ_LATENCY`

What it measures:

- request duration by path

### `PRED_COUNT`

What it measures:

- number of successful predictions

Important function:

### `metrics(...)`

What it does:

- returns Prometheus-format metrics for scraping

Prometheus config:

- `scrape_interval: 15s`
- target: `api:8000`
- path: `/metrics`

## 16. Grafana

What it is:

- a dashboard tool for visualizing metrics

Why it matters:

- raw metrics are useful, but dashboards make trends much easier to understand

Where it appears:

- `monitoring/grafana/provisioning/datasources/datasource.yml`
- `docker-compose.yml`

Important config:

- datasource name: `Prometheus`
- datasource URL: `http://prometheus:9090`
- `isDefault: true`

Why provisioning matters:

- Grafana becomes ready immediately after startup

## 17. Docker

What it is:

- containerization technology

Why it matters:

- a project should run consistently across machines

Where it appears:

- `docker/Dockerfile.api`
- `docker/Dockerfile.mlflow`

API container does:

- installs runtime dependencies
- copies source code
- starts Uvicorn

MLflow container does:

- installs MLflow
- starts the MLflow server

## 18. Docker Compose

What it is:

- a way to run multiple containers together

Why it matters:

- this project is a multi-service platform, not only one process

Where it appears:

- `docker-compose.yml`

Services:

- `mlflow`
- `api`
- `nginx`
- `prometheus`
- `grafana`

Why this matters:

- it lets the project feel like a real local platform

## 19. Nginx

What it is:

- a reverse proxy

Why it matters:

- production systems often place a proxy in front of the app

Where it appears:

- `nginx/nginx.conf`
- `docker-compose.yml`

What it does here:

- routes requests to the API
- forwards request IDs
- adds browser security headers
- enables gzip

## 20. pytest

What it is:

- the testing framework used in the repo

Why it matters:

- code changes should not silently break behavior

Where it appears:

- `tests/conftest.py`
- `tests/test_api.py`

Important files:

### `tests/conftest.py`

What it does:

- adds `src/` to Python import path

### `tests/test_api.py`

What it tests:

- `/health`
- `/ready`
- `/`
- `/predict`
- API key behavior

Important helper:

### `_make_test_image_bytes()`

What it does:

- creates a tiny in-memory PNG image so tests do not depend on external files

Important MLOps idea:

- the tests use `DISABLE_MODEL_LOAD=1` so CI does not need real weights

## 21. GitHub Actions CI

What it is:

- automated checks that run on GitHub

Where it appears:

- `.github/workflows/ci.yml`

What it does:

1. checks out the code
2. installs Python
3. installs dependencies
4. runs Ruff linting
5. runs pytest

Why it matters:

- every push and pull request gets basic quality checks

## 22. Configuration Files And Why They Matter

## 23. `params.yaml`

Purpose:

- keep training parameters outside Python code

Current values include:

- model
- data
- epochs
- imgsz
- batch

Why this matters:

- experiments become easier to change and compare

## 24. `data/dataset.yaml`

Purpose:

- teach YOLO where the dataset lives

Main keys:

- `path`
- `train`
- `val`
- `names`

## 25. `pyproject.toml`

Purpose:

- configure development tooling

Used for:

- Ruff settings
- pytest settings

## 26. `requirements*.txt`

Why there are three files:

- `requirements.txt`: runtime app dependencies
- `requirements-mlops.txt`: MLOps and pipeline dependencies
- `requirements-dev.txt`: testing and linting dependencies

Why this separation is helpful:

- easier to understand what each dependency group is for

## 27. File-By-File MLOps Map

This section maps the project to MLOps responsibilities.

### Data

- `scripts/download_mvtec_ad.py`
- `scripts/validate_data.py`
- `data/dataset.yaml`
- `data/manifest.csv`

### Training

- `scripts/train.py`
- `params.yaml`

### Model Packaging And Registry

- `src/defect_detection/mlflow_models.py`
- `src/defect_detection/mlflow_utils.py`

### Serving

- `api/main.py`
- `src/defect_detection/yolo.py`
- `api/frontend/*`

### Monitoring

- `scripts/set_reference_predictions.py`
- `scripts/drift_report.py`
- `monitoring/prometheus.yml`
- `monitoring/grafana/provisioning/datasources/datasource.yml`

### Orchestration

- `pipelines/prefect_flow.py`
- `dvc.yaml`

### Deployment

- `docker/Dockerfile.api`
- `docker/Dockerfile.mlflow`
- `docker-compose.yml`
- `nginx/nginx.conf`

### Quality

- `tests/test_api.py`
- `tests/conftest.py`
- `.github/workflows/ci.yml`

## 28. Most Important Functions In The Project

If a beginner wants the shortest “important function” list, start here.

### `load_api_config()`

File:

- `src/defect_detection/config.py`

Why it matters:

- serving config starts here

### `configure_mlflow(...)`

File:

- `src/defect_detection/mlflow_utils.py`

Why it matters:

- training and serving need the correct tracking backend

### `resolve_model_uri(...)`

File:

- `src/defect_detection/mlflow_utils.py`

Why it matters:

- model-serving source is decided here

### `YOLOv8PyFuncModel.predict(...)`

File:

- `src/defect_detection/mlflow_models.py`

Why it matters:

- MLflow model serving contract lives here

### `YoloPredictor.predict_image_bytes(...)`

File:

- `src/defect_detection/yolo.py`

Why it matters:

- raw image bytes become predictions here

### `_startup()`

File:

- `api/main.py`

Why it matters:

- API startup lifecycle lives here

### `predict(...)`

File:

- `api/main.py`

Why it matters:

- live business inference endpoint

### `_validate_with_ge(df)`

File:

- `scripts/validate_data.py`

Why it matters:

- data quality gate

### `_extract_eval_metrics(result)`

File:

- `scripts/train.py`

Why it matters:

- training evaluation becomes usable metrics here

### `_get_production_map50(...)`

File:

- `scripts/train.py`

Why it matters:

- champion-vs-challenger comparison uses this

### `_explode_predictions(df)`

File:

- `scripts/drift_report.py`

Why it matters:

- raw prediction logs become drift-ready numeric features here

## 29. What Happens During A Real Prediction Request?

Step by step:

1. browser or client sends an image to `POST /predict`
2. middleware creates a request ID and starts timing
3. API checks auth if configured
4. API reads file bytes
5. API checks file size
6. API calls `predict_image_bytes(...)`
7. predictor runs MLflow or raw YOLO logic
8. prediction result is normalized
9. API builds a JSON record
10. API writes a JSONL log line
11. API returns JSON to the client
12. metrics are updated

This single request touches many MLOps ideas:

- serving
- observability
- monitoring
- logging
- API stability

## 30. What Happens During A Real Training Run?

Step by step:

1. config is loaded
2. MLflow is configured
3. experiment run starts
4. parameters are logged
5. YOLO trains
6. training metrics are logged
7. `best.pt` is located
8. validation runs
9. evaluation metrics are extracted
10. artifacts are logged
11. PyFunc model is logged
12. optional registry registration happens
13. optional quality gate is enforced
14. optional promotion to Production happens

This touches:

- reproducibility
- experiment tracking
- model governance
- deployment policy

## 31. What Happens After Deployment?

Many beginners stop thinking after "the API works".

MLOps continues after deployment.

This project continues with:

- health checks
- readiness checks
- metrics scraping
- dashboard viewing
- prediction logging
- drift report generation
- future retraining support

That is why the repo includes monitoring and orchestration tools.

## 32. How To Study This Repo As An MLOps Learner

If you want a beginner-friendly order, read like this:

1. `README.md`
2. `docs/MLOPS_FOR_BEGINNERS.md`
3. `docs/WALKTHROUGH.md`
4. `docs/PROJECT_BOOK.md`
5. `api/main.py`
6. `src/defect_detection/yolo.py`
7. `scripts/train.py`
8. `scripts/validate_data.py`
9. `scripts/drift_report.py`
10. `docker-compose.yml`

Why this order works:

- first understand the system
- then understand the tools
- then understand the code

## 33. Final Beginner Summary

If you remember only one thing, remember this:

This repo teaches that a machine learning project is not only:

- train model
- save weights

It is also:

- validate data
- track runs
- package models
- serve predictions
- secure and monitor services
- observe model behavior after deployment
- test changes
- automate workflows

That is the core idea of MLOps.

And this repository gives you a practical example of that idea across real files, real functions, and real tools.
