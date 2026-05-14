# DefectGuard Project Book

This document is the "read it like a book" guide to the repository.

It explains:

- what this project does
- why each part exists
- how the code is organized
- how data, training, serving, monitoring, and deployment connect together
- what the main functions do
- how to run the project end to end

The writing style is intentionally beginner-friendly. You do not need strong coding experience to follow the main ideas.

## 1. What This Project Is

This repository is an end-to-end MLOps project for manufacturing defect detection.

In simple words:

1. A model looks at product images.
2. It tries to detect defects.
3. The system exposes that model through an API so other software can use it.
4. The system logs predictions so we can monitor behavior over time.
5. The system includes training, validation, monitoring, registry, orchestration, and deployment pieces.

This makes it more than "just a model training notebook".

It is a full production-shaped machine learning system.

## 2. The Problem It Solves

Imagine a factory line producing:

- metal components
- circuit boards
- textiles
- mechanical parts

Normally, a human inspector looks for scratches, cracks, missing parts, stains, or shape problems.

That process has challenges:

- humans get tired
- inspection quality can vary
- defects may be subtle
- production lines move quickly

This project shows how computer vision plus MLOps can help:

- a model checks images automatically
- the API returns detection boxes and confidence scores
- logs are stored for monitoring
- retraining can be repeated with a pipeline
- production deployment can use MLflow, Docker, Nginx, Prometheus, and Grafana

## 3. What "End To End" Means Here

"End to end" means the repository does not stop at one stage.

It covers the full machine learning lifecycle:

1. Data acquisition
2. Data validation
3. Training
4. Evaluation
5. Model packaging
6. Model registration
7. Model serving
8. Frontend interaction
9. Prediction logging
10. Drift monitoring
11. Workflow orchestration
12. Local deployment
13. CI testing

That is why the repo has many folders instead of only one script.

## 4. The System In One Story

Here is the easiest way to understand the full system:

1. A dataset is downloaded or prepared.
2. A manifest or dataset description tells the system where files are.
3. Validation checks run before training.
4. YOLOv8 trains a defect detection model.
5. Training results, parameters, and artifacts are logged to MLflow.
6. The trained model can be wrapped as an MLflow PyFunc model.
7. The FastAPI service loads the model once at startup.
8. A user uploads an image from the browser UI or via API.
9. The API runs prediction and returns boxes, classes, and scores.
10. The API writes one JSON log line for that prediction.
11. Monitoring scripts compare current logs against a baseline reference.
12. Prometheus and Grafana monitor service metrics.

That is the full product loop.

## 5. Project Phases

To make the architecture easier to learn, think of the repository in phases.

### Phase 1: Foundation

Goal: create a clean, reproducible project structure.

Main pieces:

- `src/defect_detection/` for reusable Python code
- `api/` for serving
- `scripts/` for operations like training and validation
- `pipelines/` for orchestration
- `tests/` for automated verification
- `docker-compose.yml` for local infrastructure
- `README.md` and docs for human understanding

Why this phase matters:

- without structure, a project becomes hard to maintain
- without docs, new teammates cannot learn quickly
- without reproducibility, ML systems become fragile

### Phase 2: Data Layer

Goal: make data input explicit and repeatable.

Main pieces:

- `scripts/download_mvtec_ad.py`
- `data/dataset.yaml`
- `data/manifest.csv`
- `scripts/validate_data.py`
- `dvc.yaml`

Why this phase matters:

- training quality depends on data quality
- broken file paths should fail early
- reproducible data preparation is part of MLOps

### Phase 3: Training Layer

Goal: train a YOLOv8 defect detector and keep track of what happened.

Main pieces:

- `scripts/train.py`
- `params.yaml`
- `src/defect_detection/mlflow_models.py`
- `src/defect_detection/mlflow_utils.py`

Why this phase matters:

- experiments need parameters and metrics
- model files alone are not enough
- teams need to compare runs and register better models

### Phase 4: Model Registry And Quality Control

Goal: prevent low-quality models from becoming production models by accident.

Main pieces:

- MLflow Model Registry
- evaluation metric extraction
- quality gate based on `mAP@0.5`
- champion vs challenger promotion logic

Why this phase matters:

- new models are not always better
- production systems need rules, not hope

### Phase 5: Serving Layer

Goal: make the model available to users and other systems.

Main pieces:

- `api/main.py`
- `src/defect_detection/yolo.py`
- `api/frontend/`

Why this phase matters:

- a trained model is only useful if something can call it
- APIs make models usable by websites, mobile apps, and factory systems

### Phase 6: Monitoring Layer

Goal: observe whether the service and model behavior stay healthy over time.

Main pieces:

- `data/predictions.jsonl`
- `scripts/set_reference_predictions.py`
- `scripts/drift_report.py`
- `monitoring/prometheus.yml`
- Grafana datasource provisioning

Why this phase matters:

- production data changes
- model output behavior can drift
- service health and latency need tracking

### Phase 7: Workflow And Deployment Layer

Goal: run the system in an organized, production-shaped way.

Main pieces:

- `pipelines/prefect_flow.py`
- `docker/Dockerfile.api`
- `docker/Dockerfile.mlflow`
- `docker-compose.yml`
- `nginx/nginx.conf`
- GitHub Actions CI

Why this phase matters:

- reliable systems need automation
- local reproducibility is a step toward production deployment

## 6. Folder-By-Folder Tour

### `src/defect_detection/`

This is the reusable library code.

Think of it as the "brain helpers" used by other parts of the project.

Files:

- `config.py`: reads config from environment variables and YAML
- `mlflow_utils.py`: small helper functions for MLflow setup and model URI resolution
- `mlflow_models.py`: wraps YOLO inside an MLflow PyFunc model
- `yolo.py`: creates one stable prediction interface for API usage
- `__init__.py`: package exports

### `api/`

This is the live serving application.

Files:

- `main.py`: FastAPI service
- `frontend/index.html`: page layout
- `frontend/styles.css`: UI styling
- `frontend/app.js`: browser logic for upload, preview, and drawing boxes

### `scripts/`

These are focused command-line tools.

Each script does one job clearly.

Files:

- `download_mvtec_ad.py`
- `validate_data.py`
- `train.py`
- `evaluate.py`
- `set_reference_predictions.py`
- `drift_report.py`

### `pipelines/`

This folder contains workflow orchestration with Prefect.

### `monitoring/`

This folder contains monitoring configuration such as Prometheus and Grafana datasource setup.

### `docker/`

This folder contains the container build instructions.

### `tests/`

This folder contains automated checks for API behavior.

## 7. Important Technical Concepts In Plain Language

Before reading file details, these ideas help a lot.

### YOLOv8

YOLOv8 is the object detection model used here.

Object detection means:

- locate an object in an image
- draw a rectangle around it
- assign a class label
- provide a confidence score

In this project, the "object" is usually a defect.

### MLflow

MLflow is the experiment tracking and model registry system.

It helps store:

- run parameters
- metrics
- artifacts like `best.pt`
- registered model versions

### PyFunc Model

A PyFunc model is MLflow's standard prediction interface.

Instead of every model using different input and output formats, PyFunc gives a common contract:

- input: usually a pandas DataFrame
- output: standard Python / table-like data

This makes serving more consistent.

### FastAPI

FastAPI is the web framework used for the API service.

It gives us routes such as:

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `POST /predict`

### JSONL

JSONL means "JSON Lines".

Each line is a separate JSON object.

It is useful for logs because:

- appending is easy
- each request can become one line
- later scripts can read line by line

### DVC

DVC describes ML pipeline stages and their dependencies.

It helps answer:

- which step should run
- what inputs each step depends on
- what outputs each step creates

### Prefect

Prefect is used for workflow orchestration.

It helps combine multiple steps into a tracked flow.

### Prometheus And Grafana

- Prometheus collects metrics
- Grafana visualizes those metrics

This is about service observability, not only model quality.

## 8. How The Main Code Works

This section explains the most important files and functions in plain language.

## 9. `src/defect_detection/config.py`

Purpose:

- centralize configuration loading
- avoid repeating `os.getenv(...)` everywhere
- prepare paths early

### `ApiConfig`

This dataclass is a simple container.

It stores:

- MLflow tracking URI
- MLflow model URI
- local model path
- prediction log path

Why this is useful:

- one object is easier to pass around than many environment variable reads

### `load_api_config()`

What it does:

1. Reads environment variables.
2. Decides where prediction logs should be written.
3. Creates the log directory if needed.
4. Returns an `ApiConfig` object.

Why it matters:

- the API should not fail only because a log folder does not exist yet

### `load_params()`

What it does:

1. Reads `params.yaml`.
2. Returns a Python dictionary.
3. Returns `{}` if the file is missing.

Why it matters:

- scripts can keep working with safe defaults

## 10. `src/defect_detection/mlflow_utils.py`

Purpose:

- keep MLflow-specific helper logic in one place

### `configure_mlflow(tracking_uri)`

What it does:

- points the current Python process to the right MLflow tracking server

Why it matters:

- both training and serving may need MLflow access

### `resolve_model_uri(explicit_model_uri=None)`

What it does:

- decides which MLflow model URI should be used

Priority:

1. explicit function argument
2. `MLFLOW_MODEL_URI`
3. `MLFLOW_MODEL_NAME` + `MLFLOW_MODEL_STAGE`

Example result:

```text
models:/defect-yolo/Production
```

Why it matters:

- it supports both direct and registry-based serving styles

### `promote_best_to_production(...)`

What it does:

1. takes a logged run model
2. registers it
3. moves it to the `Production` stage

Why it matters:

- deployment workflows often need a single official production model reference

## 11. `src/defect_detection/mlflow_models.py`

Purpose:

- bridge between raw YOLO weights and MLflow model serving

### `YOLOv8PyFuncModel`

This class lets MLflow load YOLO as a standardized model.

### `load_context(...)`

What it does:

1. MLflow passes in bundled artifacts.
2. The code looks for the `weights` artifact.
3. It loads YOLO weights once.

Why it matters:

- repeated reloading inside every prediction would be slow

### `predict(context, model_input)`

What it does:

1. Accepts a pandas DataFrame.
2. Supports either `image_b64` or `image_path`.
3. Converts each image into a numpy RGB array.
4. Runs YOLO prediction.
5. Returns JSON-friendly dictionaries.

Why it matters:

- MLflow now has a stable interface for this model

### `_decode_b64_image(value)`

What it does:

1. decodes base64 text
2. opens the image with PIL
3. converts it to numpy

Why it matters:

- APIs often move image data as text-safe formats

### `_format_result(result)`

What it does:

- converts rich Ultralytics result objects into plain Python lists

Why this is necessary:

- tensors and special objects are harder to serialize into JSON or tables

## 12. `src/defect_detection/yolo.py`

Purpose:

- give the rest of the project one simple predictor interface

Important design idea:

The API should not care whether the model comes from:

- a local `.pt` file
- an MLflow model URI

That is why `YoloPredictor` exists.

### `Prediction`

This dataclass stores:

- `boxes`
- `scores`
- `class_ids`
- `class_names`

Why it matters:

- it gives the codebase one normalized output structure

### `YoloPredictor.__init__(model_path, mlflow_model_uri)`

What it does:

1. chooses serving mode
2. loads either:
   - an MLflow PyFunc model, or
   - local YOLO weights

Why it matters:

- one project can support both registry-based serving and direct weights serving

### `predict_image_bytes(image_bytes)`

This is one of the most important functions in the whole project.

What it does in MLflow mode:

1. converts image bytes to base64 text
2. builds a one-row DataFrame
3. calls the MLflow PyFunc model
4. converts output into a `Prediction`

What it does in weights mode:

1. decodes bytes into an image
2. converts the image into a numpy array
3. runs YOLO directly
4. normalizes output into a `Prediction`

Why it matters:

- this is the bridge between raw request bytes and structured prediction results

### `_format_ultralytics(result)`

What it does:

- converts YOLO result tensors into Python lists

Why it matters:

- APIs and logs need JSON-friendly data

## 13. `api/main.py`

This is the main runtime service.

If you only remember one file for serving, remember this one.

Its responsibilities are:

- startup configuration
- model loading
- security and middleware
- UI serving
- health endpoints
- prediction endpoint
- metrics endpoint
- logging

### Global App Setup

The file creates:

- a FastAPI app
- Prometheus counters and histograms
- a request ID context variable
- a static mount for frontend assets

Why this matters:

- these are long-lived application-level resources

### `_env_int(...)`

Purpose:

- safely read an integer environment variable

Why it matters:

- environment values are strings by default

### `_env_csv(...)`

Purpose:

- split a comma-separated environment variable into a list

Used for:

- allowed hosts
- CORS origins

### `_configure_logging()`

Purpose:

- set the logging level and format

### `_json_log(event, **fields)`

Purpose:

- write structured logs as JSON text

Why structured logs are useful:

- easier to search
- easier for log platforms to parse

### `_request_context(request, call_next)`

This is middleware.

Middleware means:

- code that runs around every request

What it does:

1. creates or reads a request ID
2. stores it in request context
3. measures latency
4. adds security and tracing headers
5. writes a structured access log

Why it matters:

- every request becomes traceable

### Exception Handlers

There are handlers for:

- `HTTPException`
- generic `Exception`

Why they matter:

- consistent JSON error responses
- request IDs stay attached to errors
- debug mode can optionally reveal more details

### `_require_api_key(x_api_key)`

What it does:

- enforces API key authentication only if `API_KEY` is configured

Why this is useful:

- local development can stay simple
- production can require a secret header

### `_max_upload_bytes()`

What it does:

- converts `MAX_UPLOAD_MB` into bytes

Why it matters:

- file size checks prevent oversized uploads

### `_startup()`

This is one of the most important lifecycle functions.

What it does:

1. configures logging
2. sets trusted hosts middleware if needed
3. sets CORS middleware
4. enables GZip compression
5. loads API config
6. configures MLflow tracking
7. resolves which model to load
8. stores config and predictor in `app.state`
9. supports a dummy predictor mode for CI tests

Why `app.state` matters:

- it stores shared objects for the whole process
- requests do not need to reload the model every time

### `ui()` and `ui_alias()`

These return the frontend page.

Routes:

- `/`
- `/ui`

### `health()`

Returns:

```json
{"status": "ok"}
```

Purpose:

- tells you the service process is alive

### `ready()`

Purpose:

- checks whether the predictor has been loaded

Why it differs from health:

- a service can be running but still not ready to serve predictions

### `version()`

Purpose:

- exposes deployment metadata like Git SHA and build time

### `metrics(...)`

Purpose:

- exposes Prometheus metrics in scrape format

### `predict(...)`

This is the main business endpoint.

What it does step by step:

1. starts a timer
2. checks API key if configured
3. reads uploaded file bytes
4. rejects empty files
5. rejects files above the upload limit
6. sends bytes to the predictor
7. increments prediction metrics
8. computes SHA256 for the image
9. builds a JSON record with timestamp and detection results
10. appends one line to the prediction log file
11. returns the same record in the HTTP response
12. updates request count and latency metrics in `finally`

Why this function matters:

- it is where user input becomes model output plus monitoring data

## 14. Frontend Files

The frontend makes the project easier to demonstrate and easier for non-technical users to test.

### `api/frontend/index.html`

Purpose:

- create the page structure

Main UI pieces:

- title and description
- API key input
- file chooser
- image preview
- canvas overlay
- JSON output panel
- status message area

### `api/frontend/styles.css`

Purpose:

- make the page readable and organized

What it controls:

- colors
- spacing
- cards
- grid layout
- responsive behavior
- overlay positioning

### `api/frontend/app.js`

Purpose:

- handle browser-side behavior

Important logic:

- save and load API key from `localStorage`
- preview the selected image
- send the upload request to `/predict`
- read the JSON response
- draw detection boxes on a canvas overlay
- scale model coordinates to displayed image size

Why the scaling logic matters:

- the model predicts using original image pixel coordinates
- the browser may display the image at a different size
- boxes need to be resized to match the displayed image

## 15. Data Scripts

## 16. `scripts/download_mvtec_ad.py`

Purpose:

- make dataset acquisition explicit and reproducible

### `_sha256(path)`

What it does:

- computes a checksum for the downloaded archive

Why it matters:

- helps verify the file is correct and not corrupted

### `_download(url, dst)`

What it does:

1. prepares destination folder
2. downloads to a temporary partial file
3. shows progress
4. renames the file after successful download

Why download to a temporary file first:

- failed downloads should not look complete

### `_extract(archive, out_dir)`

What it does:

- extracts the archive into the output folder

### `main()`

What it does:

- parses CLI arguments
- optionally downloads
- optionally extracts

## 17. `scripts/validate_data.py`

Purpose:

- prevent training from starting with broken input data

### `_file_exists(path)`

Purpose:

- safe existence check that never raises

### `_validate_with_ge(df)`

Purpose:

- run Great Expectations checks on the manifest DataFrame

Current checks:

- `image_path` column exists
- `image_path` is not null
- `image_path` values are strings

### `main()`

Step by step:

1. parse CLI arguments
2. load the manifest CSV
3. make sure `image_path` exists
4. run schema validation
5. verify image file paths
6. optionally verify label paths
7. write `reports/validation.ok`

Why the marker file matters:

- pipelines can treat validation as a formal successful stage

## 18. `scripts/train.py`

This is the central training script.

If `api/main.py` is the central serving file, then `scripts/train.py` is the central training file.

### `EvalMetrics`

Purpose:

- store the two main evaluation metrics used in this project

Fields:

- `map50`
- `map50_95`

### `_load_yaml(path)`

Purpose:

- load YAML safely into a dictionary

### `_extract_eval_metrics(result)`

Purpose:

- read validation metrics from different Ultralytics result formats

Why this helper is needed:

- library objects change across versions
- this code tries several places to find the metric values

### `_get_production_map50(client, model_name)`

Purpose:

- read the current production model's `map50` from MLflow registry tags

Why this matters:

- it supports champion vs challenger comparison

### `main()`

This function runs the full training workflow.

Step by step:

1. parse arguments and environment-driven defaults
2. load training config from `params.yaml`
3. choose dataset YAML
4. configure MLflow
5. set the experiment
6. start an MLflow run
7. log training parameters
8. create a YOLO model
9. train it
10. log training metrics
11. find `best.pt`
12. run validation on the best weights
13. extract evaluation metrics
14. log evaluation metrics to MLflow
15. log `best.pt` as an artifact
16. log an MLflow PyFunc model
17. optionally register the model
18. optionally store metric tags on the model version
19. optionally enforce the quality gate
20. optionally promote the challenger to Production

### Quality Gate Logic

This is a production-style safeguard.

Rule:

- if gate enforcement is enabled and `map50` is too low, stop the pipeline

Why it matters:

- bad models should fail before deployment

### Champion Vs Challenger Logic

Vocabulary:

- champion = current Production model
- challenger = newly trained model

Promotion rule in this repo:

- if there is no Production model, promote
- if challenger metric is missing, current policy still allows promotion
- otherwise promote only if challenger `map50` is better

This is a policy choice, not a law.

In a stricter company setup, you could require:

- no promotion when metrics are missing
- no promotion unless several metrics improve
- no promotion unless approval is recorded

## 19. Monitoring Scripts

## 20. `scripts/set_reference_predictions.py`

Purpose:

- define the baseline prediction behavior

What it does:

- copies `data/predictions.jsonl` to `data/reference_predictions.jsonl`

Why it matters:

- drift needs something to compare against

## 21. `scripts/drift_report.py`

Purpose:

- compare baseline behavior against recent behavior

### `_read_jsonl(path)`

Purpose:

- read line-by-line JSON logs into a DataFrame

### `_explode_predictions(df)`

Purpose:

- convert variable-length prediction arrays into fixed numeric features

Created features:

- `n_boxes`
- `max_score`
- `mean_score`

Why this is needed:

- drift tools work better on fixed columns than on lists of varying size

### `main()`

Step by step:

1. read reference log
2. read current log
3. transform both into comparable numeric features
4. fail if either side is empty
5. create an Evidently report using `DataDriftPreset`
6. save an HTML report

Important limitation:

- this monitors prediction behavior drift
- it does not directly prove real-world accuracy

That would require fresh labeled production data.

## 22. `pipelines/prefect_flow.py`

Purpose:

- combine steps into named, trackable workflows

### `_run(cmd)`

Purpose:

- run subprocess commands with `PYTHONPATH=src`

Why it matters:

- scripts need consistent imports

### `_py(args)`

Purpose:

- use the same Python interpreter as the current environment

Why it matters:

- avoids "python not found" or wrong-environment issues

### `validate_data()`

- runs the validation script

### `train()`

- runs the training script
- optionally lets environment variables override dataset YAML

### `set_reference_predictions()`

- snapshots the current log as baseline

### `drift_report()`

- generates the monitoring report

### `retraining_flow()`

Workflow:

- validate -> train

### `monitoring_flow()`

Workflow:

- set reference -> drift report

## 23. DVC Files

## 24. `dvc.yaml`

Purpose:

- describe the project pipeline as stages

Current stages:

- `download_mvtec_ad`
- `validate`
- `train`

What DVC looks at:

- commands
- dependencies
- outputs
- parameters

Why this matters:

- if inputs change, DVC knows which stage should rerun

## 25. `params.yaml`

Purpose:

- keep training settings outside the code

Current keys:

- model
- data
- epochs
- imgsz
- batch

Why this matters:

- experiments become easier to tweak and reproduce

## 26. Docker And Infrastructure Files

## 27. `docker/Dockerfile.api`

Purpose:

- build the API container image

Main ideas:

- start from Python base image
- copy dependency list
- install packages
- copy source code
- set environment
- expose port 8000
- start Uvicorn

## 28. `docker/Dockerfile.mlflow`

Purpose:

- build a container dedicated to the MLflow server

## 29. `docker-compose.yml`

Purpose:

- start the local multi-service stack

Main services:

- `mlflow`
- `api`
- `nginx`
- `prometheus`
- `grafana`

How they work together:

- API serves predictions
- MLflow stores experiment and model registry data
- Nginx acts as reverse proxy
- Prometheus scrapes metrics
- Grafana visualizes monitoring data

## 30. `nginx/nginx.conf`

Purpose:

- reverse proxy in front of the API

Benefits:

- central entrypoint
- timeout control
- request header forwarding
- browser security headers
- gzip support

## 31. Monitoring Config Files

### `monitoring/prometheus.yml`

Purpose:

- tell Prometheus what to scrape

Current target:

- the API's `/metrics` endpoint

### `monitoring/grafana/provisioning/datasources/datasource.yml`

Purpose:

- automatically connect Grafana to Prometheus at startup

Why it matters:

- no manual UI setup required

## 32. Tests

### `tests/conftest.py`

Purpose:

- shared pytest setup

### `tests/test_api.py`

Purpose:

- verify important API behavior automatically

Examples of what tests protect:

- health endpoints
- UI route
- readiness behavior
- authentication behavior
- prediction endpoint behavior

Why tests matter:

- they reduce accidental regressions

## 33. CI Pipeline

File:

- `.github/workflows/ci.yml`

What it does:

1. checks out the repository
2. sets up Python
3. installs dependencies
4. runs Ruff linting
5. runs pytest

Special note:

- `DISABLE_MODEL_LOAD=1` is used in CI

Why:

- CI should not depend on a heavy real model file
- tests can run with the dummy predictor mode

## 34. End-To-End Request Flow

This section explains what happens when a user uploads an image in the UI.

1. The user opens the web page.
2. The browser loads HTML, CSS, and JavaScript.
3. The user selects an image file.
4. JavaScript previews the image.
5. The user clicks Predict.
6. The browser sends a `POST /predict` request with multipart form data.
7. FastAPI receives the upload.
8. Middleware creates a request ID and starts timing.
9. The API checks the API key if configured.
10. The API reads the image bytes.
11. The API rejects empty or oversized files if needed.
12. The API calls `app.state.predictor.predict_image_bytes(...)`.
13. The predictor runs either:
    - MLflow model prediction, or
    - direct YOLO prediction
14. The API builds a prediction record.
15. The API appends that record to `data/predictions.jsonl`.
16. The API returns the JSON response.
17. The frontend reads the response.
18. JavaScript draws boxes over the displayed image.
19. Metrics and structured logs are updated.

That is the full live prediction path.

## 35. End-To-End Training Flow

1. A dataset YAML and training params are prepared.
2. `scripts/train.py` is started.
3. The script loads config.
4. MLflow tracking is configured.
5. An MLflow experiment run begins.
6. YOLOv8 trains on the dataset.
7. Training metrics are logged.
8. The best weights file is found.
9. Validation runs on the best weights.
10. Evaluation metrics are extracted.
11. Metrics are logged to MLflow.
12. The weights artifact is stored.
13. An MLflow PyFunc model is logged.
14. If configured, the model is registered.
15. If configured, a quality gate is enforced.
16. If configured, the model may be promoted to Production.

That is the full model lifecycle path inside the training script.

## 36. End-To-End Monitoring Flow

1. The API keeps writing prediction logs.
2. At some stable point, `set_reference_predictions.py` creates a baseline snapshot.
3. New predictions continue accumulating.
4. `drift_report.py` compares baseline vs current logs.
5. The script derives simple numeric features from both.
6. Evidently computes drift statistics.
7. An HTML report is written.

This helps answer:

- is the system behaving differently now than before?

## 37. How To Run The Project

This section gives practical commands.

## 38. Install Dependencies

```bash
pip3 install -r requirements.txt -r requirements-mlops.txt -r requirements-dev.txt
```

## 39. Run The API Locally

Set the Python package path:

```bash
export PYTHONPATH=src
```

Serve a local weights file:

```bash
export MODEL_PATH=/absolute/path/to/best.pt
uvicorn api.main:app --reload
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`

## 40. Run With Docker Compose

```bash
docker compose up --build
```

Useful URLs:

- Nginx entrypoint: `http://127.0.0.1:8080`
- API direct: `http://127.0.0.1:8000`
- MLflow: `http://127.0.0.1:5000`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## 41. Train A Model

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT_NAME=defect-detection
export MLFLOW_MODEL_NAME=defect-yolo
PYTHONPATH=src python3 scripts/train.py --data data/dataset.yaml
```

## 42. Enforce The Quality Gate

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_MODEL_NAME=defect-yolo
export ENFORCE_GATE=1
export MIN_MAP50=0.85
export PROMOTE_MODEL=1
PYTHONPATH=src python3 scripts/train.py --data data/dataset.yaml
```

## 43. Run Data Validation

```bash
PYTHONPATH=src python3 scripts/validate_data.py --manifest data/manifest.csv
```

## 44. Download MVTec AD

```bash
python3 scripts/download_mvtec_ad.py --out data/raw/mvtec_ad
```

## 45. Create A Reference Baseline

```bash
python3 scripts/set_reference_predictions.py
```

## 46. Generate A Drift Report

```bash
python3 scripts/drift_report.py
```

## 47. Run Prefect Flow

```bash
python3 pipelines/prefect_flow.py
```

## 48. Use DVC Stages

Run the full DVC pipeline:

```bash
dvc repro
```

Run one stage:

```bash
dvc repro train
```

## 49. Important Environment Variables

### API Variables

- `MODEL_PATH`: path to local YOLO weights
- `MLFLOW_TRACKING_URI`: where MLflow logs and registry live
- `MLFLOW_MODEL_URI`: exact MLflow model URI to serve
- `MLFLOW_MODEL_NAME`: model registry name
- `MLFLOW_MODEL_STAGE`: registry stage like `Production`
- `PREDICTION_LOG_PATH`: JSONL prediction log path
- `API_KEY`: secret for `/predict` and `/metrics`
- `MAX_UPLOAD_MB`: upload size limit
- `LOG_LEVEL`: logging level
- `DEBUG`: detailed 500 errors when set to `1`
- `CORS_ORIGINS`: allowed browser origins
- `ALLOWED_HOSTS`: trusted hostnames
- `DISABLE_MODEL_LOAD`: dummy predictor mode for tests/CI

### Training Variables

- `MLFLOW_EXPERIMENT_NAME`
- `MIN_MAP50`
- `ENFORCE_GATE`
- `PROMOTE_MODEL`

### Dataset Helper Variables

- `MVTEC_AD_URL`
- `MVTEC_AD_ARCHIVE`
- `MVTEC_AD_OUT`

## 50. What The Project Produces

During normal use, the system creates or updates things like:

- `runs/` from YOLO training
- `mlruns/` or MLflow backend data
- `data/predictions.jsonl`
- `data/reference_predictions.jsonl`
- `reports/validation.ok`
- `reports/drift_report.html`

These are the "outputs" of different pipeline steps.

## 51. Why This Project Is Production-Shaped

Many ML repos stop at:

- notebook
- training code
- maybe an API

This repo goes further by including:

- experiment tracking
- model registry support
- quality gating
- promotion policy
- frontend demo
- structured logging
- health and readiness endpoints
- metrics endpoint
- reverse proxy
- monitoring stack
- orchestration
- CI tests
- beginner-friendly documentation

That combination is what makes it MLOps-focused.

## 52. What This Project Does Not Fully Solve Yet

Being honest is also part of good engineering documentation.

This repository is strong as a portfolio and learning project, but a real factory deployment may still need:

- real labeled production data pipelines
- stronger model evaluation suites
- role-based access control
- secret management platform
- persistent external databases
- alerting rules
- long-term artifact storage
- GPU-specific deployment tuning
- Kubernetes manifests or Terraform
- human review workflow for uncertain predictions

That is normal.

This repo is a strong foundation, not the final possible version of an industrial platform.

## 53. Best Reading Order For Beginners

If you are new, read in this order:

1. this file
2. `README.md`
3. `docs/WALKTHROUGH.md`
4. `api/main.py`
5. `src/defect_detection/yolo.py`
6. `scripts/train.py`
7. `scripts/validate_data.py`
8. `scripts/drift_report.py`
9. `pipelines/prefect_flow.py`
10. `docker-compose.yml`

That order moves from big picture to implementation details.

## 54. Final Summary

This project is a full machine learning product skeleton for manufacturing defect detection.

At the model level, it trains YOLOv8.

At the MLOps level, it adds:

- validation
- experiment tracking
- registry support
- serving
- monitoring
- orchestration
- deployment
- testing
- documentation

At the code level, the most important idea is separation of responsibilities:

- helpers live in `src/defect_detection/`
- serving lives in `api/`
- operations live in `scripts/`
- workflows live in `pipelines/`
- infrastructure lives in Docker, Nginx, and monitoring config files

If you understand that separation, you understand the architecture of the whole repository.
