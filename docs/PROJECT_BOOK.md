# DefectGuard Project Book

This document is the "read it like a book" guide to the repository.

If you want a more startup- and stakeholder-oriented explanation of the same system, read `docs/STARTUP_ARCHITECTURE.md`.

It explains:

- what this project does
- why each part exists
- how the code is organized
- how data, training, serving, monitoring, and deployment connect together
- what the main functions do
- how to run the project end to end

The writing style is intentionally beginner-friendly. You do not need strong coding experience to follow the main ideas.

## Table Of Contents

- `1.` What This Project Is
- `2.` The Problem It Solves
- `3.` What "End To End" Means Here
- `4.` The System In One Story
- `5.` Project Phases
- `6.` Folder-By-Folder Tour
- `7.` Important Technical Concepts In Plain Language
- `8.` How The Main Code Works
- `9.` `src/defect_detection/config.py`
- `10.` `src/defect_detection/mlflow_utils.py`
- `11.` `src/defect_detection/mlflow_models.py`
- `12.` `src/defect_detection/yolo.py`
- `13.` `api/main.py`
- `14.` Frontend Files
- `15.` Data Scripts
- `16.` `scripts/download_mvtec_ad.py`
- `17.` `scripts/validate_data.py`
- `18.` `scripts/train.py`
- `19.` Monitoring Scripts
- `20.` `scripts/set_reference_predictions.py`
- `21.` `scripts/drift_report.py`
- `22.` `pipelines/prefect_flow.py`
- `23.` DVC Files
- `24.` `dvc.yaml`
- `25.` `params.yaml`
- `26.` Docker And Infrastructure Files
- `27.` `docker/Dockerfile.api`
- `28.` `docker/Dockerfile.mlflow`
- `29.` `docker-compose.yml`
- `30.` `nginx/nginx.conf`
- `31.` Monitoring Config Files
- `32.` Tests
- `33.` CI Pipeline
- `34.` End-To-End Request Flow
- `35.` End-To-End Training Flow
- `36.` End-To-End Monitoring Flow
- `37.` How To Run The Project
- `38.` Install Dependencies
- `39.` Run The API Locally
- `40.` Run With Docker Compose
- `41.` Train A Model
- `42.` Enforce The Quality Gate
- `43.` Run Data Validation
- `44.` Download MVTec AD
- `45.` Create A Reference Baseline
- `46.` Generate A Drift Report
- `47.` Run Prefect Flow
- `48.` Use DVC Stages
- `49.` Important Environment Variables
- `50.` What The Project Produces
- `51.` Why This Project Is Production-Shaped
- `52.` What This Project Does Not Fully Solve Yet
- `53.` Best Reading Order For Beginners
- `54.` Deep Dive Chapters For The Three Most Important Files
- `55.` Deep Dive: Frontend And Deployment Flow
- `56.` Deep Dive: Monitoring, Drift, Tests, And CI
- `57.` How To Study This Repo In 7 Days
- `58.` Final Summary

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

## 54. Deep Dive Chapters For The Three Most Important Files

This appendix goes one level deeper than the earlier sections.

The goal here is not only to say what each file does.

The goal is to help you mentally "walk through the code" in the same order the computer does.

If you can follow these three files, you understand the core logic of the product:

- `api/main.py` controls serving
- `src/defect_detection/yolo.py` controls prediction abstraction
- `scripts/train.py` controls training and model lifecycle decisions

## 54.1 Deep Dive: `api/main.py`

Think of `api/main.py` as the receptionist, security desk, traffic controller, and response writer of the whole system.

It is the file that turns the model into a network service.

### Mental Model

When the API process starts, this file:

1. creates the web app
2. prepares metrics and logging
3. loads configuration
4. loads the model once
5. waits for incoming HTTP requests

When a request arrives, this file:

1. checks the request
2. routes it to the correct function
3. runs business logic
4. returns a response
5. records logs and metrics

### Part 1: Imports

Why are there many imports?

Because this file is where several responsibilities meet:

- `fastapi` handles the web framework
- `prometheus_client` handles metrics
- `logging`, `json`, and `time` handle observability
- `hashlib` handles stable image hashing
- `uuid` and `contextvars` help with request tracing
- `Path` and file operations handle UI files and prediction logs
- project imports like `load_api_config`, `resolve_model_uri`, and `YoloPredictor` connect serving to the rest of the codebase

This is normal for a service entrypoint.

It is not a "small algorithm file".

It is a "coordination file".

### Part 2: Application Creation

The line that creates the `FastAPI` app is the birth of the service.

From that point on, routes and middleware are attached to this `app` object.

You can think of `app` as the central object representing the running web service.

### Part 3: Process-Wide Objects

The file creates a few objects at import time:

- request ID context storage
- Prometheus counters
- Prometheus histograms
- frontend static mounting

Why do this early?

Because these objects should exist for the whole lifetime of the process.

They are not per-request objects.

If they were created on every request:

- performance would be worse
- metrics could break
- behavior could become inconsistent

### Part 4: Helper Functions

#### `_env_int(name, default)`

This helper exists because environment variables are always strings.

If you want `MAX_UPLOAD_MB=10` to behave like a number, you must convert it.

This helper also protects the program:

- if conversion fails
- if someone provides a broken value

then the code falls back to the default.

That is safer than crashing on startup.

#### `_env_csv(name, default)`

Some environment variables contain a comma-separated list.

Examples:

- `CORS_ORIGINS`
- `ALLOWED_HOSTS`

This helper converts one string like:

```text
https://a.com,https://b.com
```

into a Python list like:

```python
["https://a.com", "https://b.com"]
```

That makes later middleware configuration easier.

#### `_configure_logging()`

This is where the service decides how loud or quiet logging should be.

The code reads `LOG_LEVEL` and sets up Python logging.

Why does this matter?

Because in real environments:

- local development may want more detail
- production may want cleaner logs
- debugging incidents may require temporarily more verbosity

#### `_json_log(event, **fields)`

This helper creates structured logs.

Instead of plain text like:

```text
something failed
```

it writes structured JSON-like content with keys such as:

- event
- timestamp
- request ID
- method
- path
- status

Why this is better:

- machines can parse it easily
- log platforms can index fields
- humans can search specific events faster

### Part 5: Request Middleware

#### `_request_context(request, call_next)`

This is one of the most important functions in `api/main.py`.

It runs around every request.

Think of it like airport security and tracking combined.

Before the request reaches the final route function, middleware can:

- inspect headers
- start timers
- attach metadata

After the route finishes, middleware can:

- add response headers
- record metrics
- write logs

This middleware does the following:

1. reads or creates a request ID
2. stores it in a context variable
3. starts a timer
4. calls the real route handler
5. adds tracing and browser security headers
6. logs method, path, status, and duration

Why request ID matters:

- one user request may create several logs
- the request ID lets you connect them

Why timing matters:

- response latency is a core operational metric

Why security headers matter:

- they reduce some browser-based attack risks

### Part 6: Exception Handling

#### `_http_exception_handler(...)`

This handles expected API errors such as:

- unauthorized access
- bad request input
- oversized upload
- missing UI file

Instead of letting error formatting vary, this function guarantees a clean JSON response.

#### `_unhandled_exception_handler(...)`

This is the catch-all safety net.

If something unexpected breaks:

- the event is logged
- the client receives a 500 response
- debug mode can optionally expose more detail

Why this matters:

- production systems should fail in a controlled way

### Part 7: Security Helpers

#### `_require_api_key(x_api_key)`

This function makes authentication optional by configuration.

If `API_KEY` is not set:

- no auth is required

If `API_KEY` is set:

- requests must send the same value in the `X-API-Key` header

This design is practical because:

- local demos stay easy
- real deployments can add a basic protection layer

#### `_max_upload_bytes()`

This helper converts megabytes to bytes.

Why not hardcode bytes directly?

Because humans usually think in MB, not in raw byte counts.

### Part 8: Startup Lifecycle

#### `_startup()`

This function runs once when the API service starts.

This is extremely important.

You do not want to load a model for every request.

That would make the API painfully slow.

So the code loads the predictor once at startup and keeps it in memory.

Step by step, `_startup()`:

1. configures logging
2. reads allowed hosts
3. configures trusted host protection if needed
4. reads CORS settings
5. configures browser access policy
6. enables gzip compression
7. loads API config from environment variables
8. configures MLflow tracking if needed
9. resolves the chosen model source
10. stores config on `app.state`
11. creates either:
    - a dummy predictor for tests, or
    - a real `YoloPredictor`

Why `app.state` is used:

- it is a safe shared place for app-wide objects
- all route functions can access the same predictor and config

Why dummy predictor mode exists:

- CI tests should not depend on a heavy real model
- tests need predictable behavior

### Part 9: UI Routes

#### `ui()` and `ui_alias()`

These functions serve the browser UI.

`ui()`:

- checks whether `index.html` exists
- returns the file if present
- returns 404 if not present

`ui_alias()`:

- simply reuses `ui()`

Why have both `/` and `/ui`?

- convenience for users
- easier demo experience

### Part 10: Operational Endpoints

#### `health()`

This endpoint answers:

"Is the process alive?"

It does not prove the model is loaded.

It only proves the service itself is up enough to answer.

#### `ready()`

This endpoint answers:

"Is the service ready to serve real predictions?"

That is a stronger promise than health.

#### `version()`

This endpoint answers:

"What build of the service is running?"

That becomes useful during deployment debugging.

### Part 11: Metrics Endpoint

#### `metrics(...)`

This endpoint returns Prometheus-formatted metrics.

Prometheus scrapes it on a schedule.

This function is small, but operationally important.

Without it:

- Prometheus sees nothing
- Grafana dashboards stay empty

### Part 12: The Main Business Function

#### `predict(...)`

This is the most important request handler in the entire service.

It is where uploaded bytes become usable machine learning output.

Step by step:

1. start timing
2. assume the request will succeed unless proven otherwise
3. check API key
4. read uploaded image bytes
5. reject an empty file
6. reject files over the allowed size
7. send bytes to the predictor
8. count the successful prediction
9. hash the image for stable identification
10. create a record with timestamp and predictions
11. append the record to the JSONL log
12. return the record to the client
13. if a known API error happened, preserve its status
14. if an unknown error happened, convert it to 500
15. always record request metrics in `finally`

Why hashing is smart:

- you can identify repeated images without storing raw image content in the log

Why logging predictions is smart:

- later monitoring does not need to replay old requests

Why metrics are updated in `finally`:

- success and failure should both appear in operational metrics

### Part 13: The Full Serving Chain

To truly understand `api/main.py`, follow this chain:

1. browser sends request
2. middleware starts trace and timer
3. route validates input
4. route calls predictor
5. predictor returns normalized data
6. route writes log and response
7. middleware writes access log
8. Prometheus metrics are updated

That is the living heart of the online service.

## 54.2 Deep Dive: `src/defect_detection/yolo.py`

This file solves a very important architecture problem:

How can the rest of the system ask for predictions without caring how the model was loaded?

That is the entire reason this file exists.

### Mental Model

This file is an adapter.

An adapter hides complexity behind one clean interface.

Without this file, `api/main.py` would need to know:

- how MLflow models accept input
- how raw YOLO weights accept input
- how to normalize outputs from both branches

That would make the API file much more complicated.

Instead, `YoloPredictor` keeps that complexity here.

### Part 1: `Prediction`

The `Prediction` dataclass is the normalized output contract.

No matter how the model was loaded, the rest of the code can expect:

- `boxes`
- `scores`
- `class_ids`
- `class_names`

This is an important software design decision.

It means:

- logging code stays simple
- API response format stays stable
- frontend code has one predictable shape to read

### Part 2: `YoloPredictor.__init__(...)`

The constructor decides the model-loading strategy.

There are two modes.

#### Mode A: MLflow Mode

If `mlflow_model_uri` is provided:

- the class stores `_kind = "mlflow"`
- it loads a PyFunc model with `mlflow.pyfunc.load_model(...)`

This is useful when:

- the model is stored in MLflow
- you want registry-based serving
- you want URIs like `models:/defect-yolo/Production`

#### Mode B: Direct Weights Mode

If `model_path` is provided instead:

- the class stores `_kind = "weights"`
- it loads a local YOLO `.pt` file directly

This is useful when:

- you have a weights file on disk
- you want simple local serving
- you do not want registry dependency for a demo

#### Why `_kind` Exists

The `_kind` field is a simple switch.

Later, during prediction, the code does not need to guess again.

It already knows which branch to run.

That keeps runtime logic clean.

### Part 3: `predict_image_bytes(image_bytes)`

This is the main function in `yolo.py`.

The input is raw image bytes.

That is exactly what the FastAPI upload handler naturally provides.

So this function is the right place to translate raw bytes into model-friendly input.

#### Branch A: MLflow Prediction

MLflow PyFunc models expect table-like input, usually a DataFrame.

That means raw image bytes cannot just be passed directly.

So the code does this:

1. base64-encode the image bytes
2. convert the bytes into normal UTF-8 text
3. place that text in a one-row pandas DataFrame
4. call `self._model.predict(df)`
5. read the returned object
6. normalize that output into `Prediction`

Why base64 is used:

- DataFrames and JSON-like systems handle text better than raw binary bytes

Why a one-row DataFrame is used:

- PyFunc models are designed for tabular batch-style input

Why output normalization is needed:

- different PyFunc implementations may return:
  - a Python list
  - a pandas DataFrame

This code protects the rest of the system from that variation.

#### Branch B: Direct YOLO Prediction

Raw YOLO weights do not expect a DataFrame.

They work naturally with images.

So the code does this:

1. wrap bytes in an in-memory buffer
2. open the image with PIL
3. convert to RGB
4. convert to a numpy array
5. call YOLO prediction
6. normalize the first result into plain Python lists

Why convert to RGB:

- images may come in grayscale, RGBA, CMYK, or other modes
- converting to RGB creates a more predictable input format

Why convert to numpy:

- Ultralytics accepts numpy image arrays naturally

### Part 4: `_format_ultralytics(result)`

Ultralytics results are rich objects with tensors and metadata.

That is useful for advanced Python code, but not ideal for:

- JSON responses
- logs
- frontend display

So this helper converts those rich objects into plain values.

Step by step:

1. read `result.boxes`
2. handle the no-detections case
3. convert `xyxy` box tensor to a Python list
4. convert confidence scores to a Python list
5. convert class IDs to integers
6. map class IDs to human-readable names
7. return a simple dictionary

Why CPU conversion happens:

- tensors may live on GPU
- JSON serialization requires plain Python-compatible data

### Part 5: Why This File Matters Architecturally

This file keeps the API clean.

The API can say:

```python
pred = predictor.predict_image_bytes(content)
```

and not worry about:

- registry models
- local weight models
- DataFrames
- base64 conversion
- tensor formatting

That is good architecture.

One file owns one hard problem.

## 54.3 Deep Dive: `scripts/train.py`

This file is the main offline machine learning workflow.

If `api/main.py` is the online heart of the product, `scripts/train.py` is the offline heart.

It takes the system from configuration to trained and optionally promoted model.

### Mental Model

This script has five big jobs:

1. read configuration
2. train a YOLO model
3. evaluate the trained model
4. log and package the model in MLflow
5. optionally decide whether it deserves Production

That is why this file feels more complex than a simple training script.

It is not only "fit model and exit".

It is also:

- experiment tracking
- packaging
- quality control
- registry workflow

### Part 1: `EvalMetrics`

This dataclass stores the two evaluation metrics the project cares about most:

- `map50`
- `map50_95`

Why store them in a dataclass?

- it gives a clear named structure
- it is easier to pass around than a loose tuple
- it can be converted to a dictionary cleanly with `asdict(...)`

### Part 2: `_load_yaml(path)`

This helper reads YAML safely.

Why not inline this code inside `main()`?

Because reading configuration is a small reusable responsibility.

Keeping it separate makes the main workflow easier to read.

### Part 3: `_extract_eval_metrics(result)`

This helper exists because third-party libraries do not always expose metrics in one perfectly stable shape.

Ultralytics versions may store metrics in slightly different places.

So this function tries:

- `results_dict`
- multiple key names
- `.box.map50`
- `.box.map`

Why this is good engineering:

- it makes the script more robust across library versions
- it keeps compatibility logic in one place

### Part 4: `_get_production_map50(client, model_name)`

This helper looks up the current Production model in MLflow Model Registry.

Then it tries to read the `val_map50` tag.

Why store the metric as a model version tag?

Because later promotion logic wants a simple answer:

"What is the current Production model's map50?"

If that information lives directly on the model version, comparison becomes easy.

### Part 5: Argument Parsing And Runtime Inputs

At the start of `main()`, the script builds an argument parser.

It supports:

- params file path
- dataset YAML override
- experiment name
- model register name
- minimum map50 threshold
- gate enable flag
- promotion enable flag

Why support both CLI arguments and environment variables?

Because different environments want different control styles:

- humans often prefer CLI overrides
- CI/CD and Docker often prefer environment variables

### Part 6: Loading Training Configuration

The script reads `params.yaml` and focuses on the `train` section.

This section contains values such as:

- model
- data
- epochs
- imgsz
- batch

Then the script decides the final dataset YAML:

- CLI `--data` wins if provided
- otherwise `params.yaml` is used

Why this precedence rule is helpful:

- stable default config in Git
- fast ad hoc overrides when experimenting

### Part 7: MLflow Setup

Before training starts, the script:

- configures the MLflow tracking URI
- sets the experiment name

Why this happens early:

- all later metrics and artifacts need a valid experiment context

### Part 8: Starting The Run

`with mlflow.start_run() as run:` opens the tracking context.

Inside that block:

- params are logged
- metrics are logged
- artifacts are logged
- model registration may happen

Think of an MLflow run as the full official record of one training attempt.

### Part 9: Logging Parameters

The script logs training settings before calling YOLO training.

Why log them?

Because later you want to answer questions like:

- Which learning setup produced this model?
- Which dataset YAML was used?
- How many epochs did we train?
- What image size and batch size were used?

Without parameter logging, experiment comparison becomes weak.

### Part 10: Model Training

The script creates a YOLO object and calls `.train(...)`.

This is the actual model-learning step.

At this stage:

- images are loaded
- labels are used
- optimization runs over epochs
- weights are updated

The result object may include a metrics dictionary and a save directory.

### Part 11: Training Metrics Logging

After training, the script checks `results.results_dict`.

If numeric metrics exist, it logs them to MLflow.

This keeps training information searchable and comparable.

### Part 12: Locating `best.pt`

YOLO training writes outputs under a run directory such as `runs/...`.

The script finds:

```text
weights/best.pt
```

Why this matters:

- later steps depend on the final best weights
- if the file is missing, the training workflow should fail loudly

That is why the script raises `SystemExit` if it cannot find it.

### Part 13: Validation After Training

This is a major production-minded step.

The script does not assume training success means deployment quality.

It reloads the best weights and runs validation:

```python
evaluator = YOLO(str(best_weights))
val_result = evaluator.val(data=data_yaml, verbose=False)
```

Why do this separately?

- training metrics and validation metrics are not the same thing
- deployment decisions should be based on evaluation quality

### Part 14: Evaluation Metrics Extraction

The script calls `_extract_eval_metrics(val_result)`.

This turns a complex library object into the cleaner `EvalMetrics` dataclass.

Then those values are logged to MLflow.

This is important because:

- future comparisons need saved metric values
- registry tagging and promotion logic depend on them

### Part 15: Artifact Logging

The script logs the raw weights file:

- useful for debugging
- useful for direct weights-based serving

Then it logs a PyFunc model:

- useful for standardized MLflow-based serving

This dual logging strategy is practical because it supports two serving styles.

### Part 16: Model Registration

If a register name exists, the script registers the model in MLflow Model Registry.

That creates:

- a model name
- a version number
- a registry object that can later be staged or promoted

Why registration matters:

- it turns a run artifact into a managed model asset

### Part 17: Registry Tags

After registration, the script stores evaluation metrics as model version tags.

Why not only keep metrics in the run?

Because promotion logic later thinks in terms of model versions, not only run history.

Tags make the registry itself more informative.

### Part 18: Quality Gate

This is one of the most important production-minded ideas in the repo.

If `--enforce-gate` is enabled:

- missing `map50` causes failure
- `map50` below threshold causes failure

Why this matters:

- pipelines should stop automatically when quality is not acceptable

Without a gate, weak models can move too far downstream.

### Part 19: Champion Vs Challenger Promotion

If `--promote` is enabled, the script compares the new model against the current Production model.

Terms:

- champion = current Production model
- challenger = newly trained model

Current policy in this repo:

1. if no Production model exists, promote the challenger
2. if challenger metric is missing, still allow promotion
3. otherwise promote only if challenger `map50` is higher

Why have a policy at all?

Because "latest" is not always "best".

This is a simple but meaningful real-world MLOps concept.

### Part 20: Why This Script Is Important

This file is where machine learning work becomes platform work.

It does not only train.

It also:

- records evidence
- packages the result
- applies deployment rules
- updates the registry

That is why it is one of the best files in the repo for understanding MLOps thinking.

### Part 21: The Full Offline Lifecycle

To mentally simulate `scripts/train.py`, remember this chain:

1. read config
2. choose dataset
3. open MLflow run
4. log parameters
5. train YOLO
6. collect training metrics
7. find best weights
8. validate best weights
9. extract evaluation metrics
10. log artifacts and packaged model
11. optionally register
12. optionally gate
13. optionally promote

That is the full offline lifecycle of the model in this repository.

## 55. Deep Dive: Frontend And Deployment Flow

The earlier sections explained what the frontend and deployment files do.

This appendix explains how to mentally follow them as working systems.

These files matter because they answer two practical questions:

- how does a human actually use the model from a browser?
- how do all services run together as one local platform?

## 55.1 Deep Dive: Frontend User Flow

The frontend is intentionally simple.

That simplicity is a strength.

It lets a beginner understand the entire browser interaction without learning a large framework such as React, Vue, or Angular first.

The frontend is made from three files:

- `api/frontend/index.html`
- `api/frontend/styles.css`
- `api/frontend/app.js`

Each file has one main responsibility:

- HTML defines structure
- CSS defines appearance
- JavaScript defines behavior

### Part 1: `index.html`

`index.html` is the skeleton of the page.

If you imagine the page as a building:

- HTML creates the rooms
- CSS paints and sizes the rooms
- JavaScript makes the doors, buttons, and screens respond

The key sections in `index.html` are:

- header area
- control panel
- preview panel
- response panel

#### Header Area

The header tells the user what the product is.

It shows:

- product name
- short explanation

Why this matters:

- a UI should immediately communicate purpose

#### Control Panel

This section contains:

- image file input
- optional API key input
- Predict button

This is the part of the UI where the user gives instructions to the system.

#### Utility Links

The page also includes links to:

- Swagger docs
- `/metrics`
- `/health`

Why include these links?

Because this is not only a customer-facing UI.

It is also a developer and demo UI.

That means it should help someone inspect the backend quickly.

#### Preview Panel

This panel contains:

- the preview image element
- the overlay canvas

Why use both?

Because the uploaded photo and the detection boxes are different things:

- the image element displays the actual uploaded image
- the canvas draws graphics on top of it

That separation makes box drawing easier.

#### Response Panel

This panel shows raw JSON from the backend.

Why show the raw JSON?

- transparency for debugging
- useful for learning what the API returns
- helpful for developers integrating the service

### Part 2: `styles.css`

CSS is sometimes underestimated, but in a product demo it matters a lot.

Why?

Because layout clarity changes how understandable the system feels.

This file is organized around reusable UI patterns.

#### `:root` Variables

The `:root` section stores color and style variables such as:

- background
- panel colors
- border colors
- accent colors
- danger colors
- shadow

Why use variables?

- one place controls the design system
- future style changes become easier

#### Reusable Card Design

The `.panel` class is reused across multiple sections.

Why that is good design:

- consistent appearance
- less duplicated CSS
- easier future maintenance

#### Flex And Grid Layout

The page uses:

- flex rows for controls
- grid layout for the main two-column body

Why use both?

- flex is good for small aligned rows
- grid is good for larger page sections

#### Responsive Behavior

The media query changes the layout when the screen becomes narrow.

Why this matters:

- a modern UI should still be readable on smaller screens

#### Overlay Positioning

One of the most important visual details is:

```css
position: relative;
```

on the image container and:

```css
position: absolute;
inset: 0;
```

on the canvas overlay.

Why?

Because that makes the canvas cover exactly the same visible area as the image.

Without that, detection boxes would not line up correctly.

### Part 3: `app.js`

This file controls browser behavior.

It is the "brains" of the UI.

The browser loads the page first, then this file connects user actions to API requests and drawing logic.

### Part 4: DOM Element References

At the top of the file, JavaScript reads the main HTML elements:

- file input
- API key input
- predict button
- status element
- preview image
- canvas overlay
- JSON output element

Why gather these once at the top?

- later functions can reuse them
- repeated DOM lookups are avoided
- code stays easier to read

### Part 5: API Key Persistence

The constant `API_KEY_STORAGE` defines the localStorage key name.

This is used by:

- `loadApiKey()`
- `saveApiKey()`

Why localStorage is helpful:

- the user does not need to retype the API key every page refresh

This is a convenience feature, not a high-security secret-management system.

That distinction matters.

For a demo UI, convenience is fine.

For a high-security enterprise product, you would usually use a stronger auth flow.

### Part 6: UI Helper Functions

#### `setStatus(text, kind)`

Purpose:

- show small human-readable status messages

Examples:

- `Idle`
- `Image selected`
- `Predicting...`
- `Done`
- error states

Why this matters:

- good UX tells the user what the system is doing

#### `pretty(obj)`

Purpose:

- convert JavaScript objects into readable formatted JSON text

Why this matters:

- raw one-line JSON is hard for humans to read

#### `clearOverlay()`

Purpose:

- remove previously drawn detection boxes

Why this matters:

- old boxes should not remain visible when a new image is selected

#### `resizeCanvasToImage()`

Purpose:

- make the overlay canvas match the displayed size of the image

This is one of the most important frontend details.

If the canvas size and image size differ, boxes will appear in the wrong place.

### Part 7: Detection Box Drawing

#### `drawBoxes(payload)`

This is the key visualization function.

Its job is to turn API prediction results into visible rectangles and labels on the image.

Step by step:

1. clear old drawings
2. confirm `payload.boxes` exists
3. get drawing context from the canvas
4. read the displayed image size
5. compute scale factors from original image size to displayed size
6. prepare drawing styles
7. loop through each predicted box
8. scale each box coordinate to browser coordinates
9. draw the translucent rectangle
10. draw the border
11. compute label text
12. measure text width
13. clamp label position inside the canvas
14. draw label background
15. draw label text

Why the scaling logic matters so much:

The backend returns coordinates relative to the original image pixels.

But the browser may display the image:

- smaller than the original
- larger than the original
- resized to fit the page width

So the code computes:

- `sx`
- `sy`

These are the x and y scale factors.

That is how the system keeps boxes aligned visually.

### Part 8: Event-Driven Browser Logic

Browsers are event-driven.

That means code often runs in response to user actions.

This file listens to several events.

#### File Selection Event

When the user selects a file:

1. JavaScript reads the chosen file
2. updates status
3. clears old JSON output
4. clears old boxes
5. creates a temporary browser preview URL
6. shows the image immediately

Why preview before prediction?

- immediate feedback improves confidence that the right file was selected

#### API Key Change Events

When the user edits the API key field:

- the latest value is saved to localStorage

Why save on both `change` and `keyup`?

- better chance the latest value is preserved

#### Image Load Event

When the image finishes loading:

- the browser finally knows the displayed size
- the overlay canvas is resized to match

Why not do this earlier?

Because before image load completes, the browser may not know the final layout size.

#### Window Resize Event

When the browser window changes size:

- the image layout may change
- the overlay must be resized again

This is another subtle but important detail.

Without it, boxes can become misaligned after resizing the browser window.

### Part 9: Predict Button Flow

The click handler on `predictBtn` is the main frontend workflow.

Step by step:

1. confirm a file exists
2. disable the button to prevent duplicate clicks
3. update status to `Predicting...`
4. build a `FormData` object
5. attach the selected file
6. build optional headers
7. include `X-API-Key` if present
8. call `fetch("/predict", ...)`
9. try to parse JSON response
10. if the response is not OK:
    - show error status
    - show error JSON
    - clear overlay
11. if successful:
    - show JSON response
    - set success status
    - draw boxes
12. handle network errors
13. re-enable the button in `finally`

Why disable the button during the request?

- prevents accidental duplicate requests

Why use `finally`?

- the button must be re-enabled whether the request succeeds or fails

### Part 10: Full Frontend User Journey

To mentally simulate the entire frontend:

1. page loads
2. saved API key is restored
3. user selects an image
4. image preview appears
5. overlay is sized to match image
6. user clicks Predict
7. browser sends multipart request
8. backend returns JSON
9. frontend shows JSON
10. frontend draws boxes

That is the full client-side story of the product.

## 55.2 Deep Dive: `docker-compose.yml` And Local Platform Runtime

`docker-compose.yml` is the file that turns many separate services into one working local platform.

Without it, you would need to start each service manually.

That is possible, but slower, easier to misconfigure, and harder for new teammates.

### Mental Model

Think of Docker Compose as a local operations blueprint.

It answers:

- what services exist?
- how are they built?
- which ports are exposed?
- which environment variables are passed in?
- which data should persist?
- which service depends on which other service?

### Part 1: Why Multiple Services Exist

This project is not only one Python app.

It is a small platform made from multiple pieces:

- MLflow tracking server
- FastAPI inference API
- Nginx reverse proxy
- Prometheus metrics collector
- Grafana dashboard UI

Each service has one focused responsibility.

That is generally better than putting everything into one giant container.

### Part 2: The `mlflow` Service

This service runs MLflow.

Its job is to store:

- experiment runs
- metrics
- artifacts
- model registry information

Important pieces in Compose:

- build from `docker/Dockerfile.mlflow`
- map port `5000:5000`
- mount a named volume
- set restart policy
- define a healthcheck

Why the port mapping matters:

- `5000:5000` means:
  - host machine port 5000
  - container port 5000

So if you open:

```text
http://127.0.0.1:5000
```

you reach the MLflow UI running inside the container.

Why the named volume matters:

- MLflow data survives container recreation

Why the healthcheck matters:

- Docker can tell whether the service is really responding

### Part 3: The `api` Service

This service runs the FastAPI backend.

Important pieces:

- build from `docker/Dockerfile.api`
- environment variables configure model and logging behavior
- port `8000:8000`
- bind mount `./data:/app/data`
- depends on `mlflow`
- healthcheck on `/health`

Why the `data` mount matters:

The container writes prediction logs to `/app/data/...`.

Because `./data` on the host is mounted there, the files are visible outside the container too.

That makes debugging and monitoring scripts easier.

Why `depends_on` matters:

- it tells Compose the API expects MLflow to be present

It does not guarantee application-level readiness by itself, but it does express startup ordering intent.

Why environment variables matter here:

- the same image can behave differently in different environments without changing code

### Part 4: The `nginx` Service

This service is the public entrypoint of the local stack.

It listens on host port 8080 and forwards traffic to the API container.

Important pieces:

- uses official Nginx image
- maps `8080:80`
- mounts custom `nginx.conf`
- depends on `api`

Why not expose only the API directly?

Because modern production-style systems often place a proxy in front of the app.

That proxy can handle:

- security headers
- timeouts
- compression
- request forwarding
- future TLS termination

### Part 5: `nginx.conf` In Context

The Compose file mounts:

```text
./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
```

This means:

- use the repo's custom config
- mount it read-only into the container

Inside that config:

- gzip is enabled
- all requests are proxied to `http://api:8000`
- host and forwarding headers are preserved
- request IDs are forwarded
- browser security headers are added

Why request ID forwarding matters:

- the proxy and backend can participate in the same request-tracing story

### Part 6: The `prometheus` Service

This service collects metrics.

Important pieces:

- official Prometheus image
- port `9090:9090`
- custom config mounted from `monitoring/prometheus.yml`
- persistent volume
- depends on `api`

Prometheus works by scraping.

That means it repeatedly requests the API's `/metrics` endpoint.

Why this matters:

- service metrics become time series data
- dashboards can then visualize request counts and latencies

### Part 7: The `grafana` Service

This service provides a dashboard UI on top of Prometheus.

Important pieces:

- official Grafana image
- port `3000:3000`
- default admin credentials for local use
- persistent storage volume
- provisioning mount for datasource config

Why provisioning matters:

- Grafana comes up already connected to Prometheus
- users do not need to click through manual setup

### Part 8: Named Volumes

At the bottom of the Compose file, named volumes are defined:

- `mlflow`
- `prometheus`
- `grafana`

Why use named volumes?

- they survive container recreation
- they separate runtime data from the container image

That is an important container concept:

- image = packaged application
- volume = persisted mutable data

### Part 9: Restart Policies

Most services use:

```text
restart: unless-stopped
```

Why this is useful:

- containers restart automatically after failure or daemon restart
- developers do not need to manually relaunch every service in common cases

### Part 10: Healthchecks

Both `mlflow` and `api` define healthchecks.

Healthchecks are important because:

- a started container is not always a healthy application

A process might exist but still be:

- not listening yet
- hung
- misconfigured

Healthchecks give Docker more realistic service health information.

### Part 11: Full Compose Startup Story

To mentally simulate `docker compose up --build`:

1. Docker reads `docker-compose.yml`
2. it builds the custom images
3. it creates networks and volumes
4. it starts the `mlflow` container
5. it starts the `api` container
6. it starts the `nginx` container
7. it starts `prometheus`
8. it starts `grafana`
9. healthchecks begin polling services
10. the host machine can now access the stack through mapped ports

That is the local platform boot process.

### Part 12: Full Request Path Through The Compose Stack

If a user opens:

```text
http://127.0.0.1:8080
```

the request flow is:

1. browser calls host port 8080
2. Docker forwards that to Nginx container port 80
3. Nginx proxies the request to `api:8000`
4. FastAPI handles the request
5. the response travels back through Nginx
6. the browser receives the final page or API response

If Prometheus scrapes metrics:

1. Prometheus container calls the API metrics endpoint
2. API returns Prometheus-formatted metrics
3. Prometheus stores them
4. Grafana reads them from Prometheus
5. Grafana renders dashboards for humans

That is the monitoring data path.

### Part 13: Why This Compose File Matters For Learning

This file is one of the best teaching files in the repo because it shows that MLOps is not only model code.

It also includes:

- runtime services
- networking
- persistence
- health awareness
- observability

If someone understands this file, they start thinking like a platform engineer, not only like a model trainer.

## 56. Deep Dive: Monitoring, Drift, Tests, And CI

This appendix covers the last major operational parts of the project.

These parts are very important because a real machine learning system is not finished when the API returns predictions.

A mature system also needs to answer:

- Is the service healthy?
- Is prediction behavior changing over time?
- Are code changes breaking existing behavior?
- Does every new commit get checked automatically?

That is what this section is about.

## 56.1 Deep Dive: Monitoring And Drift

Monitoring in this project happens at two different levels.

### Level 1: Service Monitoring

This is about the software system itself.

Examples:

- how many requests arrived?
- how long did requests take?
- is the service up?

These questions are answered with:

- Prometheus
- Grafana
- `/metrics`

### Level 2: Model Behavior Monitoring

This is about model output behavior.

Examples:

- are we seeing more detections than before?
- are confidence scores changing over time?
- does the current prediction distribution differ from the baseline?

These questions are answered with:

- prediction logs
- reference snapshots
- drift report generation
- Evidently

Both levels matter.

A system can have:

- healthy infrastructure but drifting model behavior

or:

- stable model behavior but broken infrastructure

Good MLOps watches both.

## 56.2 Deep Dive: `monitoring/prometheus.yml`

This file tells Prometheus where to collect metrics from.

Prometheus uses a pull model.

That means:

- Prometheus does not wait for services to send metrics
- Prometheus actively requests metrics from targets on a schedule

### Key Sections

#### `global.scrape_interval`

This controls how often Prometheus asks targets for new metrics.

In this project it is:

```text
15s
```

That means every 15 seconds Prometheus attempts a fresh scrape.

Why this matters:

- shorter intervals give more detailed data
- longer intervals reduce load

#### `scrape_configs`

This section defines which targets should be scraped.

For this project:

- job name: `api`
- metrics path: `/metrics`
- target: `api:8000`

Why `api:8000` works:

- inside Docker Compose, services can reach each other by service name

So Prometheus does not need `localhost`.

It needs the internal service hostname.

### Mental Model

Every 15 seconds:

1. Prometheus calls `http://api:8000/metrics`
2. the API returns metric values in Prometheus text format
3. Prometheus stores those values as time series

That is the full collection loop.

## 56.3 Deep Dive: Grafana Datasource Provisioning

File:

- `monitoring/grafana/provisioning/datasources/datasource.yml`

This file makes Grafana usable immediately.

Without provisioning:

- Grafana would start
- but a human would still need to open the UI
- manually add Prometheus as a datasource

This file automates that.

### Key Concepts

#### `apiVersion`

This tells Grafana which provisioning file format version is being used.

#### Datasource Definition

The datasource is named `Prometheus` and points to:

```text
http://prometheus:9090
```

Again, this uses the Docker Compose internal service name.

#### `access: proxy`

This means requests go through Grafana, not directly from the browser to Prometheus.

Why this is useful:

- simplifies access pattern
- works well inside the containerized stack

#### `isDefault: true`

This tells Grafana:

- use this datasource by default for dashboards unless another one is specified

### Mental Model

When Grafana starts:

1. it reads provisioning files
2. it creates the Prometheus datasource automatically
3. dashboards can use it immediately

That is a small but valuable operator convenience feature.

## 56.4 Deep Dive: Prediction Logging As Monitoring Input

Before drift reporting can happen, the system needs historical prediction records.

That is why `api/main.py` writes each prediction to:

- `data/predictions.jsonl`

Each line contains fields such as:

- timestamp
- image hash
- boxes
- scores
- class IDs
- class names

Why this matters:

- drift monitoring needs data to compare
- logging prediction behavior creates that data source

This is an important MLOps lesson:

Monitoring is often enabled by earlier design choices in serving code.

If the API did not log prediction results, later drift reporting would be much harder.

## 56.5 Deep Dive: `scripts/drift_report.py`

This file is the center of model behavior monitoring in the repo.

Its job is to compare:

- reference prediction behavior
- current prediction behavior

and generate an HTML report.

### Mental Model

This script does not check true accuracy.

It checks whether prediction behavior looks statistically different.

That is a very important distinction.

It answers:

- "Are outputs changing?"

It does not fully answer:

- "Is the model still correct?"

That second question requires fresh labels.

### Part 1: `_read_jsonl(path)`

Purpose:

- read a JSONL log file into a pandas DataFrame

Step by step:

1. create an empty row list
2. return empty DataFrame if file does not exist
3. open the file
4. strip each line
5. skip empty lines
6. parse JSON for each line
7. build a DataFrame from collected rows

Why JSONL is convenient here:

- each log entry is independent
- no full-file JSON array format is required
- appending in the API stays simple

### Part 2: `_explode_predictions(df)`

This is the most conceptually important helper in the file.

The raw prediction log contains variable-length arrays:

- one image may have 0 boxes
- another may have 3 boxes
- another may have 20 boxes

Drift tools work better on fixed columns.

So the script converts variable-length records into derived numeric features:

- `n_boxes`
- `max_score`
- `mean_score`

Why this is smart:

- these columns are simple
- they summarize behavior
- they are easy for drift tooling to compare across datasets

This is feature engineering for monitoring.

### Part 3: `main()`

The main workflow is:

1. read CLI arguments
2. load reference predictions
3. load current predictions
4. transform both into numeric monitoring features
5. fail if either side is missing
6. create an Evidently `Report`
7. use `DataDriftPreset`
8. run the report
9. save HTML output

### Why The HTML Report Matters

HTML output is useful because:

- it is easy for humans to open
- it can be shared with teammates
- it makes drift easier to inspect visually

### Practical Limitation

This monitoring method is intentionally lightweight.

It is a strong portfolio pattern, but in a larger production system you might later add:

- label-based monitoring
- class-wise drift analysis
- alert thresholds
- scheduled reporting jobs

## 56.6 Deep Dive: Reference Baselines

File:

- `scripts/set_reference_predictions.py`

This script creates the baseline file:

- `data/reference_predictions.jsonl`

Why a baseline is needed:

- drift is always "change compared to something"

Without a reference period, the word "drift" has no anchor.

### Mental Model

The workflow is:

1. let the system run during a stable period
2. decide that period represents normal behavior
3. copy current predictions to the reference file
4. compare future predictions against that reference

This is a simple but effective beginner-friendly baseline strategy.

## 56.7 Deep Dive: End-To-End Monitoring Story

To mentally simulate the monitoring stack:

1. API serves predictions
2. API appends JSONL records
3. Prometheus scrapes service metrics
4. Grafana visualizes service metrics
5. reference prediction snapshot is created
6. new predictions accumulate
7. drift script compares reference vs current logs
8. Evidently generates an HTML report

This means the project monitors both:

- system health
- model behavior

That is the real MLOps mindset.

## 56.8 Deep Dive: Testing Philosophy In This Repo

The tests in this project focus on the API because that is the user-facing contract.

Why API tests are high value here:

- many parts of the platform depend on API stability
- frontend, monitoring, and users all rely on the API response shape

These tests are not huge, but they are strategically chosen.

They protect important guarantees.

## 56.9 Deep Dive: `tests/conftest.py`

This is a small file, but an important one.

Pytest automatically imports `conftest.py` before running tests.

In this project, it modifies Python's import path so tests can import the project package from `src/`.

### Why This Matters

Without this setup, tests might fail because Python cannot locate:

- `defect_detection`

The file computes the absolute `src/` path and inserts it into `sys.path` if needed.

This is a practical test-environment setup detail.

## 56.10 Deep Dive: `tests/test_api.py`

This file verifies core API behavior.

It uses FastAPI's `TestClient`, which lets tests send requests to the application without needing a real external server process.

### Mental Model

Each test follows a pattern:

1. configure environment for test safety
2. import the app
3. create a `TestClient`
4. call an endpoint
5. assert expected behavior

### Why `DISABLE_MODEL_LOAD=1` Is Important

Most tests set:

```text
DISABLE_MODEL_LOAD=1
```

Why?

Because tests should not require:

- a real weights file
- an MLflow server
- slow model loading

The dummy predictor mode makes tests:

- fast
- deterministic
- CI-friendly

### `_make_test_image_bytes()`

This helper generates a tiny in-memory PNG image.

Why do this instead of reading a file from disk?

- keeps tests self-contained
- avoids fixture file management
- avoids dependency on external assets

The function:

1. creates a black numpy image
2. converts it to a PIL image
3. writes it to an in-memory buffer
4. returns raw bytes

That gives the tests realistic upload input without needing real sample files.

### `test_health_endpoint()`

Purpose:

- verify `/health` returns success and the expected JSON structure

Why this matters:

- health endpoints are often used by deployment systems

### `test_ready_endpoint()`

Purpose:

- verify `/ready` reports ready when the dummy predictor is active

Why this matters:

- readiness behavior affects deployment orchestration

### `test_ui_root_serves_html()`

Purpose:

- verify `/` serves the UI page successfully

Why this matters:

- the project includes a user-facing demo frontend

### `test_predict_endpoint_returns_schema()`

Purpose:

- verify `/predict` returns the expected output keys

This is one of the most important tests in the file.

Why?

Because downstream consumers rely on this schema.

If a future edit accidentally removed or renamed:

- `boxes`
- `scores`
- `class_ids`
- `class_names`
- `image_sha256`
- `ts`

then clients could break.

This test protects that contract.

### `test_api_key_auth_for_predict()`

Purpose:

- verify that prediction auth works correctly when `API_KEY` is configured

It checks both:

- unauthorized request returns 401
- authorized request succeeds

Why this matters:

- auth logic is easy to accidentally break during refactors

## 56.11 Deep Dive: CI Workflow

File:

- `.github/workflows/ci.yml`

CI means Continuous Integration.

In practice, it means:

- every push and pull request gets automated checks

That helps catch problems early.

### Mental Model

A CI workflow is like an automated reviewer that never gets tired.

Whenever code changes:

1. GitHub starts a fresh runner machine
2. the repository is checked out
3. Python is installed
4. dependencies are installed
5. linting runs
6. tests run

If something fails, the workflow becomes a visible warning.

### `on: push` And `on: pull_request`

These triggers mean:

- check direct pushes
- also check code proposed through pull requests

That covers common collaboration flows.

### `actions/checkout`

This step downloads the repository contents into the runner.

### `actions/setup-python`

This step installs Python 3.11 and enables pip caching.

Why caching matters:

- repeated runs become faster

### Install Dependencies Step

The workflow installs:

- runtime dependencies
- MLOps dependencies
- developer/testing dependencies

This makes the CI environment capable of:

- importing the app
- running tests
- running lint

### Lint Step

This runs:

```text
ruff check .
```

Linting catches issues such as:

- import problems
- unused code
- style problems
- some correctness issues

Why lint before tests?

- it provides a fast first quality gate

### Test Step

This runs:

```text
pytest
```

with environment variables such as:

- `DISABLE_MODEL_LOAD=1`
- `PYTHONPATH=src`

Why these matter:

- tests use the dummy predictor
- package imports work inside CI

## 56.12 Deep Dive: Why Tests And Monitoring Belong Together

Tests and monitoring serve different time horizons.

Tests help before deployment:

- "Did we break expected behavior?"

Monitoring helps after deployment:

- "Is the live system behaving differently now?"

You need both.

Without tests:

- broken code may reach production

Without monitoring:

- production problems may go unnoticed

This repository includes both, which is one reason it feels more like a real platform than a notebook project.

## 56.13 Final Operational Mental Model

If you want one compact operational summary, remember this sequence:

1. code changes are linted and tested in CI
2. the API exposes health and metrics
3. Prometheus collects service metrics
4. Grafana visualizes those metrics
5. predictions are logged as JSONL
6. reference baselines are created
7. drift reports compare past vs present prediction behavior

That is the complete reliability story of this repository.

## 57. How To Study This Repo In 7 Days

This section is for someone who wants a practical learning plan, not only reference material.

You do not need to understand everything in one sitting.

A better approach is to learn the repository layer by layer.

### Day 1: Understand The Product Story

Read:

- this handbook from section `1` through section `7`
- [README.md](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/README.md)

Goal:

- understand what the project does
- understand why manufacturing defect detection is a good MLOps use case
- understand the major tools in plain language

By the end of Day 1, you should be able to explain the whole project in your own words without looking at code.

### Day 2: Understand The Serving Path

Read:

- [main.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/api/main.py)
- [yolo.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/src/defect_detection/yolo.py)
- handbook section `54.1`
- handbook section `54.2`

Goal:

- understand how an uploaded image becomes a prediction response
- understand middleware, auth, startup, and prediction logging
- understand the abstraction between MLflow serving and direct weights serving

By the end of Day 2, you should be able to trace one `/predict` request from browser to JSON response.

### Day 3: Understand The Training Path

Read:

- [train.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/scripts/train.py)
- [mlflow_models.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/src/defect_detection/mlflow_models.py)
- [mlflow_utils.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/src/defect_detection/mlflow_utils.py)
- handbook section `18`
- handbook section `54.3`

Goal:

- understand how training starts
- understand how MLflow logging works in the repo
- understand quality gates and promotion logic

By the end of Day 3, you should be able to explain how a new model becomes a registered production candidate.

### Day 4: Understand The Data And Pipeline Layer

Read:

- [download_mvtec_ad.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/scripts/download_mvtec_ad.py)
- [validate_data.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/scripts/validate_data.py)
- [dvc.yaml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/dvc.yaml)
- [params.yaml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/params.yaml)
- [prefect_flow.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/pipelines/prefect_flow.py)

Goal:

- understand how data is prepared and validated
- understand how DVC describes pipeline stages
- understand how Prefect orchestrates repeatable flows

By the end of Day 4, you should be able to explain the offline workflow from data to trained model.

### Day 5: Understand The Frontend And Deployment Stack

Read:

- [index.html](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/api/frontend/index.html)
- [styles.css](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/api/frontend/styles.css)
- [app.js](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/api/frontend/app.js)
- [docker-compose.yml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/docker-compose.yml)
- [nginx.conf](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/nginx/nginx.conf)
- handbook section `55`

Goal:

- understand how a non-technical user interacts with the system
- understand how services run together locally
- understand port mapping, volumes, proxying, and browser overlay drawing

By the end of Day 5, you should be able to explain both the UI path and the Docker Compose runtime path.

### Day 6: Understand Monitoring And Reliability

Read:

- [prometheus.yml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/monitoring/prometheus.yml)
- [datasource.yml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/monitoring/grafana/provisioning/datasources/datasource.yml)
- [drift_report.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/scripts/drift_report.py)
- [set_reference_predictions.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/scripts/set_reference_predictions.py)
- handbook section `56.1` through `56.7`

Goal:

- understand the difference between service monitoring and model behavior monitoring
- understand how Prometheus, Grafana, JSONL logs, and Evidently fit together

By the end of Day 6, you should be able to explain how the project observes both system health and model behavior.

### Day 7: Understand Quality Protection

Read:

- [conftest.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/tests/conftest.py)
- [test_api.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/tests/test_api.py)
- [ci.yml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/.github/workflows/ci.yml)
- handbook section `56.8` through `56.13`

Goal:

- understand how the repo protects itself against regressions
- understand why CI and tests matter in MLOps
- understand how dummy predictor mode makes the tests practical

By the end of Day 7, you should be able to explain how the project checks itself before and after deployment.

### End Of The 7 Days

If you complete the full study plan, you should be able to explain:

- the business purpose of the project
- the training path
- the serving path
- the deployment path
- the monitoring path
- the testing and CI path

That means you are not only reading the repo anymore.

You are understanding the system as a whole.

## 58. Final Summary

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
