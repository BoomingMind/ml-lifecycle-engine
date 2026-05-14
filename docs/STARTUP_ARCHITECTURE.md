# DefectGuard Startup Architecture

This document explains DefectGuard the way you might present it to:

- a startup CTO
- a hiring manager
- a technical founder
- an engineering lead
- a platform team

It is written in a more product-and-architecture style than the beginner docs.

The goal is to answer questions such as:

- what problem does this product solve?
- why is the architecture designed this way?
- what are the business and technical trade-offs?
- how would this system evolve from pilot to production?
- how do the current codebase and infrastructure support that story?

If `docs/MLOPS_FOR_BEGINNERS.md` explains the project slowly for learners, this document explains the same project as a serious startup-grade system.

## 1. Executive Summary

DefectGuard is a production-shaped computer vision MLOps platform for automated manufacturing defect detection.

At a business level, it helps manufacturers move from manual visual inspection to a repeatable AI-assisted inspection workflow.

At a technical level, it combines:

- computer vision training with YOLOv8
- experiment tracking and model lifecycle management with MLflow
- API-based model serving with FastAPI
- lightweight operator-facing UI for inspection and demos
- drift and service monitoring with Evidently, Prometheus, and Grafana
- repeatable workflows using DVC and Prefect
- local platform deployment with Docker Compose and Nginx

This makes it a good example of a startup-ready MVP platform:

- strong enough to demonstrate real product thinking
- modular enough to grow toward enterprise deployment
- documented enough to communicate clearly to technical stakeholders

## 2. Problem Statement

Manufacturing organizations often depend on visual quality control for:

- metal surface defects
- scratches and cracks
- PCB flaws
- textile anomalies
- missing or malformed components

Traditional manual inspection has several limitations:

- inspection quality varies by operator
- repetitive work leads to fatigue
- throughput becomes a bottleneck
- root-cause analysis is hard without structured logs
- deployment of AI models often stops at prototype stage

Most teams can train a model.

Far fewer teams can turn that model into a reliable, governable product.

DefectGuard focuses on that gap.

It treats defect detection not only as a model-training problem, but as a full product and operations problem.

## 3. Product Vision

The product vision is to provide a practical inspection platform with three core outcomes:

1. Detect likely product defects from images
2. Expose that capability through a production-shaped API and UI
3. Manage the model lifecycle from data validation through monitoring

That means the system is designed around both:

- user value
- operational reliability

In startup terms, DefectGuard is not just a demo model.

It is a minimum viable inspection platform.

## 4. Who This Product Is For

### Primary Users

- manufacturing engineers
- quality assurance teams
- operations teams
- data and ML teams

### Secondary Users

- platform engineers
- MLOps engineers
- technical decision-makers
- prospective customers evaluating inspection automation

### Internal Stakeholders

- CTO or Head of Engineering
- ML lead
- product or solutions engineer
- customer-facing demo or pilot team

## 5. Why This Architecture Matters

Many portfolio projects prove only that someone can train a model.

This architecture tries to prove a broader set of capabilities:

- can the system validate its inputs?
- can it track experiments and models?
- can it serve predictions reliably?
- can it monitor runtime and model behavior?
- can it automate key workflows?
- can it be explained clearly to technical stakeholders?

That broader view matters for startup credibility.

A startup-grade project should signal:

- product thinking
- platform thinking
- operability
- change management

DefectGuard is structured to show all four.

## 6. System Overview

The current architecture can be understood as six connected subsystems.

### 1. Data Subsystem

Purpose:

- define and validate the inputs required for training

Key files:

- `scripts/download_mvtec_ad.py`
- `scripts/validate_data.py`
- `data/dataset.yaml`
- `data/manifest.csv`
- `dvc.yaml`

### 2. Training And Model Management Subsystem

Purpose:

- train YOLOv8
- track runs in MLflow
- package and register models

Key files:

- `scripts/train.py`
- `src/defect_detection/mlflow_models.py`
- `src/defect_detection/mlflow_utils.py`

### 3. Serving Subsystem

Purpose:

- expose trained models through a stable API contract

Key files:

- `api/main.py`
- `src/defect_detection/yolo.py`

### 4. Product Experience Subsystem

Purpose:

- give operators, demo users, and stakeholders a direct visual interaction layer

Key files:

- `api/frontend/index.html`
- `api/frontend/app.js`
- `api/frontend/styles.css`

### 5. Monitoring And Reliability Subsystem

Purpose:

- observe system health and model behavior after deployment

Key files:

- `scripts/set_reference_predictions.py`
- `scripts/drift_report.py`
- `monitoring/prometheus.yml`
- `monitoring/grafana/provisioning/datasources/datasource.yml`

### 6. Platform And Workflow Subsystem

Purpose:

- run the stack locally in a repeatable way
- define repeatable workflows
- ensure basic software quality

Key files:

- `pipelines/prefect_flow.py`
- `docker-compose.yml`
- `docker/Dockerfile.api`
- `docker/Dockerfile.mlflow`
- `nginx/nginx.conf`
- `.github/workflows/ci.yml`
- `tests/test_api.py`

## 7. Architectural Principles

This repo is organized around a few principles that are worth naming explicitly.

### Separation Of Responsibilities

The project separates:

- reusable library logic
- API logic
- training scripts
- orchestration
- monitoring
- infrastructure configuration

Why that matters:

- easier maintenance
- easier ownership boundaries
- easier onboarding

### Reproducibility

The architecture avoids hidden behavior where possible.

Examples:

- training config in `params.yaml`
- DVC stage definitions in `dvc.yaml`
- explicit Dockerfiles
- explicit Prefect flows

Why that matters:

- startup teams move fast
- unclear state becomes expensive quickly

### Production-Shaped Interfaces

The repo includes interfaces that real systems need:

- HTTP endpoints
- health checks
- readiness checks
- metrics
- structured logs
- registry-stage model URIs

Why that matters:

- a real product must be operable, not just accurate

### Replaceability

Major components are decoupled enough to evolve later.

Examples:

- predictor abstraction hides MLflow-vs-weights serving details
- frontend is lightweight and separate from backend logic
- monitoring is based on logs and metrics rather than deeply coupled business logic

Why that matters:

- startups change stack choices as they grow

## 8. End-To-End Product Flow

This is the simplest way to explain the whole product in a technical discussion.

1. A product image is captured or selected.
2. Training data is prepared and validated.
3. A YOLOv8 model is trained and logged to MLflow.
4. The model is evaluated and optionally promoted.
5. The API loads the selected model.
6. A user uploads an image in the UI or through the API.
7. The system returns boxes, scores, and labels.
8. The prediction is logged for monitoring.
9. Runtime metrics are scraped by Prometheus.
10. Grafana dashboards visualize service behavior.
11. Drift reports compare current behavior against a reference baseline.

That sequence is simple enough for a hiring manager to follow, but deep enough for a CTO to see real platform thinking behind it.

## 9. Request Path Architecture

When a live prediction request happens, the runtime path is:

1. client or browser sends an image to the API
2. Nginx can proxy the request to FastAPI
3. FastAPI middleware creates request context and timing
4. API checks API key if configured
5. API reads upload bytes
6. API sends bytes to `YoloPredictor`
7. predictor runs either:
   - MLflow model prediction
   - direct YOLO weights prediction
8. API returns JSON response
9. API appends the prediction record to JSONL logs
10. Prometheus metrics are updated

This matters because it shows that the request path is not just inference.

It also includes:

- auth
- observability
- error handling
- logging
- performance tracking

## 10. Training Path Architecture

The training path is equally important to explain.

1. `scripts/train.py` loads configuration
2. MLflow experiment context is set
3. YOLO training runs
4. training metrics are logged
5. best weights are located
6. validation runs on the trained model
7. evaluation metrics are extracted
8. weights are logged as artifacts
9. an MLflow PyFunc model is logged
10. the model may be registered
11. a quality gate may be enforced
12. champion-vs-challenger logic may promote the model

This is the strongest signal in the codebase that the project is built as a lifecycle system rather than a notebook workflow.

## 11. Why YOLOv8 Was Chosen

YOLOv8 is a practical choice for this use case because:

- it is a mainstream object detection framework
- it has a simple training and inference API
- it is fast enough for real-world inspection-style demos
- it maps naturally to defect boxes and confidence scores

From a startup point of view, it is a good choice because it balances:

- implementation speed
- recognizability
- production relevance

## 12. Why MLflow Was Chosen

MLflow gives the project a strong model lifecycle story.

It helps answer:

- which experiment produced this model?
- what parameters were used?
- what metrics did it achieve?
- which artifact should serving load?
- which model version is Production?

That is a strong startup-architecture move because it reduces ambiguity in model operations.

Without MLflow, the project would still train models.

But it would look much less mature.

## 13. Why FastAPI Was Chosen

FastAPI is a good fit because:

- it is lightweight
- it has clean request/response handling
- it supports OpenAPI docs automatically
- it is common in production Python services

From a startup angle, FastAPI is especially useful for:

- fast iteration
- customer demos
- internal integration work
- clear API contracts

## 14. Why The Frontend Exists

The frontend is not only a cosmetic addition.

It adds product value in three ways:

### 1. Demonstration Value

Stakeholders can understand a visual demo faster than they can understand raw API JSON.

### 2. Inspection Value

Bounding box overlays make predictions easier to inspect.

### 3. Sales / Pilot Value

A startup often needs a usable pilot or demo interface before a customer integration is complete.

That is why the lightweight frontend matters.

## 15. Why DVC And Prefect Both Exist

A CTO may ask:

"Why both DVC and Prefect? Aren't they similar?"

The answer:

- DVC defines reproducible ML/data pipeline stages
- Prefect orchestrates multi-step workflows and gives a higher-level flow concept

They are related but not identical.

In this project:

- DVC is the pipeline-as-code layer
- Prefect is the orchestration layer

That separation is reasonable for a learning and platform-style repo.

## 16. Why Monitoring Is Split Into Two Parts

A mature ML system monitors two different things.

### Service Monitoring

Questions:

- Is the API up?
- Is latency increasing?
- Are error rates rising?

Tools:

- Prometheus
- Grafana
- FastAPI metrics

### Model Behavior Monitoring

Questions:

- Are predictions changing?
- Are confidence distributions shifting?
- Is the system producing different output patterns than before?

Tools:

- JSONL logs
- reference snapshots
- Evidently drift reports

This distinction is important in any serious architecture conversation.

A service can be healthy while the model is drifting.

And a model can be stable while the infrastructure is failing.

## 17. Why JSONL Logging Was Chosen

Prediction logs are written as JSONL because:

- append operations are simple
- each request becomes one independent line
- later processing with pandas is easy

For a startup MVP, this is a strong design choice because it is:

- simple
- transparent
- practical

It is not the final answer for large-scale systems, but it is an excellent early-stage operational data pattern.

## 18. Why Docker Compose Was Chosen

Docker Compose is the right local-platform choice here because:

- the stack has multiple services
- teams need repeatable local setup
- onboarding should be fast

It gives a clear local environment for:

- MLflow
- API
- Nginx
- Prometheus
- Grafana

That is exactly what a small startup or internal platform team would want during early development.

## 19. Why Nginx Is In Front Of The API

Nginx adds a more production-shaped network layer.

It helps with:

- proxying
- request forwarding
- future TLS handling
- security headers
- compression

This signals architectural maturity.

Even if the local deployment is simple, the project shows awareness of real edge-proxy patterns.

## 20. Why Tests And CI Matter To The Startup Story

A strong startup technical story is not only:

- we built the product

It is also:

- we can change the product without breaking it every week

Tests and CI help support that story.

### Tests

They protect:

- API schema expectations
- health endpoints
- auth behavior
- readiness behavior

### CI

It protects:

- code style and import quality
- regression detection on every push and pull request

This gives the project better engineering credibility.

## 21. Business Value Of The Current Architecture

This architecture creates value in several ways.

### Faster Demo To Customer

Because there is:

- a UI
- an API
- a local stack
- a clear docs story

the project is demo-ready much faster than a raw model notebook.

### Better Engineering Signal

Because there is:

- MLflow
- DVC
- Prefect
- monitoring
- CI

the project signals stronger production readiness.

### Better Operational Visibility

Because there are:

- health endpoints
- readiness checks
- metrics
- logs
- drift reports

the platform is easier to operate and explain.

## 22. Current Limitations

A CTO-level discussion should also mention limitations honestly.

This system is strong for a startup-grade portfolio or pilot architecture, but not yet a full industrial deployment platform.

Current limitations include:

- local-first infrastructure
- simple JSONL log storage
- no dedicated database layer
- no alerting rules
- no role-based access control
- no secret management platform
- no Kubernetes or Terraform deployment layer
- no long-term object storage strategy
- behavior drift monitoring without labeled production accuracy checks

Naming these limitations is actually a strength.

It shows the architecture is being evaluated realistically.

## 23. Growth Path: MVP To Startup Product

A useful way to present this architecture is by stage.

### Stage 1: Demo / Portfolio / Internal Prototype

Current repo already supports:

- local training
- local serving
- browser demo
- experiment tracking
- basic monitoring

### Stage 2: Pilot Deployment

Next likely additions:

- external artifact storage
- remote MLflow backend
- persistent database for prediction metadata
- customer-specific auth
- alerting
- stronger deployment automation

### Stage 3: Production Platform

Future likely additions:

- Kubernetes or ECS deployment
- secret management
- customer isolation
- full observability stack with alerts
- model approval workflows
- richer production feedback loop

This staged story is useful for communicating roadmap maturity.

## 24. How To Explain This Project In An Interview

If a hiring manager asks what you built, a strong answer would sound like this:

"I built a production-shaped MLOps platform for manufacturing defect detection. It trains YOLOv8 models, tracks experiments and versions with MLflow, serves predictions through FastAPI, includes a browser inspection UI, logs predictions for monitoring, generates drift reports with Evidently, exposes service metrics through Prometheus, and runs locally as a multi-service stack with Docker Compose and Nginx. I also added quality gates, registry promotion logic, tests, CI, and detailed docs so the system is understandable and operable, not just trainable."

That answer communicates:

- business use case
- ML capability
- MLOps maturity
- product thinking

## 25. How To Explain This Project To A CTO

If a CTO asks why the architecture is interesting, a strong answer would be:

"The project is intentionally designed as a lifecycle platform rather than a one-off model demo. It validates data before training, tracks and packages models in MLflow, serves through a production-shaped API, includes observability and post-deployment drift monitoring, and separates concerns cleanly enough to evolve from local pilot to more serious infrastructure later."

That answer emphasizes:

- lifecycle thinking
- change management
- system design maturity

## 26. How To Explain This Project To A Recruiter

A simpler version could be:

"It is an end-to-end AI inspection platform for manufacturing. I did not only train the model. I also built the API, monitoring, registry workflow, deployment stack, tests, and detailed documentation."

That helps non-specialist readers understand the project breadth.

## 27. File References For Architecture Discussions

If you want to support your explanation with code references, these are the most useful anchors:

- [main.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/api/main.py)
- [train.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/scripts/train.py)
- [yolo.py](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/src/defect_detection/yolo.py)
- [docker-compose.yml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/docker-compose.yml)
- [prometheus.yml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/monitoring/prometheus.yml)
- [ci.yml](file:///Users/mdmirajulkarim/Documents/trae_projects/Manufacturing-Defect-Detection-Pipeline/.github/workflows/ci.yml)

## 28. Final Architecture Summary

DefectGuard is a strong startup-style project because it combines:

- a clear industrial use case
- modern computer vision
- practical MLOps tooling
- production-shaped serving
- monitoring and observability
- workflow automation
- strong documentation

It does not pretend to be the final version of an enterprise platform.

Instead, it demonstrates something more useful:

- a realistic early-stage architecture
- a credible technical foundation
- a clear path from prototype to pilot to product

That makes it valuable both as:

- a portfolio project
- a startup architecture case study
