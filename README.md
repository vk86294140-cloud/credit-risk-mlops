# credit-risk-mlops

[![CI](https://github.com/vk86294140-cloud/credit-risk-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/vk86294140-cloud/credit-risk-mlops/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An **end-to-end, reproducible MLOps pipeline** that generates data, trains and
**versions** a credit-default risk model, and serves it behind a **FastAPI**
API — containerized and CI-tested.

The emphasis is production ML *engineering*, not a one-off notebook:
reproducibility, train/serve parity, model versioning, schema-validated
serving, tests, and a Docker image that ships ready to score.

```
data ──► features ──► train + evaluate ──► versioned artifact ──► FastAPI /predict
        (ColumnTransformer)      (HistGradientBoosting)   (latest.json)
```

## Highlights

- **Reproducible by construction** — one global seed drives data generation,
  splits, and training. Re-running the pipeline yields the same model and
  metrics.
- **No train/serve skew** — preprocessing lives *inside* the sklearn pipeline,
  so the exact transforms used in training run again at inference.
- **Lightweight model registry** — every run writes a timestamped artifact +
  a metrics manifest and updates a `latest.json` pointer the API loads. Full
  audit trail without a heavyweight platform.
- **Schema-validated API** — Pydantic v2 request/response contracts; invalid
  payloads get a clean `422`. Auto-generated OpenAPI docs at `/docs`.
- **Containerized + CI** — image trains a model at build time and smoke-tests
  `/health`; GitHub Actions runs the test matrix and the container smoke test.

## The model

Predicts probability of loan default from application features (income,
debt-to-income, utilization, delinquencies, employment, loan terms, home
ownership, purpose). Algorithm: `HistGradientBoostingClassifier`. Evaluated
with ROC-AUC, PR-AUC, precision/recall/F1, and Brier score (calibration).

> **On the data:** the default dataset is *synthetic but realistic* — defaults
> are drawn from a documented logistic process over genuine risk drivers (see
> [`src/credit_risk/data.py`](src/credit_risk/data.py)). This keeps the project
> fully reproducible and runnable offline / in CI. Pointing the pipeline at a
> real CSV is a one-liner: `python pipelines/run_pipeline.py --data your.csv`.

## Quickstart

```bash
pip install -e ".[dev]"

# Run the full pipeline: generate data -> train -> evaluate -> register
python pipelines/run_pipeline.py --save-data

# Serve the model
uvicorn credit_risk.api:app --reload --port 8000
```

Score an application:

```bash
curl -s http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 35, "annual_income": 72000, "loan_amount": 15000,
  "loan_term_months": 36, "interest_rate": 12.5, "debt_to_income": 0.28,
  "credit_utilization": 0.45, "num_open_accounts": 6, "num_delinquencies": 0,
  "months_employed": 48, "home_ownership": "mortgage",
  "loan_purpose": "debt_consolidation"
}'
# {"default_probability": 0.07, "risk_band": "low", "model_id": "credit_risk_2026..."}
```

Interactive docs: open `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness probe |
| `GET` | `/model/info` | current model id, features, and metrics |
| `POST` | `/predict` | score one application |
| `POST` | `/predict/batch` | score many applications |

## Run with Docker

```bash
docker compose up --build
# trains a model during the image build, then serves on :8000
curl http://localhost:8000/health
```

## Project layout

```
src/credit_risk/
  config.py     paths, seed, shared feature schema
  data.py       reproducible dataset generation / CSV loading
  features.py   ColumnTransformer (impute + scale + one-hot)
  train.py      train, evaluate, version artifact + metrics
  model.py      load latest artifact, score requests
  schema.py     Pydantic request/response contracts
  api.py        FastAPI app
pipelines/
  run_pipeline.py   one-command reproducible pipeline entrypoint
tests/              data, training, model, and API tests
```

## Testing

```bash
pytest -v
```

Tests cover data reproducibility, that the trained model clears a ROC-AUC bar
and beats the base rate, monotonic risk behaviour (worse credit → higher
probability), and the full API surface including validation errors.

## Deploy

The container is a standard stateless web service — deploy it anywhere that
runs a Docker image (AWS ECS/App Runner, Google Cloud Run, Render, Fly.io). On
AWS SageMaker, the same artifact + `model.py` wrapper drop into an inference
handler. See [`Dockerfile`](Dockerfile) and [`docker-compose.yml`](docker-compose.yml).

## License

MIT — see [LICENSE](LICENSE).
