# MLOps Lifecycle - House Price Prediction

**Project:** Predict house sale price (INR lakhs) from area, bedrooms, bathrooms, age and location.
**Type:** Supervised regression. **Business goal:** price estimate within 10% of the true value for at least 70% of listings.

## Lifecycle overview

```
 Problem  ->  Data  ->  Experiment  ->  Validate  ->  Deploy  ->  Monitor
 framing     mgmt      & training      & test        (serve)      & retrain
    ^                                                                 |
    +-----------------------------------------------------------------+
```

## 1. Problem framing
- Define target (`price`), metric (R2, MAE, share of predictions within 10%), and acceptance threshold.
- Output: success criteria written before any modelling.

## 2. Data management
- Source: `data/dataset.csv` (columns: area_sqft, bedrooms, bathrooms, age_years, location, price).
- Versioning: DVC tracks the file; Git tracks the `.dvc` pointer. (Assignment 2)
- Quality checks: nulls, duplicates, impossible values (area <= 0), schema.

## 3. Experimentation and training
- Code in `src/`, hyper-parameters in `params.yaml`, stages in `dvc.yaml`.
- Every experiment = one Git commit + one `dvc repro` run + `metrics/metrics.json`.
- Reproducibility: fixed `random_state`, pinned `requirements.txt`.

## 4. Validation
- Hold-out test split; metrics logged to JSON.
- Automated tests (`pytest`) and linting (`flake8`) gate every pull request. (Assignment 4)
- Model promoted only if metric >= current production model.

## 5. Deployment
- Model packaged as a FastAPI service inside a Docker image. (Assignment 3)
- Environments: local -> staging (on Git tag) -> production.
- Rollback: redeploy previous image tag.

## 6. Monitoring
- Data drift: compare live feature distributions with training data (KS test, PSI).
- Concept drift: track error metrics over time windows against the training baseline. (Assignment 5)
- Alert threshold breach -> trigger retraining pipeline (Airflow DAG).

## 7. Retraining loop
- Trigger: drift alert, schedule, or new labelled data.
- Same pipeline reruns end to end; new model passes the validation gate before release.

## Tool map

| Stage        | Tool                    |
|--------------|-------------------------|
| Code         | Git, GitHub             |
| Data/models  | DVC                     |
| CI/CD        | GitHub Actions          |
| Orchestration| Apache Airflow          |
| Serving      | FastAPI, Docker         |
| Monitoring   | Python (scipy, matplotlib) |

## Roles and artifacts

| Artifact            | Owner stage      | Location                 |
|---------------------|------------------|--------------------------|
| Raw dataset         | Data management  | `data/dataset.csv`       |
| Parameters          | Experimentation  | `params.yaml`            |
| Trained model       | Training         | `models/model.pkl`       |
| Metrics             | Validation       | `metrics/metrics.json`   |
| Docker image        | Deployment       | registry                 |
| Drift report        | Monitoring       | `monitoring/reports/`    |
