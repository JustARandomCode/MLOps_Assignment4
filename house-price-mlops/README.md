# House Price MLOps

End-to-end MLOps project: a house price prediction model taken from versioned
experiments to a deployed, monitored service. Built incrementally across five
lab assignments; each assignment is tagged in Git (`assignment-1` ... `assignment-5`).

| Assignment | Topic                                   | Git tag        |
|------------|-----------------------------------------|----------------|
| 1          | Git branching + MLOps lifecycle docs    | `assignment-1` |
| 2          | Git + DVC reproducible pipeline         | `assignment-2` |
| 3          | FastAPI + Docker model serving          | `assignment-3` |
| 4          | CI/CD (GitHub Actions) + Airflow DAG    | `assignment-4` |
| 5          | Drift monitoring                        | `assignment-5` |

## Branching strategy

| Branch          | Purpose                                              | Merges into |
|-----------------|------------------------------------------------------|-------------|
| `main`          | Production-ready, tagged releases only               | -           |
| `dev`           | Integration branch; all features land here first     | `main`      |
| `feature/<name>`| One unit of work (data, model, API, docs)            | `dev`       |

Rules:
1. Never commit directly to `main` or `dev`.
2. Branch `feature/*` from `dev`; merge back with a pull request (use `--no-ff` locally).
3. Release: merge `dev` into `main`, then tag (`v0.1.0`).
4. Commit messages: `type: summary` (`feat`, `fix`, `docs`, `exp`, `ci`).

Full lifecycle documentation: [docs/MLOPS_LIFECYCLE.md](docs/MLOPS_LIFECYCLE.md)

---

## Assignment 2: Git + DVC reproducible pipeline

Dataset: synthetic house price data (`python src/generate_data.py`), tracked by DVC.

```
raw data (data/dataset.csv) -> preprocess -> train -> evaluate -> metrics/metrics.json
```

### Reproduce
```bash
git clone <repo-url> && cd house-price-mlops
pip install -r requirements.txt
dvc pull          # fetch data + models from remote
dvc repro         # re-run only what changed
dvc metrics show
```

### Experiments

| Tag     | n_estimators | max_depth | R2     | MAE    | RMSE   | Accuracy (within 10%) |
|---------|--------------|-----------|--------|--------|--------|-----------------------|
| `exp-1` | 100          | 5         | 0.9623 | 8.7398 | 11.624 | 0.655                 |
| `exp-2` | 200          | 10        | 0.9867 | 4.918  | 6.909  | 0.8825                |

Compare: `dvc metrics diff exp-1 exp-2` and `dvc params diff exp-1 exp-2`.
Switch version: `git checkout exp-1 && dvc checkout`.

Note: the task is regression, so classification metrics (precision/recall/F1) do not apply.
`accuracy_within_10pct` = share of predictions within 10% of the true price.

---

## Assignment 3: FastAPI + Docker model serving

| Endpoint        | Method | Purpose                      |
|-----------------|--------|------------------------------|
| `/health`       | GET    | liveness + model loaded      |
| `/predict`      | POST   | price prediction (INR lakhs) |
| `/docs`         | GET    | Swagger UI                   |

### Run locally
```bash
dvc pull                       # or: dvc repro
uvicorn api.main:app --reload
pytest tests/
```

### Run in Docker
```bash
docker build -t house-price-api:1.0 .
docker run -d -p 8000:8000 --name hp-api house-price-api:1.0
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"area_sqft":1200,"bedrooms":3,"bathrooms":2,"age_years":8,"location":"Pune"}'
```
Expected: `{"predicted_price":117.72,"unit":"INR lakhs","model_version":"dev"}`

Invalid input (`area_sqft: -5`) returns HTTP 422.

---

## Assignment 4: CI/CD (GitHub Actions) + Airflow DAG

### PBL: GitHub Actions

| Workflow                         | Trigger               | Action                                              |
|----------------------------------|-----------------------|-----------------------------------------------------|
| `.github/workflows/ci.yml`       | pull request to main/dev | flake8 lint + pytest                             |
| `.github/workflows/train.yml`    | push to `main`        | `dvc repro`, upload model + metrics as artifact     |
| `.github/workflows/deploy-staging.yml` | tag `v*`        | build image, smoke test, push to GHCR (`:staging`)  |

Demo flow:
```bash
git checkout -b feature/demo dev && echo "# x" >> README.md && git commit -am "docs: demo"
git push -u origin feature/demo          # open PR -> CI runs
# merge PR -> merge dev into main         -> Train workflow runs
git tag v0.4.0 && git push origin v0.4.0 # Deploy-to-staging runs
```

### Tool-based: Airflow DAG (`dags/house_price_dag.py`)

```
download_data -> preprocess -> train_model -> evaluate_model -> accuracy_improved
                                                                 |-> deploy_model
                                                                 |-> skip_deployment
```
Gate: candidate `accuracy_within_10pct` must be strictly greater than the production model's.
First run (no production model) always deploys.

```bash
python -m venv af-venv && source af-venv/bin/activate
pip install "apache-airflow==2.10.5" pandas scikit-learn pyyaml joblib \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt
export AIRFLOW_HOME=$PWD/airflow_home AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags \
       AIRFLOW__CORE__LOAD_EXAMPLES=False PROJECT_DIR=$PWD
airflow standalone          # UI on http://localhost:8080, password printed in terminal
```
CLI check: `airflow dags test house_price_training 2026-01-01`
