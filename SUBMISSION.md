# Assignment 4 - CI/CD (PBL) + Airflow DAG (tool-based)

> **Zip contents:** source files at the `assignment-4` tag plus `data/dataset.csv` and `models/model.pkl`. It has **no `.git` history** and **no DVC remote**. Branch/tag history, `exp-1`/`exp-2` and the DVC remote are in `house-price-mlops-full.zip`.


## Submit
- PBL: `.github/workflows/ci.yml`, `train.yml`, `deploy-staging.yml` + Actions run screenshots
- Tool: `dags/house_price_dag.py` + Airflow UI screenshots

## PBL steps
```bash
git push -u origin --all && git push origin --tags
git checkout -b feature/ci-demo dev && echo "" >> README.md && git commit -am "docs: ci demo"
git push -u origin feature/ci-demo          # open a PR -> CI workflow runs (lint + tests)
# merge PR into dev, merge dev into main    -> Train workflow runs
git tag v0.4.1 && git push origin v0.4.1    # -> Deploy-to-staging workflow runs
```
Repo Settings > Actions > allow workflows; create an Environment named `staging`.
Screenshots: CI run (green), Train run + artifact, Deploy run (build, smoke test, push), GHCR package page.

## Airflow steps
```bash
python -m venv af-venv && source af-venv/bin/activate
pip install "apache-airflow==2.10.5" pandas scikit-learn pyyaml joblib \
  --constraint https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt
export AIRFLOW_HOME=$PWD/airflow_home AIRFLOW__CORE__DAGS_FOLDER=$PWD/dags \
       AIRFLOW__CORE__LOAD_EXAMPLES=False PROJECT_DIR=$PWD
airflow standalone      # UI http://localhost:8080 (password printed in terminal)
```
In the UI: unpause `house_price_training`, trigger it. Run 1 deploys (no production model yet). Trigger again: `skip_deployment` runs because accuracy is not strictly higher.
Screenshots: DAG graph view, run grid all green, `accuracy_improved` task log (candidate vs production), `deploy_model` log, second run showing the skipped branch.

## Not verified by me
GitHub Actions cannot run here (YAML syntax-checked only). The DAG ran end to end under `airflow dags test` on Airflow 2.10.5 (deploy path, skip path, and improvement path all exercised); the Airflow web UI was not opened.
