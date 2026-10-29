"""Airflow DAG: download data -> preprocess -> train -> evaluate -> deploy if accuracy improves.

Written for Apache Airflow 2.9+ (tested layout: standalone mode).
Environment:
  PROJECT_DIR  repo root (default: two levels above this file's folder)
  DATA_URL     optional CSV URL; if unset, the synthetic generator is used
"""
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).resolve().parents[1]))
WORK = PROJECT_DIR / "airflow_runtime" / "workdir"
PROD = PROJECT_DIR / "airflow_runtime" / "production"
sys.path.insert(0, str(PROJECT_DIR))

ACCURACY_KEY = "accuracy_within_10pct"


def download_data():
    import pandas as pd

    WORK.mkdir(parents=True, exist_ok=True)
    url = os.getenv("DATA_URL")
    if url:
        df = pd.read_csv(url)
    else:
        from src.generate_data import generate
        df = generate(n=2000, seed=int(datetime.now().strftime("%d")))  # data changes daily
    df.to_csv(WORK / "raw.csv", index=False)
    print(f"downloaded {len(df)} rows")


def preprocess():
    import pandas as pd
    from src.preprocess import clean, engineer

    df = engineer(clean(pd.read_csv(WORK / "raw.csv")))
    df.to_csv(WORK / "processed.csv", index=False)
    print(f"processed shape {df.shape}")


def train_model():
    import joblib
    import pandas as pd
    import yaml
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    params = yaml.safe_load(open(PROJECT_DIR / "params.yaml"))
    df = pd.read_csv(WORK / "processed.csv")
    X, y = df.drop(columns="price"), df["price"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"])
    model = RandomForestRegressor(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["train"]["random_state"]).fit(X_tr, y_tr)
    joblib.dump({"model": model, "features": list(X.columns)}, WORK / "model.pkl")
    X_te.assign(price=y_te).to_csv(WORK / "test.csv", index=False)


def evaluate_model():
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score

    bundle = joblib.load(WORK / "model.pkl")
    test = pd.read_csv(WORK / "test.csv")
    y = test.pop("price")
    pred = bundle["model"].predict(test[bundle["features"]])
    metrics = {
        "r2": round(float(r2_score(y, pred)), 4),
        "mae": round(float(mean_absolute_error(y, pred)), 4),
        ACCURACY_KEY: round(float(np.mean(np.abs(pred - y) / y <= 0.10)), 4),
    }
    json.dump(metrics, open(WORK / "metrics.json", "w"), indent=2)
    print("candidate metrics:", metrics)


def choose_branch():
    new = json.load(open(WORK / "metrics.json"))[ACCURACY_KEY]
    prod_file = PROD / "metrics.json"
    if not prod_file.exists():
        print("no production model yet -> deploy")
        return "deploy_model"
    old = json.load(open(prod_file))[ACCURACY_KEY]
    print(f"candidate={new} production={old}")
    return "deploy_model" if new > old else "skip_deployment"


def deploy_model():
    PROD.mkdir(parents=True, exist_ok=True)
    shutil.copy(WORK / "model.pkl", PROD / "model.pkl")
    shutil.copy(WORK / "metrics.json", PROD / "metrics.json")
    print("promoted to production: MODEL_PATH=airflow_runtime/production/model.pkl")


def skip_deployment():
    print("accuracy did not improve; keeping current production model")


default_args = {"owner": "ojas", "retries": 1, "retry_delay": timedelta(minutes=1)}

with DAG(
    dag_id="house_price_training",
    description="Train house price model and deploy only if accuracy improves",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "house-price"],
) as dag:
    t_download = PythonOperator(task_id="download_data", python_callable=download_data)
    t_prep = PythonOperator(task_id="preprocess", python_callable=preprocess)
    t_train = PythonOperator(task_id="train_model", python_callable=train_model)
    t_eval = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    t_branch = BranchPythonOperator(task_id="accuracy_improved", python_callable=choose_branch)
    t_deploy = PythonOperator(task_id="deploy_model", python_callable=deploy_model)
    t_skip = PythonOperator(task_id="skip_deployment", python_callable=skip_deployment)

    t_download >> t_prep >> t_train >> t_eval >> t_branch >> [t_deploy, t_skip]
