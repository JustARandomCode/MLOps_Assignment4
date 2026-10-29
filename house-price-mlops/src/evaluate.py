"""Stage 3: evaluate the trained model on the held-out test split."""
import json

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    bundle = joblib.load("models/model.pkl")
    df = pd.read_csv("data/processed.csv")
    X, y = df.drop(columns="price"), df["price"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"])
    pred = bundle["model"].predict(X_test[bundle["features"]])
    metrics = {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "accuracy_within_10pct": round(float(np.mean(np.abs(pred - y_test) / y_test <= 0.10)), 4),
    }
    json.dump(metrics, open("metrics/metrics.json", "w"), indent=2)
    print(metrics)
