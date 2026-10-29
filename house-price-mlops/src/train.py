"""Stage 2: train the model using parameters from params.yaml."""
import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    df = pd.read_csv("data/processed.csv")
    X, y = df.drop(columns="price"), df["price"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"])
    model = RandomForestRegressor(
        n_estimators=params["model"]["n_estimators"],
        max_depth=params["model"]["max_depth"],
        random_state=params["train"]["random_state"])
    model.fit(X_train, y_train)
    joblib.dump({"model": model, "features": list(X.columns)}, "models/model.pkl")
    print("model saved -> models/model.pkl")
