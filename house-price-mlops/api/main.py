"""House price prediction REST API."""
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import HouseFeatures, Prediction
from src.preprocess import engineer

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev")

app = FastAPI(title="House Price API", version="1.0.0")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.get("/health")
def health():
    try:
        get_bundle()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(payload: HouseFeatures):
    bundle = get_bundle()
    row = pd.DataFrame([payload.model_dump()])
    feats = engineer(row)
    feats = feats.reindex(columns=bundle["features"], fill_value=0)
    price = float(bundle["model"].predict(feats)[0])
    return Prediction(predicted_price=round(price, 2), model_version=MODEL_VERSION)
