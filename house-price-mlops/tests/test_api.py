from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
GOOD = {"area_sqft": 1200, "bedrooms": 3, "bathrooms": 2, "age_years": 8, "location": "Pune"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert r.json()["predicted_price"] > 0


def test_predict_location_matters():
    mumbai = client.post("/predict", json={**GOOD, "location": "Mumbai"}).json()["predicted_price"]
    nagpur = client.post("/predict", json={**GOOD, "location": "Nagpur"}).json()["predicted_price"]
    assert mumbai > nagpur


def test_predict_invalid_input():
    r = client.post("/predict", json={**GOOD, "area_sqft": -5})
    assert r.status_code == 422
    r = client.post("/predict", json={**GOOD, "location": "Atlantis"})
    assert r.status_code == 422
