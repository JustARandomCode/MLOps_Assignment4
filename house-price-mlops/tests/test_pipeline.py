import pandas as pd

from src.generate_data import generate
from src.preprocess import clean, engineer


def test_generate_shape_and_columns():
    df = generate(n=100, seed=1)
    assert len(df) == 100
    cols = {"area_sqft", "bedrooms", "bathrooms", "age_years", "location", "price"}
    assert set(df.columns) == cols


def test_generate_is_deterministic():
    pd.testing.assert_frame_equal(generate(n=50, seed=3), generate(n=50, seed=3))


def test_clean_removes_bad_rows():
    df = generate(n=20, seed=1)
    df.loc[0, "area_sqft"] = -1
    df.loc[1, "price"] = None
    assert len(clean(df)) == 18


def test_engineer_adds_features():
    out = engineer(generate(n=20, seed=1))
    assert {"rooms_total", "area_per_bedroom"} <= set(out.columns)
    assert not any(c == "location" for c in out.columns)
