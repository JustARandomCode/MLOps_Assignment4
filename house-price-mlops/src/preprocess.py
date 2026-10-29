"""Stage 1: clean raw data and engineer features."""
import pandas as pd

RAW = "data/dataset.csv"
OUT = "data/processed.csv"


def clean(df):
    df = df.drop_duplicates().dropna()
    df = df[(df["area_sqft"] > 0) & (df["price"] > 0)]
    return df.reset_index(drop=True)


def engineer(df):
    df = df.copy()
    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]
    df["area_per_bedroom"] = df["area_sqft"] / df["bedrooms"]
    return pd.get_dummies(df, columns=["location"], dtype=int)


if __name__ == "__main__":
    data = engineer(clean(pd.read_csv(RAW)))
    data.to_csv(OUT, index=False)
    print(f"processed {data.shape} -> {OUT}")
