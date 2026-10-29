"""Generate the synthetic house price dataset (run once; output is tracked by DVC)."""
import sys

import numpy as np
import pandas as pd

LOCATIONS = {"Nashik": 1.0, "Pune": 1.6, "Mumbai": 2.8, "Nagpur": 0.9}


def generate(n=2000, seed=7, shift=False):
    rng = np.random.default_rng(seed)
    loc = rng.choice(list(LOCATIONS), size=n, p=[0.3, 0.3, 0.2, 0.2])
    area = rng.normal(1200, 300, n).clip(400, 3000)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, 5)
    age = rng.integers(0, 40, n)
    if shift:  # used later for drift simulation
        area = area * 1.25
        bedrooms = np.clip(bedrooms + 1, 1, 6)
    mult = np.array([LOCATIONS[x] for x in loc])
    price = (0.045 * area + 6 * bedrooms + 3 * bathrooms - 0.4 * age) * mult
    price = price + rng.normal(0, 4, n)
    return pd.DataFrame({
        "area_sqft": area.round(0), "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age, "location": loc, "price": price.round(2),
    })


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/dataset.csv"
    generate().to_csv(out, index=False)
    print(f"wrote {out}")
