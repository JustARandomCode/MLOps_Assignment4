from typing import Literal

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    area_sqft: float = Field(..., gt=0, le=10000, examples=[1200])
    bedrooms: int = Field(..., ge=1, le=10, examples=[3])
    bathrooms: int = Field(..., ge=1, le=10, examples=[2])
    age_years: int = Field(..., ge=0, le=200, examples=[8])
    location: Literal["Nashik", "Pune", "Mumbai", "Nagpur"] = Field(..., examples=["Pune"])


class Prediction(BaseModel):
    predicted_price: float
    unit: str = "INR lakhs"
    model_version: str
