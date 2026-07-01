"""Pydantic schema for the /predict response body.

Every field here maps to a key in the dict returned by
PredictionService.predict() -- if they don't match,
FastAPI raises a 500 at response time.
"""

from pydantic import BaseModel


class FeatureImpact(BaseModel):
    """A single SHAP feature impact entry."""
    feature: str
    impact: float


class PredictionResponse(BaseModel):
    """Full prediction response returned by /predict endpoint."""
    prediction: str              # "FRAUD" or "LEGITIMATE"
    probability: float           # 0.0 to 1.0
    risk_level: str              # "LOW", "MEDIUM", or "HIGH"
    top_features: list[FeatureImpact]  # top 5 SHAP features

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "FRAUD",
                "probability": 0.9821,
                "risk_level": "HIGH",
                "top_features": [
                    {"feature": "V14", "impact": -0.82},
                    {"feature": "V4",  "impact":  0.61},
                    {"feature": "V12", "impact": -0.43},
                    {"feature": "V10", "impact": -0.38},
                    {"feature": "Amount", "impact": 0.21},
                ]
            }
        }