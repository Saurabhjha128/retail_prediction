from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Retail Spend Prediction API")

FEATURES = [
    'spend_past_30d','orders_past_30d',
    'spend_past_60d','orders_past_60d',
    'spend_past_90d','orders_past_90d',
    'spend_past_180d','orders_past_180d',
    'recency_days'
]

models = {
    "30d": joblib.load("models/LinearRegression_30d.joblib"),
    "60d": joblib.load("models/XGBoost_60d.joblib"),
    "90d": joblib.load("models/XGBoost_90d.joblib"),
    "180d": joblib.load("models/XGBoost_180d.joblib")
}

@app.get("/")
def home():
    return {"message": "Retail Spend Prediction API is running"}

@app.post("/predict")
def predict_spend(data: dict):
    X = pd.DataFrame([data], columns=FEATURES)

    predictions = {}
    for horizon, model in models.items():
        pred = model.predict(X)[0]
        predictions[horizon] = max(pred, 0)

    return predictions

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Retail Spend Prediction API")

# ------------------ FEATURES ------------------
FEATURES = [
    "spend_past_30d", "orders_past_30d",
    "spend_past_60d", "orders_past_60d",
    "spend_past_90d", "orders_past_90d",
    "spend_past_180d", "orders_past_180d",
    "recency_days"
]

# ------------------ INPUT SCHEMA ------------------
class PredictInput(BaseModel):
    spend_past_30d: float = Field(..., ge=0)
    orders_past_30d: int = Field(..., ge=0)

    spend_past_60d: float = Field(..., ge=0)
    orders_past_60d: int = Field(..., ge=0)

    spend_past_90d: float = Field(..., ge=0)
    orders_past_90d: int = Field(..., ge=0)

    spend_past_180d: float = Field(..., ge=0)
    orders_past_180d: int = Field(..., ge=0)

    recency_days: int = Field(..., ge=0)

# ------------------ LOAD MODELS ------------------
try:
    models = {
        "30d": joblib.load("models/LinearRegression_30d.joblib"),
        "60d": joblib.load("models/XGBoost_60d.joblib"),
        "90d": joblib.load("models/XGBoost_90d.joblib"),
        "180d": joblib.load("models/XGBoost_180d.joblib"),
    }
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ------------------ ROUTES ------------------
@app.get("/")
def home():
    return {"message": "Retail Spend Prediction API is running"}

@app.post("/predict")
def predict_spend(data: PredictInput):
    try:
        # Convert input to DataFrame (safe & ordered)
        X = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)

        predictions = {}

        for horizon, model in models.items():
            pred = float(model.predict(X)[0])
            predictions[horizon] = max(pred, 0)

        return predictions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
