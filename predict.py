from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# ------------------ App Init ------------------
app = FastAPI(
    title="Retail Spend Prediction API",
    description="Predict customer spend for multiple future horizons",
    version="1.0.0"
)

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Features ------------------
FEATURES = [
    "spend_past_30d", "orders_past_30d",
    "spend_past_60d", "orders_past_60d",
    "spend_past_90d", "orders_past_90d",
    "spend_past_180d", "orders_past_180d",
    "recency_days"
]

# ------------------ Request Schema ------------------
class PredictionRequest(BaseModel):
    spend_past_30d: float = Field(..., ge=0)
    orders_past_30d: int = Field(..., ge=0)

    spend_past_60d: float = Field(..., ge=0)
    orders_past_60d: int = Field(..., ge=0)

    spend_past_90d: float = Field(..., ge=0)
    orders_past_90d: int = Field(..., ge=0)

    spend_past_180d: float = Field(..., ge=0)
    orders_past_180d: int = Field(..., ge=0)

    recency_days: int = Field(..., ge=0)

# ------------------ Response Schema ------------------
class PredictionResponse(BaseModel):
    spend_next_30d: float
    spend_next_60d: float
    spend_next_90d: float
    spend_next_180d: float

# ------------------ Load Models ------------------
try:
    models = {
        "30d": joblib.load("models/LinearRegression_30d.joblib"),
        "60d": joblib.load("models/XGBoost_60d.joblib"),
        "90d": joblib.load("models/XGBoost_90d.joblib"),
        "180d": joblib.load("models/XGBoost_180d.joblib")
    }
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# ------------------ Routes ------------------
@app.get("/")
def home():
    return {
        "message": "Retail Spend Prediction API is running 🚀",
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_spend(data: PredictionRequest):
    try:
        X = pd.DataFrame([data.dict()], columns=FEATURES)

        preds = {
            "spend_next_30d": max(models["30d"].predict(X)[0], 0),
            "spend_next_60d": max(models["60d"].predict(X)[0], 0),
            "spend_next_90d": max(models["90d"].predict(X)[0], 0),
            "spend_next_180d": max(models["180d"].predict(X)[0], 0)
        }

        return preds

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
