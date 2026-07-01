"""FastAPI application entrypoint.

HOW TO RUN LOCALLY:
    uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

Then visit:
    http://localhost:8000/docs       <- Swagger UI (interactive)
    http://localhost:8000/redoc      <- ReDoc UI (cleaner docs)
    http://localhost:8000/api/v1/health
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import prediction
from src.core.logger import get_logger

logger = get_logger(__name__)

# main.py — just update the existing lifespan function

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up model...")
    try:
        from src.core.config import get_settings
        from src.services.data_7prediction_service import PredictionService
        settings = get_settings()
        svc = PredictionService(settings)
        svc._load_artifacts()
        logger.info("Model warmed up successfully.")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
    yield
    logger.info("API shutting down.")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Production-grade REST API for detecting fraudulent credit card "
        "transactions using XGBoost + SHAP explainability."
    ),
    version="1.0.0",
    contact={
        "name": "Kafil Aslam",
        "url": "https://github.com/Kafil-Aslam-13/credit-card-fraud-detection",
        "email": "aslamkafil13@gmail.com",
    },
    lifespan=lifespan
)

# CORS -- required for Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "https://KAFIL-ASLAM-credit-card-fraud-frontend.hf.space",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)


@app.get("/", tags=["Root"])
def root() -> dict:
    return {
        "message": "Credit Card Fraud Detection API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
    }

