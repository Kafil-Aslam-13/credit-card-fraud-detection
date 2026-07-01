"""Prediction API routes.

Each route function does three things only:
1. Receives the validated request (Pydantic does validation)
2. Calls the service
3. Returns the response

No business logic here. No ML code. Just HTTP in, service call, HTTP out.
If anything goes wrong, catch custom exceptions and convert them
to HTTP status codes with clear messages.

"""

from fastapi import APIRouter, HTTPException

from app.api.schemas.request_schema import TransactionRequest
from app.api.schemas.response_schema import PredictionResponse
from src.core.config import get_settings
from src.core.exceptions import FraudDetectionError, ModelNotFoundError
from src.core.logger import get_logger
from src.services.data_7prediction_service import PredictionService

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Fraud Detection"])

# Instantiated once at startup -- model cached after first request
_settings = get_settings()
_prediction_service = PredictionService(_settings)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict if a transaction is fraudulent",
    description=(
        "Accepts a transaction's 30 features (Time, V1-V28, Amount) "
        "and returns a fraud prediction, probability, risk level, "
        "and the top 5 SHAP features driving the decision."
    ),
)
def predict(transaction: TransactionRequest) -> PredictionResponse:
    """Run fraud detection on a single transaction.

    - **prediction**: FRAUD or LEGITIMATE
    - **probability**: fraud probability between 0 and 1
    - **risk_level**: LOW / MEDIUM / HIGH
    - **top_features**: top 5 features influencing the prediction
    """
    try:
        result = _prediction_service.predict(
            transaction.model_dump(),
            _settings.numeric_columns,
        )
        return PredictionResponse(**result)

    except ModelNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not trained yet. "
                "Run 'python train.py' first then restart the API."
            ),
        )

    except FraudDetectionError as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500, detail="Internal server error"
        ) from e


@router.get(
    "/health",
    summary="Health check",
)
def health_check() -> dict:
    return {"status": "ok", "model": "credit-card-fraud-detection"}


@router.get(
    "/model-info",
    summary="Model information",
)
def model_info() -> dict:
    return {
        "model_name": _settings.model_name,
        "threshold": _settings.eval_threshold,
        "features": len(_settings.numeric_columns),
        "feature_names": _settings.numeric_columns,
    }