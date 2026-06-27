"""Thin wiring layer for inference."""

from src.core.config import get_settings
from src.core.logger import get_logger
from src.services.data_7prediction_service import PredictionService

logger = get_logger(__name__)


def run_prediction_pipeline(transaction: dict) -> dict:

    settings = get_settings()
    service = PredictionService(settings)
    result = service.predict(transaction, settings.numeric_columns)
    return result


if __name__ == "__main__":
    # Quick manual test with zeroed features
    settings = get_settings()
    sample = {col: 0.0 for col in settings.numeric_columns}
    result = run_prediction_pipeline(sample)
    print(result)