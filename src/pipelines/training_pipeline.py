

from src.core.config import get_settings
from src.core.logger import get_logger
from src.services.data_5training_service import TrainingService

logger = get_logger(__name__)


def run_training_pipeline() -> dict:
    """Run the full training pipeline and return metrics."""
    settings = get_settings()
    service = TrainingService(settings)
    metrics = service.run()
    return metrics


if __name__ == "__main__":
    run_training_pipeline()