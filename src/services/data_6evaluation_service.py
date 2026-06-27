"""Standalone re-evaluation without retraining.

USE CASE:
You want to test a saved model against new data without
running the full training pipeline again.
"""

from src.core.logger import get_logger
from src.models.evaluator import Evaluator
from src.repositories.artifact_repository import ArtifactRepository
from src.repositories.model_repository import ModelRepository

logger = get_logger(__name__)


class EvaluationService:

    def __init__(self, settings) -> None:
        self.settings = settings
        self.model_repo = ModelRepository(
            settings.model_dir, settings.preprocessor_dir
        )
        self.artifact_repo = ArtifactRepository(
            settings.metrics_dir, settings.reports_dir, settings.shap_dir
        )
        self.evaluator = Evaluator()

    def run(self, X_test, y_test) -> dict:
        model = self.model_repo.load_model()
        metrics = self.evaluator.evaluate(
            model, X_test, y_test, self.settings.eval_threshold
        )
        clean_metrics = {
            k: v for k, v in metrics.items()
            if not k.startswith("_")
        }
        self.artifact_repo.save_metrics(
            clean_metrics, filename="eval_metrics.json"
        )
        logger.info(f"Evaluation complete: {clean_metrics}")
        return clean_metrics