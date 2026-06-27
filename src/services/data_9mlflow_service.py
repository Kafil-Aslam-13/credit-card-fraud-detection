"""Wraps all MLflow experiment tracking calls.
"""

import mlflow
import mlflow.xgboost

from src.core.logger import get_logger

logger = get_logger(__name__)


class MlflowService:

    def __init__(
        self, tracking_url: str, experiment_name: str
    ) -> None:
        mlflow.set_tracking_uri(tracking_url)
        mlflow.set_experiment(experiment_name)
        logger.info(
            f"MLflow initialised → "
            f"url={tracking_url} | experiment={experiment_name}"
        )

    def start_run(self, run_name: str | None = None):
        """Start an MLflow run. Use as a context manager:
            with mlflow_service.start_run("my_run"):
                mlflow_service.log_params(...)
        """
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict) -> None:
        """Log hyperparameters. MLflow flattens nested dicts."""
        # Flatten nested dicts for MLflow
        flat = {}
        for k, v in params.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat[f"{k}.{sub_k}"] = sub_v
            else:
                flat[k] = v
        mlflow.log_params(flat)

    def log_metrics(self, metrics: dict) -> None:
        """Log evaluation metrics — numeric values only.

        WHY FILTER NON-NUMERIC:
        mlflow.log_metrics only accepts int/float. Our metrics dict
        contains confusion_matrix (list) and _roc_curve (dict) which
        would crash mlflow.log_metrics if passed directly.
        """
        numeric = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float))
            and not k.startswith("_")
        }
        mlflow.log_metrics(numeric)
        logger.info(f"Logged {len(numeric)} metrics to MLflow.")

    def log_artifact(self, path: str) -> None:
        """Log a file (plot, JSON, etc.) to MLflow."""
        mlflow.log_artifact(path)

    def log_model(self, model) -> None:
        """Log the trained XGBoost model to MLflow model registry."""
        mlflow.xgboost.log_model(model, artifact_path="model")
        logger.info("Model logged to MLflow.")

    def end_run(self) -> None:
        mlflow.end_run()