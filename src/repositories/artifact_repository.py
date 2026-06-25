"""Handles persistence of non-model artifacts: metrics JSON, plots,
SHAP outputs. Separated from ModelRepository because these are
reports/by-products, not the model itself.
"""

import os

from src.core.logger import get_logger
from src.utils.common import load_json, save_json

logger = get_logger(__name__)


class ArtifactRepository:
    def __init__(self, metrics_dir: str, reports_dir: str, shap_dir: str) -> None:
        self.metrics_dir = metrics_dir
        self.reports_dir = reports_dir
        self.shap_dir = shap_dir

    def save_metrics(self, metrics: dict, filename: str = "metrics.json") -> str:
        path = os.path.join(self.metrics_dir, filename)
        save_json(metrics, path)
        logger.info(f"Metrics saved to {path}")
        return path

    def load_metrics(self, filename: str = "metrics.json") -> dict:
        path = os.path.join(self.metrics_dir, filename)
        return load_json(path)

    def report_path(self, filename: str) -> str:
        os.makedirs(self.reports_dir, exist_ok=True)
        return os.path.join(self.reports_dir, filename)

    def shap_path(self, filename: str) -> str:
        os.makedirs(self.shap_dir, exist_ok=True)
        return os.path.join(self.shap_dir, filename)