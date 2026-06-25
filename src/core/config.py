"""Central configuration loader.

Every other module pulls settings from here instead of reading
YAML/env files itself. This is the single place that knows about
configs/config.yaml, configs/params.yaml and configs/schema.yaml.
"""

import os
from functools import lru_cache

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = "configs"


def _load_yaml(filename: str) -> dict:
    path = os.path.join(_CONFIG_DIR, filename)
    with open(path, "r") as f:
        return yaml.safe_load(f)


class Settings:
    """Loads and exposes all project configuration in one object."""

    def __init__(self) -> None:
        self._config = _load_yaml("config.yaml")
        self._params = _load_yaml("params.yaml")
        self._schema = _load_yaml("schema.yaml")

        # --- paths ---
        paths = self._config["paths"]
        self.raw_data_path = paths["raw_data"]
        self.interim_data_path = paths["interim_data"]
        self.processed_train_path = paths["processed_train"]
        self.processed_test_path = paths["processed_test"]
        self.model_dir = paths["model_dir"]
        self.preprocessor_dir = paths["preprocessor_dir"]
        self.metrics_dir = paths["metrics_dir"]
        self.reports_dir = paths["reports_dir"]
        self.shap_dir = paths["shap_dir"]

        # --- mlflow ---
        self.mlflow_tracking_url = self._config["mlflow"]["tracking_url"]
        self.mlflow_experiment_name = self._config["mlflow"]["experiment_name"]

        # --- target ---
        self.target_column = self._config["target_column"]

        # --- logging ---
        self.log_level = self._config["logging"]["level"]

        # --- params ---
        self.test_size = self._params["test_size"]
        self.random_state = self._params["random_state"]
        self.smote_params = self._params["smote"]
        self.model_name = self._params["model"]["name"]
        self.model_params = self._params["model"]["params"]
        self.eval_threshold = self._params["evaluation"]["threshold"]

        # --- schema ---
        self.numeric_columns = self._schema["numeric_columns"]
        self.required_columns = self._schema["required_columns"]

        # --- env-driven (deployment) settings ---
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

    def ensure_dirs(self) -> None:
        """Create every output directory referenced in config.yaml if missing."""
        for d in [
            os.path.dirname(self.interim_data_path),
            os.path.dirname(self.processed_train_path),
            self.model_dir,
            self.preprocessor_dir,
            self.metrics_dir,
            self.reports_dir,
            self.shap_dir,
        ]:
            if d:
                os.makedirs(d, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()