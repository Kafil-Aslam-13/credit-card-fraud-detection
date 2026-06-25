"""Handles persistence of trained models and preprocessors.

Services never call joblib directly to save/load a model -- they go
through this repository. If you switch from joblib to, say, ONNX or
a model registry later, this is the only file that changes.
"""

import os

from src.core.constants import MODEL_FILE_NAME, PREPROCESSOR_FILE_NAME
from src.core.exceptions import ModelNotFoundError
from src.core.logger import get_logger
from src.utils.common import load_joblib, save_joblib

logger = get_logger(__name__)


class ModelRepository:
    def __init__(self, model_dir: str, preprocessor_dir: str) -> None:
        self.model_dir = model_dir
        self.preprocessor_dir = preprocessor_dir

    def save_model(self, model) -> str:
        path = os.path.join(self.model_dir, MODEL_FILE_NAME)
        save_joblib(model, path)
        logger.info(f"Model saved to {path}")
        return path

    def load_model(self):
        path = os.path.join(self.model_dir, MODEL_FILE_NAME)
        if not os.path.exists(path):
            raise ModelNotFoundError(
                f"No trained model found at '{path}'. Run training first."
            )
        return load_joblib(path)

    def save_preprocessor(self, preprocessor) -> str:
        path = os.path.join(self.preprocessor_dir, PREPROCESSOR_FILE_NAME)
        save_joblib(preprocessor, path)
        logger.info(f"Preprocessor saved to {path}")
        return path

    def load_preprocessor(self):
        path = os.path.join(self.preprocessor_dir, PREPROCESSOR_FILE_NAME)
        if not os.path.exists(path):
            raise ModelNotFoundError(
                f"No preprocessor found at '{path}'. Run training first."
            )
        return load_joblib(path)