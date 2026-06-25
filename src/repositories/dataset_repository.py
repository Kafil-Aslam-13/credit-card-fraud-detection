"""Handles all reading/writing of dataset CSVs.

This is the only layer that should ever call pandas read_csv/to_csv
for raw/interim/processed data. Services depend on this, not on
pandas I/O directly, so swapping CSV for a database later only
means changing this one file.
"""

import os

import pandas as pd

from src.core.exceptions import DataValidationError
from src.core.logger import get_logger

logger = get_logger(__name__)


class DatasetRepository:
    def load_raw(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise DataValidationError(
                f"Raw dataset not found at '{path}'. "
                "Download it from Kaggle and place it there first."
            )
        logger.info(f"Loading raw dataset from {path}")
        return pd.read_csv(path)

    def save_interim(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved interim dataset to {path}")

    def save_processed(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved processed dataset to {path}")

    def load_processed(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise DataValidationError(f"Processed dataset not found at '{path}'.")
        return pd.read_csv(path)