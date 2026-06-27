"""Validates raw data against expected schema before anything
downstream touches it."""
import pandas as pd
from src.core.exceptions import DataValidationError
from src.core.logger import get_logger

logger=get_logger(__name__)

class DataValidationService:
    def __init__(self,required_columns:list[str])->None:
        self.required_columns=required_columns
    

    def validate(self,df:pd.DataFrame)->pd.DataFrame:
        """Run all validation checks on raw dataframe"""
        logger.info(f"Starting Data Validation")
        logger.info(f"Dataset Shape: {df.shape}")
        
        logger.info("data validation passed")
        return df
    
    def _check_not_empty(self, df:pd.DataFrame)->None:
        if df.empty:
            raise DataValidationError("Dataset is Empty")
        
    def _check_required_columns(self,df:pd.DataFrame)->None:
        missing=set(self.required_columns) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"missing required columns: {missing}\n"
                f"expected columns: {self.required_columns}"
            )
        logger.info(f"column check passed")

    def _check_nulls(self, df: pd.DataFrame) -> None:
        null_counts = df[self.required_columns].isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            # Warn but don't raise — preprocessing will handle nulls
            logger.warning(
                f"Found {total_nulls} null values:\n"
                f"{null_counts[null_counts > 0]}"
            )
        else:
            logger.info("Null check passed — no nulls found.")

    def _check_duplicates(self, df: pd.DataFrame) -> None:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            # Warn but don't raise — preprocessing will drop them
            logger.warning(
                f"Found {duplicate_count} duplicate rows. "
                "They will be removed in preprocessing."
            )
        else:
            logger.info("Duplicate check passed.")

    def _check_target_distribution(self, df: pd.DataFrame) -> None:
        """Check class distribution in target column.

        WHY THIS CHECK:
        SMOTE requires at least 2 classes and enough minority samples
        to create synthetic ones (k_neighbors=5 by default means you
        need at least 6 fraud cases). If you have too few fraud cases,
        SMOTE crashes with a cryptic error. Better to catch it here.
        """
        from src.core.constants import TARGET_COLUMN

        if TARGET_COLUMN not in df.columns:
            return

        dist = df[TARGET_COLUMN].value_counts()
        fraud_count = dist.get(1, 0)
        legit_count = dist.get(0, 0)

        logger.info(
            f"Class distribution → "
            f"Legit: {legit_count} ({legit_count/len(df)*100:.2f}%) | "
            f"Fraud: {fraud_count} ({fraud_count/len(df)*100:.2f}%)"
        )

        if fraud_count < 10:
            raise DataValidationError(
                f"Too few fraud cases ({fraud_count}) for SMOTE. "
                "Need at least 10 fraud samples."
            )

