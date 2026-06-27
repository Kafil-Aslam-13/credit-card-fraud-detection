"""Applies SMOTE to address extreme class imbalance in training data."""

from imblearn.over_sampling import SMOTE
import pandas as pd

from src.core.exceptions import PreprocessingError
from src.core.logger import get_logger

logger=get_logger(__name__)

class BalancingService:

    def __init__(
        self,
        sampling_strategy: float,
        random_state: int,
        k_neighbors: int,
    ) -> None:
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )

    def balance(self, X_train, y_train):
        """Apply SMOTE to training data only.

        Args:
            X_train: Training features (numpy array or DataFrame)
            y_train: Training labels — must contain both classes

        Returns:
            X_resampled: Balanced feature matrix
            y_resampled: Balanced label vector

        Raises:
            PreprocessingError: If SMOTE fails (e.g. too few samples)
        """
        try:
            before = y_train.value_counts().to_dict()
            logger.info(f"Class distribution before SMOTE: {before}")

            X_resampled, y_resampled = self.smote.fit_resample(
                X_train, y_train
            )

            after = pd.Series(y_resampled).value_counts().to_dict()
            logger.info(f"Class distribution after SMOTE:  {after}")
            logger.info(
                f"Training set size: {len(y_train)} → {len(y_resampled)} "
                f"(+{len(y_resampled) - len(y_train)} synthetic samples)"
            )

            return X_resampled, y_resampled

        except Exception as e:
            raise PreprocessingError(
                f"SMOTE balancing failed: {e}\n"
                "Check that training data has enough minority samples."
            ) from e

