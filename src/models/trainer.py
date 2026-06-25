"""thin wrapper around model.fit """
from src.core.exceptions import TrainingError
from src.core.logger import get_logger

logger=get_logger(__name__)

class Trainer:
    def fit(self , model , X_train, y_train):
        """fit model on training data and return a fitted model"""
        try:
            logger.info(
                f"Training started . "
                f" X shape: {X_train.shape},"
                f"Fraud Cases: {y_train.sum()},"
                f"Legit cases : {(y_train==0).sum()}"
            )

            model.fit(X_train,y_train)
            logger.info("training complete.")
            return model
        
        except Exception as e:
            raise TrainingError(f"Model training Failed: {e}") from e