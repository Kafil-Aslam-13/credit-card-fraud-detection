"""Cleans data, splits into train/test, and scales features.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.core.exceptions import PreprocessingError
from src.core.logger import get_logger

logger = get_logger(__name__)

class PreprocessingService:
    def __init__(self,target_column:str,test_size:float,random_state:int)->None:
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler=StandardScaler()
    
    def clean(self,df:pd.DataFrame)->pd.DataFrame:
        original_shape=df.shape
        df = df.drop_duplicates().reset_index(drop=True)
        df=df.dropna().reset_index(drop=True)

        logger.info(
            f"Cleaning complete. "
            f"Before: {original_shape} → After: {df.shape} "
            f"(removed {original_shape[0] - df.shape[0]} rows)"
        )
        return df
    
    def split_and_scale(self,df:pd.DataFrame):
        """Split into train/test and scale features.

        Returns:
            X_train_scaled: numpy array, training features (scaled)
            X_test_scaled:  numpy array, test features (scaled)
            y_train:        Series, training labels
            y_test:         Series, test labels
            feature_names:  list of feature column names

        WHY STRATIFY=y:
        Without stratify, random splitting might put all fraud cases
        in train or test by chance. stratify=y ensures both splits
        maintain the same fraud ratio (~0.17%) as the full dataset.
        """
        try:
            X=df.drop(columns=[self.target_column])
            y=df[self.target_column]
            feature_names = X.columns.tolist()

            X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=self.test_size,random_state=self.random_state,stratify=y)
            X_train_scaled= self.scaler.fit_transform(X_train)
            X_test_scaled=self.scaler.transform(X_test)

            logger.info(
                f"Split complete → "
                f"Train: {X_train_scaled.shape} | "
                f"Test: {X_test_scaled.shape} | "
                f"Fraud in train: {y_train.sum()} | "
                f"Fraud in test: {y_test.sum()}"
            )

            return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

        except Exception as e:
            raise PreprocessingError(f"preprocessing failed : {e}") from e
        
    def transform_single(self, x:pd.DataFrame):
        """Scale a single new transaction at inference time.

        CRITICAL: self.scaler must be the SAME object fitted on
        training data — loaded from disk via ModelRepository.
        This method is called by prediction_service.py after loading
        the scaler, NOT after fitting a new one.
        """
        return self.scaler.transform(x)