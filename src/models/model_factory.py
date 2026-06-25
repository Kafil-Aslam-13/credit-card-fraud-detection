from xgboost import XGBClassifier
from src.core.exceptions import TrainingError
from src.core.logger import get_logger

logger=get_logger(__name__)

class ModelFactory:
    @staticmethod
    def create(model_name:str,params:dict):
        """Instantiate and return a model by name"""
        logger.info(f"Creating Model: {model_name} with params: {params}")

        if model_name.lower()  == "xgboost":
            return XGBClassifier(**params)
        raise TrainingError(
            f"Unknown model name: {model_name} .\n"
            f"supported models: [xgboost]\n"
            f"add new models in src/models/model_fractory.py in ordeer to implement them"
        )