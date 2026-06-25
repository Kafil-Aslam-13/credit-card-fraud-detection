import numpy as np
from src.core.exceptions import PredictionError
from src.core.logger import get_logger

logger =get_logger(__name__)

class Predictor:
    def predict(self,model, X,threshold:float=0.5) -> dict:
        """run inference on a batch of treansactions"""
        try:
            proba=model.predict_proba(X)[:,1]
            pred=(proba >=threshold).astype(int)

            return{
                "prediction": pred,
                "probability":proba
            }
        except Exception as e:
            raise PredictionError(f" Batch inference failed: {e}") from e
        
    def predict_single(self, model, x_row:np.ndarray,threshold:float=0.5)-> dict:
        try:
            logger.info("running single transaction inference...")
            result = self.predict(model,x_row.reshape(1,-1),threshold)
            prediction = int(result["prediction"][0])
            probability = float(result["probability"][0])

            logger.info(
                f"Prediction: {'FRAUD' if prediction == 1 else 'LEGIT'} "
                f"(probability: {probability:.4f})"
            )

            return {
                "prediction": prediction,       # 0 or 1
                "probability": probability,     # 0.0 to 1.0
            }
        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(
                f"Single transaction inference failed: {e}"
            ) from e
        