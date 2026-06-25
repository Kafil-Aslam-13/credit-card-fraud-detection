"""Runs a fitted model against a test set and computes metrics.

WHY predict_proba INSTEAD OF predict:
model.predict() always uses threshold=0.5 internally.
model.predict_proba() gives you raw probabilities, then you apply
your own threshold. This lets you tune the tradeoff between
catching more fraud (lower threshold) vs fewer false alarms
(higher threshold) without retraining the model.

THRESHOLD INTUITION:
threshold=0.5 → flag transaction as fraud if model is >50% sure
threshold=0.3 → flag if >30% sure (catches more fraud, more false alarms)
threshold=0.7 → flag if >70% sure (fewer false alarms, misses more fraud)
Which is better? Depends on your business. Tune via params.yaml.
"""

from sklearn.metrics import roc_curve
from src.core.logger import get_logger
from src.utils.metrics import compute_classification_metrics

logger = get_logger(__name__)

class Evaluator:
    def evaluate(self,model,X_test,y_test,threshold:float=0.5)->dict:
        """Evaluate a fitted model on test data"""
        logger.info(f"Evaluating model with threshold = {threshold}")

        y_proba=model.predict_proba(X_test)[:,1]

        y_pred=(y_proba >= threshold).astype(int)

        metrics = compute_classification_metrics(y_test,y_pred,y_proba)
        fpr ,tpr , _ =roc_curve(y_test,y_proba)
        metrics["_roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
        logger.info(
            f"Results → "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1_score']:.4f} | "
            f"roc_auc={metrics['roc_auc']:.4f} | "
            f"false_negatives={metrics['false_negatives']} "
            f"(real frauds missed)"
        )

        return metrics