"""Classification metric computation, kept separate from the evaluator
so it can be unit-tested with plain arrays and reused anywhere."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(y_true, y_pred, y_proba) -> dict:
    """Return a dict of standard classification metrics.

    Precision/recall/F1 matter far more than accuracy here, since the
    dataset is highly imbalanced (~0.17% fraud) and a model predicting
    "legit" every time would still score >99% accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp=cm.ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),   # legitimate transactions flagged as fraud
        "false_negatives": int(fn),
    }