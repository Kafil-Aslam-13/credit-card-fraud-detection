"""Plotting helpers used by the evaluation and SHAP services.

Kept separate from business logic so services don't need to know
matplotlib internals.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless backend, safe for servers/CI
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(fpr, tpr, auc_score: float, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()