"""CLI entrypoint for training.

Usage:
    python train.py

What happens:
    1. Loads data/raw/creditcard.csv
    2. Validates, cleans, splits, scales, balances
    3. Trains XGBoost model
    4. Evaluates on test set
    5. Saves model + scaler to artifacts/
    6. Logs metrics + plots to MLflow
    7. Prints final metrics to console
"""

from src.pipelines.training_pipeline import run_training_pipeline

if __name__ == "__main__":
    print("\n Starting Credit Card Fraud Detection Training...\n")

    metrics = run_training_pipeline()

    print("\n Training Complete! Final Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {round(value, 4)}")
    print("-" * 40)
    print("\n Model saved to artifacts/models/")
    print(" Scaler saved to artifacts/preprocessors/")
    print(" Metrics saved to artifacts/metrics/")
    print(" Plots saved to artifacts/reports/")
    print(" Run 'mlflow ui' to view experiment tracking\n")


    # First confirm all imports work
