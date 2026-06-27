

from src.core.config import get_settings
from src.pipelines.prediction_pipeline import run_prediction_pipeline

if __name__ == "__main__":
    settings = get_settings()

    # Replace these zeros with real feature values to test
    sample_transaction = {
        col: 0.0 for col in settings.numeric_columns
    }

    print("\n Running prediction on sample transaction...\n")
    result = run_prediction_pipeline(sample_transaction)

    print(" Prediction Result:")
    print("-" * 40)
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"  Risk Level:  {result['risk_level']}")
    print("\n  Top Features Driving This Prediction:")
    for f in result["top_features"]:
        direction = "→ FRAUD" if f["impact"] > 0 else "→ LEGIT"
        print(f"    {f['feature']:6s}: {f['impact']:+.4f}  {direction}")
    print("-" * 40)