"""Wraps SHAP explainability. Only file that imports shap directly.

1. Choose the right SHAP explainer for your model type:
   - TreeExplainer  → tree-based models (XGBoost, RandomForest) ← us
   - LinearExplainer → linear models (LogisticRegression)
   - KernelExplainer → any model (slowest, most general)
2. fit_explainer() once after training
3. explain() gives SHAP values for a row
4. top_features() converts raw SHAP values into human-readable output

WHAT SHAP VALUES MEAN:
Each feature gets a SHAP value for a prediction.
Positive SHAP → pushes prediction toward FRAUD
Negative SHAP → pushes prediction toward LEGIT
Magnitude → how much influence that feature had

Example output:
  V14: -0.82  → strongly pushes toward LEGIT
  V4:  +0.61  → pushes toward FRAUD
  Amount: +0.23 → slightly pushes toward FRAUD
"""

import shap

from src.core.exceptions import ExplainabilityError
from src.core.logger import get_logger

logger = get_logger(__name__)


class ShapService:

    def __init__(self) -> None:
        self._explainer = None

    def fit_explainer(self, model) -> None:
        """Build a TreeExplainer from a fitted model.

        Must be called after training, before explain().
        At inference time, load the model first then call this.
        """
        try:
            logger.info("Building SHAP TreeExplainer...")
            self._explainer = shap.TreeExplainer(model)
            logger.info("SHAP explainer ready.")
        except Exception as e:
            raise ExplainabilityError(
                f"Failed to build SHAP explainer: {e}"
            ) from e

    def explain(self, X) -> dict:
        """Compute SHAP values for a row or batch of rows.

        Args:
            X: Feature array (1 row for single prediction,
               multiple rows for batch)

        Returns:
            Dict containing raw shap_values array
        """
        if self._explainer is None:
            raise ExplainabilityError(
                "Explainer not built. Call fit_explainer(model) first."
            )
        try:
            shap_values = self._explainer.shap_values(X)
            return {"shap_values": shap_values}
        except Exception as e:
            raise ExplainabilityError(
                f"SHAP explanation failed: {e}"
            ) from e

    def top_features(
        self,
        shap_values,
        feature_names: list[str],
        top_n: int = 5,
    ) -> list[dict]:
        row = (
            shap_values[0]
            if hasattr(shap_values, "ndim") and shap_values.ndim > 1
            else shap_values
        )

        pairs = list(zip(feature_names, row))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)

        return [
            {"feature": name, "impact": round(float(value), 6)}
            for name, value in pairs[:top_n]
        ]