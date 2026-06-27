"""Orchestrates inference for a single transaction.

THE INFERENCE FLOW:
1. Load model + scaler from disk (once, then cached)
2. Build SHAP explainer from loaded model (once, then cached)
3. Scale the incoming transaction with training scaler
4. Predict fraud probability
5. Classify risk level
6. Generate SHAP explanation
7. Return clean response dict
"""

import pandas as pd

from src.core.constants import (
    FRAUD_LABEL,
    PREDICTION_FRAUD,
    PREDICTION_LEGIT,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
)
from src.core.logger import get_logger
from src.models.predictor import Predictor
from src.repositories.model_repository import ModelRepository
from src.services.data_8shap_service import ShapService

logger = get_logger(__name__)


class PredictionService:

    def __init__(self, settings) -> None:
        self.settings = settings
        self.model_repo = ModelRepository(
            settings.model_dir, settings.preprocessor_dir
        )
        self.predictor = Predictor()
        self.shap_svc = ShapService()

        # Cached artifacts — loaded once on first prediction
        self._model = None
        self._scaler = None

    def _load_artifacts(self):
        """Load model + scaler from disk and build SHAP explainer.

        Called lazily on first prediction. Subsequent calls return
        cached objects without hitting disk again.

        WHY LAZY LOADING:
        At API startup, the model might not exist yet (user hasn't
        trained). Lazy loading lets the API start successfully and
        only fails at prediction time with a clear error message.
        """
        if self._model is None:
            logger.info("Loading model artifacts for the first time...")
            self._model = self.model_repo.load_model()
            self._scaler = self.model_repo.load_preprocessor()
            self.shap_svc.fit_explainer(self._model)
            logger.info("Artifacts loaded and cached.")
        return self._model, self._scaler

    @staticmethod
    def _classify_risk(probability: float) -> str:
        """Convert fraud probability to a human-readable risk level.

        Thresholds:
            < 0.30  → LOW    (likely legitimate)
            0.30-0.70 → MEDIUM (uncertain, review recommended)
            > 0.70  → HIGH   (likely fraud, flag immediately)
        """
        if probability >= 0.7:
            return RISK_HIGH
        if probability >= 0.3:
            return RISK_MEDIUM
        return RISK_LOW

    def predict(
        self, transaction: dict, feature_order: list[str]
    ) -> dict:
        model, scaler = self._load_artifacts()

        # ── 1. Build feature row in correct column order ──────────
        row = pd.DataFrame([transaction])[feature_order]

        # ── 2. Scale using training scaler (transform only) ───────
        row_scaled = scaler.transform(row)

        # ── 3. Predict ────────────────────────────────────────────
        result = self.predictor.predict_single(
            model, row_scaled[0], self.settings.eval_threshold
        )

        # ── 4. Classify risk level ────────────────────────────────
        risk = self._classify_risk(result["probability"])

        # ── 5. SHAP explanation ───────────────────────────────────
        shap_result = self.shap_svc.explain(row_scaled)
        top_features = self.shap_svc.top_features(
            shap_result["shap_values"], feature_order, top_n=5
        )

        return {
            "prediction": (
                PREDICTION_FRAUD
                if result["prediction"] == FRAUD_LABEL
                else PREDICTION_LEGIT
            ),
            "probability": round(result["probability"], 6),
            "risk_level": risk,
            "top_features": top_features,
        }