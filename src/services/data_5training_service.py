"""Orchestrates the full training workflow.

THIS IS WHERE .fit() ACTUALLY GETS CALLED.

The order is:
1.  Load raw data          (DatasetRepository)
2.  Validate               (DataValidationService)
3.  Clean                  (PreprocessingService)
4.  Save interim           (DatasetRepository)
5.  Split + Scale          (PreprocessingService) ← fit_transform here
6.  Balance with SMOTE     (BalancingService)      ← fit_resample here
7.  Create model           (ModelFactory)
8.  Train model            (Trainer)               ← .fit() here
9.  Evaluate               (Evaluator)             ← predict_proba here
10. Save plots             (visualization utils)
11. Save metrics           (ArtifactRepository)
12. Save model + scaler    (ModelRepository)
13. Log to MLflow          (MlflowService)

HOW TO WRITE YOUR OWN:
For any ML project, map out these same steps on paper first.
Then this service is just those steps in code, each delegating
to the appropriate lower-layer class. The service itself has no
pandas, no sklearn, no xgboost imports — it only imports the
classes that wrap those libraries.
"""

import pandas as pd

from src.core.logger import get_logger
from src.models.evaluator import Evaluator
from src.models.model_factory import ModelFactory
from src.models.trainer import Trainer
from src.repositories.artifact_repository import ArtifactRepository
from src.repositories.dataset_repository import DatasetRepository
from src.repositories.model_repository import ModelRepository
from src.services.data_4balancing_service import BalancingService
from src.services.data_2validation_service import DataValidationService
from src.services.data_9mlflow_service import MlflowService
from src.services.data_3preprocessing_service import PreprocessingService
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve

logger = get_logger(__name__)


class TrainingService:

    def __init__(self, settings) -> None:
        # Repositories
        self.dataset_repo = DatasetRepository()
        self.model_repo = ModelRepository(
            settings.model_dir, settings.preprocessor_dir
        )
        self.artifact_repo = ArtifactRepository(
            settings.metrics_dir, settings.reports_dir, settings.shap_dir
        )

        # Services
        self.validation_svc = DataValidationService(settings.required_columns)
        self.preprocessing_svc = PreprocessingService(
            settings.target_column,
            settings.test_size,
            settings.random_state,
        )
        self.balancing_svc = BalancingService(**settings.smote_params)
        self.mlflow_svc = MlflowService(
            settings.mlflow_tracking_url,
            settings.mlflow_experiment_name,
        )

        # Model layer
        self.trainer = Trainer()
        self.evaluator = Evaluator()

        self.settings = settings

    def run(self) -> dict:
        """Execute the full training pipeline end to end.

        Returns:
            Dict of evaluation metrics for the trained model
        """
        settings = self.settings
        settings.ensure_dirs()

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        # ── 1. Load ──────────────────────────────────────────────
        df = self.dataset_repo.load_raw(settings.raw_data_path)

        # ── 2. Validate ──────────────────────────────────────────
        df = self.validation_svc.validate(df)

        # ── 3. Clean ─────────────────────────────────────────────
        df = self.preprocessing_svc.clean(df)

        # ── 4. Save interim checkpoint ───────────────────────────
        self.dataset_repo.save_interim(df, settings.interim_data_path)

        # ── 5. Split + Scale ─────────────────────────────────────
        # fit_transform on train, transform only on test
        X_train, X_test, y_train, y_test, feature_names = (
            self.preprocessing_svc.split_and_scale(df)
        )

        # ── 6. Balance training data with SMOTE ──────────────────
        # NEVER apply SMOTE to test data
        X_train_bal, y_train_bal = self.balancing_svc.balance(
            pd.DataFrame(X_train, columns=feature_names), y_train
        )

        # ── 7. Create model ──────────────────────────────────────
        model = ModelFactory.create(
            settings.model_name, settings.model_params
        )

        # ── 8. Train ─────────────────────────────────────────────
        # THIS IS WHERE .fit() IS CALLED
        model = self.trainer.fit(model, X_train_bal, y_train_bal)

        # ── 9. Evaluate on untouched test set ────────────────────
        metrics = self.evaluator.evaluate(
            model, X_test, y_test, settings.eval_threshold
        )

        # ── 10. Save plots ───────────────────────────────────────
        cm_path = self.artifact_repo.report_path("confusion_matrix.png")
        plot_confusion_matrix(metrics["confusion_matrix"], cm_path)

        roc_path = self.artifact_repo.report_path("roc_curve.png")
        plot_roc_curve(
            metrics["_roc_curve"]["fpr"],
            metrics["_roc_curve"]["tpr"],
            metrics["roc_auc"],
            roc_path,
        )

        # ── 11. Save metrics ─────────────────────────────────────
        clean_metrics = {
            k: v for k, v in metrics.items()
            if not k.startswith("_")  # skip internal keys like _roc_curve
        }
        self.artifact_repo.save_metrics(clean_metrics)

        # ── 12. Save model + scaler ──────────────────────────────
        self.model_repo.save_model(model)
        self.model_repo.save_preprocessor(self.preprocessing_svc.scaler)

        # ── 13. Log everything to MLflow ─────────────────────────
        with self.mlflow_svc.start_run(run_name=f"{settings.model_name}_run"):
            self.mlflow_svc.log_params(settings.model_params)
            self.mlflow_svc.log_params({"smote": settings.smote_params})
            self.mlflow_svc.log_metrics(clean_metrics)
            self.mlflow_svc.log_artifact(cm_path)
            self.mlflow_svc.log_artifact(roc_path)
            self.mlflow_svc.log_model(model)

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info(
            f"f1={clean_metrics['f1_score']:.4f} | "
            f"roc_auc={clean_metrics['roc_auc']:.4f} | "
            f"recall={clean_metrics['recall']:.4f}"
        )
        logger.info("=" * 60)

        return clean_metrics