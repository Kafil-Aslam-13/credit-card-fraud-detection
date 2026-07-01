import numpy as np
import pytest

from src.core.exceptions import TrainingError
from src.models.evaluator import Evaluator
from src.models.model_factory import ModelFactory
from src.models.trainer import Trainer


def test_model_factory_creates_xgboost():
    model = ModelFactory.create("xgboost", {"n_estimators": 10})
    assert model is not None


def test_model_factory_unknown_model_raises():
    with pytest.raises(TrainingError):
        ModelFactory.create("not-a-real-model", {})


def test_trainer_fits_model():
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, 50)

    model = ModelFactory.create("xgboost", {"n_estimators": 5, "max_depth": 2})
    trainer = Trainer()
    fitted_model = trainer.fit(model, X, y)

    preds = fitted_model.predict(X)
    assert len(preds) == 50


def test_evaluator_returns_expected_keys():
    X = np.random.rand(50, 3)
    y = np.random.randint(0, 2, 50)

    model = ModelFactory.create("xgboost", {"n_estimators": 5, "max_depth": 2})
    model.fit(X, y)

    evaluator = Evaluator()
    metrics = evaluator.evaluate(model, X, y)

    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"]:
        assert key in metrics
