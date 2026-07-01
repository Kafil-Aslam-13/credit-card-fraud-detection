"""Tests prediction using a real trained mini-model so it doesn't
depend on artifacts existing on disk -- this is the fast, isolated
unit test loosely-coupled architecture is supposed to enable.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models.predictor import Predictor
from src.models.model_factory import ModelFactory


@pytest.fixture
def trained_model_and_scaler():
    X = pd.DataFrame(np.random.rand(100, 3), columns=["V1", "V2", "Amount"])
    y = np.random.randint(0, 2, 100)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ModelFactory.create("xgboost", {"n_estimators": 5, "max_depth": 2})
    model.fit(X_scaled, y)
    return model, scaler


def test_predictor_single_returns_expected_keys(trained_model_and_scaler):
    model, scaler = trained_model_and_scaler
    row = np.array([[0.5, 0.3, 0.8]])
    row_scaled = scaler.transform(row)

    predictor = Predictor()
    result = predictor.predict_single(model, row_scaled[0])

    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
