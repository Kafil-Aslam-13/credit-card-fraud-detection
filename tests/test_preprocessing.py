import pandas as pd
import pytest

from src.services.preprocessing_service import PreprocessingService


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "V1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "Amount": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Class": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    })


def test_clean_removes_duplicates_and_nulls(sample_df):
    service = PreprocessingService(target_column="Class", test_size=0.2, random_state=42)
    dirty = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    cleaned = service.clean(dirty)
    assert cleaned.duplicated().sum() == 0


def test_split_and_scale_shapes(sample_df):
    service = PreprocessingService(target_column="Class", test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test, feature_names = service.split_and_scale(sample_df)

    assert X_train.shape[0] + X_test.shape[0] == len(sample_df)
    assert X_train.shape[1] == len(feature_names)
    assert "Class" not in feature_names
