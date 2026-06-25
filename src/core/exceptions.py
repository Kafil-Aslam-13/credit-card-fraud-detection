"""Custom exception hierarchy for the fraud detection system.

Every layer raises one of these instead of a bare Exception, so callers
(API routes, pipelines, tests) can catch specific failure modes.
"""


class FraudDetectionError(Exception):
    """Base exception for all custom errors in this project."""


class DataValidationError(FraudDetectionError):
    """Raised when input data fails schema or quality checks."""


class PreprocessingError(FraudDetectionError):
    """Raised when feature preprocessing fails."""


class TrainingError(FraudDetectionError):
    """Raised when model training fails."""


class PredictionError(FraudDetectionError):
    """Raised when inference fails."""


class ModelNotFoundError(FraudDetectionError):
    """Raised when a requested model artifact does not exist."""


class ExplainabilityError(FraudDetectionError):
    """Raised when SHAP explanation generation fails."""