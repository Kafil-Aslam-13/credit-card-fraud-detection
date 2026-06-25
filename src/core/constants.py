""" Projct Wide Constants. No bussiness logic here , just fixed values """


PROJECT_NAME = "credit_card_fraud_detection"

TARGET_COLUMN = "Class"
FRAUD_LABEL = 1
LEGIT_LABEL = 0


MODEL_FILE_NAME = "fraud_model.joblib"
PREPROCESSOR_FILE_NAME = "scaler.joblib"

RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"


# Prediction labels returned by the API
PREDICTION_FRAUD = "FRAUD"
PREDICTION_LEGIT = "LEGITIMATE"