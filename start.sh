```bash

set -euo pipefail


echo "======================================"
echo "Starting Credit Card Fraud Detection System..."
echo "======================================"

# Create required directories
mkdir -p data/raw
mkdir -p artifacts/models
mkdir -p /root/.kaggle

############################################
# Download dataset if it doesn't exist
############################################
if [ ! -f "data/raw/creditcard.csv" ]; then
    echo "Dataset not found."

    if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
        echo "ERROR: KAGGLE_USERNAME or KAGGLE_KEY environment variables are not set."
        exit 1
    fi

    echo "Configuring Kaggle credentials..."

    cat > /root/.kaggle/kaggle.json <<EOF
{
  "username": "${KAGGLE_USERNAME}",
  "key": "${KAGGLE_KEY}"
}
EOF

    chmod 600 /root/.kaggle/kaggle.json

    echo "Downloading dataset from Kaggle..."

    kaggle datasets download \
        -d mlg-ulb/creditcardfraud \
        -p data/raw \
        --unzip

    if [ ! -f "data/raw/creditcard.csv" ]; then
        echo "ERROR: Dataset download completed but creditcard.csv was not found."
        exit 1
    fi

    echo "Dataset downloaded successfully."
else
    echo "Dataset already exists."
fi

############################################
# Train model if necessary
############################################
if [ ! -f "artifacts/models/fraud_model.joblib" ]; then
    echo "Model not found."
    echo "Starting training..."

    python train.py

    if [ ! -f "artifacts/models/fraud_model.joblib" ]; then
        echo "ERROR: Training finished but model file was not created."
        exit 1
    fi

    echo "Model trained successfully."
else
    echo "Existing model found."
fi

############################################
# Verify files
############################################
echo
echo "Checking required files..."
ls -lh artifacts/models
ls -lh data/raw

############################################
# Start FastAPI
############################################
echo
echo "Starting FastAPI..."

uvicorn app.api.main:app \
    --host 0.0.0.0 \
    --port 8000 &

sleep 5

############################################
# Start Streamlit
############################################
echo
echo "Starting Streamlit..."

exec streamlit run app/frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true
```
