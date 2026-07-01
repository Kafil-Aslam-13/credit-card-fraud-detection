#!/bin/bash

echo "Starting Credit Card Fraud Detection System..."

# Download dataset if not present
if [ ! -f "data/raw/creditcard.csv" ]; then
    echo "Downloading dataset from Kaggle..."
    mkdir -p data/raw
    mkdir -p /root/.kaggle

    # Write kaggle.json from environment variables
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" \
        > /root/.kaggle/kaggle.json
    chmod 600 /root/.kaggle/kaggle.json

    # Download dataset
    kaggle datasets download \
        -d mlg-ulb/creditcardfraud \
        -p data/raw \
        --unzip

    echo "Dataset downloaded."
else
    echo "Dataset found. Skipping download."
fi

# Train model if not present
if [ ! -f "artifacts/models/fraud_model.joblib" ]; then
    echo "Training model..."
    python train.py
    python train.py

if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo "Training complete."
else
    echo "Model found. Skipping training."
fi

# Start FastAPI on port 8000 (internal)
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
echo "API starting on port 8000..."
sleep 5

# Start Streamlit on port 8501 (exposed)
echo "Starting Streamlit on port 8501..."
streamlit run app/frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true