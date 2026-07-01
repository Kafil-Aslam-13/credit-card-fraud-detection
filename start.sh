#!/bin/bash
set -e

echo "Starting system..."

mkdir -p data/raw artifacts/models /root/.kaggle

if [ ! -f "data/raw/creditcard.csv" ]; then
    echo "Downloading dataset..."
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > /root/.kaggle/kaggle.json
    chmod 600 /root/.kaggle/kaggle.json

    kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
fi

if [ ! -f "artifacts/models/fraud_model.joblib" ]; then
    echo "Training model..."
    python train.py
fi

echo "Starting API..."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit..."
streamlit run app/frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0