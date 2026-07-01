#!/bin/bash

# Start FastAPI on localhost:8000 (internal only)
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
echo "Waiting for API to start..."
sleep 5

# Start Streamlit on port 8501 (HF exposes this)
streamlit run app/frontend/streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true