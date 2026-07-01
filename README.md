---
title: Credit Card Fraud Detection
emoji: 💳
colorFrom: red
colorTo: orange
sdk: docker
app_port: 8501
pinned: true
---

# Credit Card Fraud Detection System

Production-grade end-to-end ML system that detects fraudulent
credit card transactions and explains why using SHAP.

## Tech Stack

| Concern | Tool |
|---|---|
| Model | XGBoost |
| Imbalanced data | SMOTE |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI |
| Frontend | Streamlit |
| Deployment | HuggingFace Spaces |
| CI/CD | GitHub Actions |

## Run Locally

```bash
conda create -n fraud-detection python=3.11 -y
conda activate fraud-detection
pip install -r requirements.txt
python train.py
uvicorn app.api.main:app --reload   # terminal 1
streamlit run app/frontend/streamlit_app.py   # terminal 2
```

## Dataset

[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
284,807 transactions · 492 fraud cases · 30 features

## Author

**Kafil Aslam**
[GitHub](https://github.com/Kafil-Aslam-13) |
[HuggingFace](https://huggingface.co/KAFIL-ASLAM)