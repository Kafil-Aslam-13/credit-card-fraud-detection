"""Renders the prediction result card."""

import streamlit as st


def render_prediction_card(result: dict) -> None:
    prediction = result["prediction"]
    probability = result["probability"]

    st.markdown("### Prediction Result")

    if prediction == "FRAUD":
        st.error("**FRAUD DETECTED**", icon="🚨")
    else:
        st.success("**LEGITIMATE TRANSACTION**", icon="✅")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Fraud Probability",
            value=f"{probability * 100:.2f}%",
        )
    with col2:
        st.metric(
            label="Confidence",
            value=f"{(1 - probability) * 100:.2f}% Legit"
            if prediction == "LEGITIMATE"
            else f"{probability * 100:.2f}% Fraud",
        )

    st.progress(probability, text=f"Fraud probability: {probability:.4f}")