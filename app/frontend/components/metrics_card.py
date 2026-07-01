"""Renders the risk level badge and threshold reference table."""

import streamlit as st

_RISK_CONFIG = {
    "LOW":    {"color": "green",  "icon": "🟢", "message": "Low risk. Transaction appears legitimate."},
    "MEDIUM": {"color": "orange", "icon": "🟡", "message": "Medium risk. Manual review recommended."},
    "HIGH":   {"color": "red",    "icon": "🔴", "message": "High risk. Flag for immediate review."},
}


def render_metrics_card(result: dict) -> None:
    risk_level = result["risk_level"]
    probability = result["probability"]
    config = _RISK_CONFIG.get(risk_level, _RISK_CONFIG["MEDIUM"])

    st.markdown("### Risk Assessment")

    st.markdown(
        f"**Risk Level:** {config['icon']} "
        f":{config['color']}[**{risk_level}**]"
    )
    st.caption(config["message"])

    with st.expander("Risk level thresholds"):
        st.markdown("""
        | Risk Level | Fraud Probability | Action |
        |---|---|---|
        | 🟢 LOW | < 30% | Allow transaction |
        | 🟡 MEDIUM | 30% - 70% | Manual review |
        | 🔴 HIGH | > 70% | Block & investigate |
        """)
        st.caption(f"Current probability: **{probability * 100:.2f}%**")