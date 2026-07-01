"""Renders the SHAP explanation card."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def render_shap_card(result: dict) -> None:
    st.markdown("### Why this prediction?")
    st.caption(
        "The chart below shows the top 5 features that influenced "
        "this prediction. Positive values push toward **FRAUD**, "
        "negative values push toward **LEGITIMATE**."
    )

    top_features = result.get("top_features", [])

    if not top_features:
        st.info("No explanation available.")
        return

    df = pd.DataFrame(top_features)
    colors = ["#ff4b4b" if v > 0 else "#21c354" for v in df["impact"]]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(df["feature"], df["impact"], color=colors)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SHAP Impact")
    ax.set_title("Feature Contributions")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("View detailed SHAP values"):
        display_df = df.copy()
        display_df["direction"] = display_df["impact"].apply(
            lambda x: "-> FRAUD" if x > 0 else "-> LEGIT"
        )
        display_df["impact"] = display_df["impact"].apply(
            lambda x: f"{x:+.6f}"
        )
        st.dataframe(display_df, use_container_width=True)

    st.caption(
        "SHAP (SHapley Additive exPlanations) assigns each feature "
        "a contribution score for this specific prediction."
    )