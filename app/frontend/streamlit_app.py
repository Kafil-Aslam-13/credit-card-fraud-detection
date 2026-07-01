"""Streamlit frontend for Credit Card Fraud Detection."""

import os

import matplotlib.pyplot as plt
import requests
import streamlit as st
from dotenv import load_dotenv

from app.frontend.components.metrics_card import render_metrics_card
from app.frontend.components.prediction_card import render_prediction_card
from app.frontend.components.shap_card import render_shap_card

load_dotenv()
# Both are in same container so API is just localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

FEATURE_DESCRIPTIONS = {
    "Time":   "Time elapsed since first transaction (seconds)",
    "V1":    "PCA: Merchant category / transaction type signal",
    "V2":    "PCA: Transaction frequency pattern",
    "V3":    "PCA: Time of day / hour pattern",
    "V4":    "PCA: Card usage pattern",
    "V5":    "PCA: Cardholder behavior signal",
    "V6":    "PCA: Location / geography signal",
    "V7":    "PCA: Online vs in-store signal",
    "V8":    "PCA: Transaction velocity signal",
    "V9":    "PCA: Card present / not present signal",
    "V10":   "PCA: Foreign transaction signal",
    "V11":   "PCA: Spending pattern deviation",
    "V12":   "PCA: High-risk merchant signal",
    "V13":   "PCA: Transaction amount pattern",
    "V14":   "PCA: Strongest fraud indicator",
    "V15":   "PCA: Weekend / weekday pattern",
    "V16":   "PCA: Repeated transaction signal",
    "V17":   "PCA: Account age signal",
    "V18":   "PCA: Multi-card usage pattern",
    "V19":   "PCA: Night transaction signal",
    "V20":   "PCA: Cash withdrawal pattern",
    "V21":   "PCA: Contactless payment signal",
    "V22":   "PCA: Cross-border signal",
    "V23":   "PCA: Decline history signal",
    "V24":   "PCA: Chip vs swipe signal",
    "V25":   "PCA: Unusual location signal",
    "V26":   "PCA: Low-value fraud signal",
    "V27":   "PCA: Device fingerprint signal",
    "V28":   "PCA: Session behavior signal",
    "Amount": "Transaction amount in dollars",
}

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💳 Fraud Detection")
    st.markdown("---")
    st.markdown("**Model:** XGBoost")
    st.markdown("**Dataset:** ULB Credit Card Fraud")
    st.markdown("**Features:** 30 (Time, V1-V28, Amount)")
    st.markdown("---")
    try:
        health = requests.get(f"{BACKEND_URL}/api/v1/health", timeout=3)
        if health.status_code == 200:
            st.success("API: Online", icon="✅")
        else:
            st.error("API: Error", icon="❌")
    except Exception:
        st.error("API: Offline", icon="❌")
        st.caption(f"Expected at: {BACKEND_URL}")
    st.markdown("---")
    st.markdown("**Built with:**")
    st.markdown("XGBoost · SHAP · MLflow · FastAPI · Streamlit")
    st.markdown("---")
    st.markdown(
    "[GitHub](https://github.com/Kafil-Aslam-13/credit-card-fraud-detection) "
    "| [HuggingFace](https://huggingface.co/KAFIL-ASLAM)"
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    "Enter a transaction's features to check whether it's "
    "**fraudulent** or **legitimate**, with a SHAP explanation of why."
)
st.markdown("---")

# ── Sample transactions ───────────────────────────────────────────────────────
SAMPLE_FRAUD = {
    "Time": 406.0,
    "V1": -2.3122265423263, "V2": 1.95199201064158,
    "V3": -1.60985073229769, "V4": 3.9979055875468,
    "V5": -0.522187864667764, "V6": -1.42654531920595,
    "V7": -2.53738730624579, "V8": 1.39165724829804,
    "V9": -2.77008927719433, "V10": -2.77227214465915,
    "V11": 3.20203320709635, "V12": -2.89990738849473,
    "V13": -0.595221881324605, "V14": -4.28925378244217,
    "V15": 0.389724120274487, "V16": -1.14074717980657,
    "V17": -2.83005567450437, "V18": -0.0168224681808257,
    "V19": 0.416955705037907, "V20": 0.126910559061474,
    "V21": 0.517232370861764, "V22": -0.0350493686052974,
    "V23": -0.465211076935195, "V24": 0.320198198514526,
    "V25": 0.0445191674731724, "V26": 0.177839798284401,
    "V27": 0.261145002567677, "V28": -0.143275874698919,
    "Amount": 0.0,
}

SAMPLE_LEGIT = {
    "Time": 0.0,
    "V1": -1.3598071336738, "V2": -0.0727811733098497,
    "V3": 2.53634673796914, "V4": 1.37815522427443,
    "V5": -0.338320769942518, "V6": 0.462387777762292,
    "V7": 0.239598554061257, "V8": 0.0986979012610507,
    "V9": 0.363786969611213, "V10": 0.0907941719789316,
    "V11": -0.551599533260813, "V12": -0.617800855762348,
    "V13": -0.991389847235408, "V14": -0.311169353699879,
    "V15": 1.46817697209427, "V16": -0.470400525259478,
    "V17": 0.207971241929242, "V18": 0.0257905801985591,
    "V19": 0.403992960255733, "V20": 0.251412098239705,
    "V21": -0.018306777944153, "V22": 0.277837575558899,
    "V23": -0.110473910188767, "V24": 0.0669280749146731,
    "V25": 0.128539358273528, "V26": -0.189114843888824,
    "V27": 0.133558376740387, "V28": -0.0210530534538215,
    "Amount": 149.62,
}

# ── Initialise session state ──────────────────────────────────────────────────
if "prefill" not in st.session_state:
    st.session_state["prefill"] = {}

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Manual Input", "Load Sample Data"])

with tab2:
    st.markdown("Load a real transaction from the dataset to demo the system:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Load Fraud Sample", type="primary", use_container_width=True
        ):
            st.session_state["prefill"] = SAMPLE_FRAUD
            st.session_state["prefill_label"] = "fraud"
            st.session_state["input_Time"] = float(SAMPLE_FRAUD["Time"])
            st.session_state["input_Amount"] = float(SAMPLE_FRAUD["Amount"])
            for name in [f"V{i}" for i in range(1, 29)]:
                st.session_state[f"input_{name}"] = float(SAMPLE_FRAUD[name])
            st.rerun()
    with col2:
        if st.button("Load Legitimate Sample", use_container_width=True):
            st.session_state["prefill"] = SAMPLE_LEGIT
            st.session_state["prefill_label"] = "legit"
            st.session_state["input_Time"] = float(SAMPLE_LEGIT["Time"])
            st.session_state["input_Amount"] = float(SAMPLE_LEGIT["Amount"])
            for name in [f"V{i}" for i in range(1, 29)]:
                st.session_state[f"input_{name}"] = float(SAMPLE_LEGIT[name])
            st.rerun()

    label = st.session_state.get("prefill_label", "")
    if label == "fraud":
        st.warning(
            "Fraud sample loaded — switch to Manual Input tab and click Analyse.",
            icon="⚠️",
        )
    elif label == "legit":
        st.info(
            "Legitimate sample loaded — switch to Manual Input tab and click Analyse.",
            icon="ℹ️",
        )

with tab1:
    # Read prefill INSIDE tab1 so it's always fresh from session state
    prefill = st.session_state.get("prefill", {})

    if prefill:
        label = st.session_state.get("prefill_label", "")
        if label == "fraud":
            st.warning("Fraud sample loaded and ready.", icon="⚠️")
        elif label == "legit":
            st.info("Legitimate sample loaded and ready.", icon="ℹ️")

    with st.form("transaction_form"):
        st.markdown("#### Transaction Details")
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input(
                "Time — seconds elapsed since first transaction",
                format="%.4f",
                help=FEATURE_DESCRIPTIONS["Time"],
                key="input_Time"
            )
        with col2:
            amount_val = st.number_input(
                "Amount — transaction value in dollars",
                min_value=0.0,
                format="%.2f",
                help=FEATURE_DESCRIPTIONS["Amount"],
                key="input_Amount"
            )

        st.markdown("#### PCA Features (V1 - V28)")
        st.caption(
            "These features are PCA-anonymized for privacy. "
            "Hover over any field to see what signal it represents."
        )

        v_values = {}
        FEATURE_NAMES = [f"V{i}" for i in range(1, 29)]
        cols = st.columns(4)
        for i, name in enumerate(FEATURE_NAMES):
            with cols[i % 4]:
                v_values[name] = st.number_input(
                    name,
                    value=float(prefill.get(name, 0.0)),
                    format="%.6f",
                    key=f"input_{name}",
                    help=FEATURE_DESCRIPTIONS.get(name, "PCA anonymized feature"),
                )

        submitted = st.form_submit_button(
            "Analyse Transaction",
            type="primary",
            use_container_width=True,
        )

# ── Prediction output ─────────────────────────────────────────────────────────
if submitted:
    payload = {"Time": time_val, "Amount": amount_val, **v_values}

    with st.spinner("Analysing transaction..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/v1/predict",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            with col1:
                render_prediction_card(result)
            with col2:
                render_metrics_card(result)

            st.markdown("---")
            render_shap_card(result)

        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot connect to API at `{BACKEND_URL}`. "
                "Start the backend: `uvicorn app.api.main:app --reload`",
                icon="❌",
            )
        except requests.exceptions.Timeout:
            st.error(
                "Request timed out. Model may still be loading.", icon="⏳"
            )
        except requests.exceptions.HTTPError as e:
            st.error(
                f"API Error {e.response.status_code}: {e.response.text}",
                icon="❌",
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}", icon="❌")