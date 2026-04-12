import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.feature_engineering import create_features
from src.explainability import get_explainer_and_values

model = joblib.load(BASE_DIR / "models/model.pkl")
pre = model.named_steps['preprocessor']
mod = model.named_steps['model']
expected_cols = list(pre.feature_names_in_)

st.set_page_config(layout="wide")
st.title("🧠 Model Explainability")
st.caption("Upload the dataset to explore SHAP-based feature explanations.")

file = st.file_uploader("Upload CSV (loan_cleaned.csv or sample_input.csv)")

if file:
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        if "Risk_Flag" in df.columns:
            df = df.drop(columns=["Risk_Flag"])

        df = create_features(df)

        if "Id" not in df.columns:
            df["Id"] = 0

        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

        sample = df.head(100)

        explainer, sv, X_dense, feature_names = get_explainer_and_values(model, sample)

        tabs = st.tabs(["📊 Summary Plot", "🔍 Waterfall (Single)", "📈 Dependence Plot"])

        # ── TAB 1: Summary Plot ────────────────────────────────────────────────
        with tabs[0]:
            st.subheader("Global Feature Importance (SHAP)")
            st.caption("Shows which features drive risk predictions most across all samples.")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(sv, X_dense, feature_names=feature_names, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

        # ── TAB 2: Waterfall for single prediction ─────────────────────────────
        with tabs[1]:
            st.subheader("Single Prediction Explanation")
            st.caption("Explains exactly why the model gave a specific risk score to one applicant.")

            n = len(sample)
            idx = st.slider("Select applicant index", 0, n - 1, 0)

            prob = model.predict_proba(sample.iloc[[idx]])[0][1]
            st.metric("Risk Probability", f"{prob:.2%}")

            # Build SHAP Explanation object for waterfall
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1]

            shap_vals_single = sv[idx]

            explanation = shap.Explanation(
                values=shap_vals_single,
                base_values=base_val,
                data=X_dense[idx],
                feature_names=feature_names
            )

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

        # ── TAB 3: Dependence Plot ─────────────────────────────────────────────
        with tabs[2]:
            st.subheader("Feature Dependence Plot")
            st.caption("Shows how a feature's value affects its SHAP contribution across all samples.")

            # Only show numeric/meaningful features
            num_features = [f for f in feature_names if f.startswith("num__")]
            display_names = [f.replace("num__", "") for f in num_features]

            selected = st.selectbox("Select feature", display_names)
            feat_full = f"num__{selected}"
            feat_idx = feature_names.index(feat_full)

            fig3, ax3 = plt.subplots(figsize=(10, 5))
            shap.dependence_plot(
                feat_idx, sv, X_dense,
                feature_names=feature_names,
                ax=ax3, show=False
            )
            st.pyplot(fig3)
            plt.clf()

    except Exception as e:
        st.error(f"Error: {e}")
