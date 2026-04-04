import streamlit as st
import joblib
import json
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from utils import build_full_input
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.feature_engineering import create_features
from src.evaluate import evaluate, threshold_analysis

model = joblib.load(BASE_DIR / "models/model.pkl")
with open(BASE_DIR / "models/columns.json") as f:
    cols = json.load(f)

st.set_page_config(layout="wide")
st.title("💳 Loan Risk Prediction")

tabs = st.tabs(["🔮 Predict", "📈 Model Performance", "⚙️ Threshold Analysis"])

# ── TAB 1: Prediction ──────────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("Income (normalized)", 0.0, 1.0, 0.5)
        age = st.slider("Age (normalized)", 0.0, 1.0, 0.3)
    with col2:
        exp = st.slider("Experience (normalized)", 0.0, 1.0, 0.2)

    if st.button("🚀 Predict"):
        user_input = {"Income": income, "Age": age, "Experience": exp}
        df_input = build_full_input(user_input, cols)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(df_input)[0][1]

        st.metric("Risk Probability", f"{prob:.2%}")

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if prob > 0.5 else "green"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "value": 50}
            },
            title={"text": "Risk Score"}
        ))
        st.plotly_chart(gauge, use_container_width=True)

        decision = "✅ Approve" if prob < 0.3 else "🔍 Manual Review" if prob < 0.6 else "❌ Reject"
        if prob < 0.3:
            st.success(f"Decision: {decision}")
        elif prob < 0.6:
            st.warning(f"Decision: {decision}")
        else:
            st.error(f"Decision: {decision}")

# ── TAB 2: Model Performance ───────────────────────────────────────────────────
with tabs[1]:
    st.info("Upload the cleaned dataset to compute live metrics.")
    perf_file = st.file_uploader("Upload loan_cleaned.csv", key="perf")

    if perf_file:
        with st.spinner("Evaluating model..."):
            try:
                from sklearn.model_selection import train_test_split

                df_eval = load_data(perf_file)
                df_eval = clean_data(df_eval)
                df_eval = create_features(df_eval)

                X_eval = df_eval.drop(columns=["Risk_Flag"])
                y_eval = df_eval["Risk_Flag"]

                _, X_test, _, y_test = train_test_split(
                    X_eval, y_eval, test_size=0.2, stratify=y_eval, random_state=42
                )

                metrics = evaluate(model, X_test, y_test)

                # KPIs
                k1, k2 = st.columns(2)
                k1.metric("ROC-AUC", metrics["AUC"])
                k2.metric("F1 Score", metrics["F1"])

                st.markdown("---")

                # Classification report
                st.subheader("Classification Report")
                report = metrics["report"]
                report_df = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

                st.markdown("---")
                col_roc, col_cm = st.columns(2)

                # ROC Curve
                with col_roc:
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = metrics["roc_curve"]
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={metrics['AUC']}",
                                                  line=dict(color="royalblue", width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                                  line=dict(dash="dash", color="gray")))
                    fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=350)
                    st.plotly_chart(fig_roc, use_container_width=True)

                # Confusion Matrix
                with col_cm:
                    st.subheader("Confusion Matrix")
                    cm = np.array(metrics["confusion_matrix"])
                    fig_cm = px.imshow(
                        cm, text_auto=True,
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Safe (0)", "Risk (1)"],
                        y=["Safe (0)", "Risk (1)"],
                        color_continuous_scale="Blues"
                    )
                    fig_cm.update_layout(height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB 3: Threshold Analysis ──────────────────────────────────────────────────
with tabs[2]:
    st.info("Upload the cleaned dataset to explore threshold tradeoffs.")
    thresh_file = st.file_uploader("Upload loan_cleaned.csv", key="thresh")

    if thresh_file:
        with st.spinner("Running threshold analysis..."):
            try:
                from sklearn.model_selection import train_test_split

                df_t = load_data(thresh_file)
                df_t = clean_data(df_t)
                df_t = create_features(df_t)

                X_t = df_t.drop(columns=["Risk_Flag"])
                y_t = df_t["Risk_Flag"]

                _, X_test_t, _, y_test_t = train_test_split(
                    X_t, y_t, test_size=0.2, stratify=y_t, random_state=42
                )

                y_prob_t = model.predict_proba(X_test_t)[:, 1]
                results = threshold_analysis(y_test_t, y_prob_t)
                df_thresh = pd.DataFrame(results)

                st.subheader("Threshold vs Metrics")
                fig_thresh = go.Figure()
                for col_name, color in [("precision", "blue"), ("recall", "red"),
                                         ("f1", "green"), ("approval_rate", "orange")]:
                    fig_thresh.add_trace(go.Scatter(
                        x=df_thresh["threshold"], y=df_thresh[col_name],
                        name=col_name.capitalize(), line=dict(color=color)
                    ))
                fig_thresh.update_layout(xaxis_title="Threshold", yaxis_title="Score", height=400)
                st.plotly_chart(fig_thresh, use_container_width=True)

                st.subheader("Threshold Table")
                st.dataframe(df_thresh.style.format("{:.3f}"), use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
