import streamlit as st
import joblib
import warnings
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.feature_engineering import create_features
from src.explainability import get_explainer_and_values
from src.data_loader import load_data
from src.preprocessing import clean_data

model = joblib.load(BASE_DIR / "models/model.pkl")
pre = model.named_steps['preprocessor']
mod = model.named_steps['model']
expected_cols = list(pre.feature_names_in_)

@st.cache_data
def load_sample():
    df = load_data(str(BASE_DIR / "data" / "loan_cleaned.csv"))
    df = clean_data(df)
    df = create_features(df)
    if "Risk_Flag" in df.columns:
        df = df.drop(columns=["Risk_Flag"])
    if "Id" not in df.columns:
        df["Id"] = 0
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols].head(200)

st.set_page_config(layout="wide")
st.title("🧠 Model Explainability")
st.caption("SHAP + LIME explanations — auto-loaded from repo.")

try:
    sample = load_sample()
    explainer, sv, X_dense, feature_names = get_explainer_and_values(model, sample)

    tabs = st.tabs(["📊 SHAP Summary", "🔍 SHAP Waterfall", "📈 SHAP Dependence", "🍋 LIME", "⚖️ Fairness"])

    # ── TAB 1: SHAP Summary ────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Global Feature Importance (SHAP)")
        st.caption("Which features drive default risk most across all applicants.")
        shap.summary_plot(sv, X_dense, feature_names=feature_names, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
        st.caption("📚 [SHAP paper — Lundberg & Lee, NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)")

    # ── TAB 2: SHAP Waterfall ──────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Single Prediction Explanation (SHAP Waterfall)")
        st.caption("Why did the model give this specific applicant their risk score?")

        idx = st.slider("Select applicant", 0, len(sample) - 1, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = model.predict_proba(sample.iloc[[idx]])[0][1]

        st.metric("Risk Probability", f"{prob:.2%}")

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1]

        explanation = shap.Explanation(
            values=sv[idx],
            base_values=base_val,
            data=X_dense[idx],
            feature_names=feature_names
        )
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

        # Top SHAP drivers as table
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": sv[idx],
            "Input Value": X_dense[idx]
        }).sort_values("SHAP Value", key=abs, ascending=False).head(10)
        st.subheader("Top 10 Feature Contributions")
        st.dataframe(shap_df.style.format({"SHAP Value": "{:.4f}", "Input Value": "{:.4f}"}),
                     use_container_width=True)

    # ── TAB 3: SHAP Dependence ─────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Feature Dependence Plot")
        st.caption("How does a feature's value affect its SHAP contribution?")

        num_features = [f for f in feature_names if f.startswith("num__")]
        display_names = [f.replace("num__", "") for f in num_features]
        selected = st.selectbox("Select feature", display_names)
        feat_idx = feature_names.index(f"num__{selected}")

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        shap.dependence_plot(feat_idx, sv, X_dense, feature_names=feature_names,
                             ax=ax3, show=False)
        st.pyplot(fig3)
        plt.clf()

    # ── TAB 4: LIME ────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("LIME — Local Interpretable Model-Agnostic Explanations")
        st.caption("LIME approximates the model locally with a simple linear model to explain one prediction.")

        try:
            import lime
            import lime.lime_tabular

            lime_idx = st.slider("Select applicant for LIME", 0, len(sample) - 1, 0, key="lime_idx")

            @st.cache_resource
            def get_lime_explainer():
                return lime.lime_tabular.LimeTabularExplainer(
                    X_dense,
                    feature_names=feature_names,
                    class_names=["Safe", "Default"],
                    mode="classification",
                    random_state=42
                )

            lime_exp_obj = get_lime_explainer()

            with st.spinner("Running LIME..."):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Wrap predict_fn to give LightGBM proper feature names
                    def predict_fn(x):
                        df_x = pd.DataFrame(x, columns=feature_names)
                        return mod.predict_proba(df_x)
                    lime_result = lime_exp_obj.explain_instance(
                        X_dense[lime_idx], predict_fn, num_features=10, top_labels=1
                    )

            lime_list = lime_result.as_list(label=1)
            df_lime = pd.DataFrame(lime_list, columns=["Feature Condition", "Weight"])
            df_lime["Direction"] = df_lime["Weight"].apply(
                lambda x: "Increases Risk" if x > 0 else "Decreases Risk")

            fig_lime = px.bar(df_lime, x="Weight", y="Feature Condition",
                              orientation="h", color="Direction",
                              color_discrete_map={"Increases Risk": "red",
                                                  "Decreases Risk": "green"},
                              title="LIME Feature Weights for Selected Applicant")
            st.plotly_chart(fig_lime, use_container_width=True)
            st.dataframe(df_lime, use_container_width=True)
            st.caption("LIME paper: Ribeiro et al., KDD 2016 — https://arxiv.org/abs/1602.04938")

        except ImportError:
            st.info("Install `lime` to enable LIME explanations.")
        except Exception as e:
            st.error(f"LIME error: {e}")

    # ── TAB 5: Fairness ────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Fairness & Bias Analysis")
        st.caption("Model performance across demographic subgroups. Disparate Impact Ratio < 0.8 = potential bias.")

        try:
            from src.fairness import subgroup_metrics, disparate_impact_ratio
            from src.data_loader import load_data
            from src.preprocessing import clean_data
            from src.feature_engineering import create_features
            from sklearn.model_selection import train_test_split

            @st.cache_data
            def load_fairness_data():
                df = load_data(str(BASE_DIR / "data" / "loan_cleaned.csv"))
                df = clean_data(df)
                df = create_features(df)
                _, df_test = train_test_split(df, test_size=0.1, stratify=df["Risk_Flag"], random_state=42)
                return df_test.reset_index(drop=True)

            with st.spinner("Computing fairness metrics..."):
                df_fair = load_fairness_data()
                metrics_df = subgroup_metrics(model, df_fair)

            if not metrics_df.empty:
                st.subheader("Subgroup Performance")
                st.dataframe(
                    metrics_df.style.format({"AUC": "{:.4f}", "F1": "{:.4f}",
                                              "approval_rate": "{:.2%}", "default_rate": "{:.2%}"}),
                    use_container_width=True
                )

                fig_fair = px.bar(metrics_df, x="group_val", y="approval_rate",
                                   color="group_col", barmode="group",
                                   title="Approval Rate by Subgroup",
                                   labels={"approval_rate": "Approval Rate", "group_val": "Group"})
                fig_fair.add_hline(y=0.8, line_dash="dash", line_color="red",
                                    annotation_text="80% threshold")
                st.plotly_chart(fig_fair, use_container_width=True)

                st.subheader("Disparate Impact Ratio (4/5ths Rule)")
                dir_df = disparate_impact_ratio(metrics_df)
                if not dir_df.empty:
                    st.dataframe(
                        dir_df.style.applymap(
                            lambda v: "background-color: #f8d7da" if v == "No" else "background-color: #d4edda",
                            subset=["fair"]
                        ),
                        use_container_width=True
                    )
                st.caption("Disparate Impact Ratio < 0.8 may indicate discriminatory outcomes. "
                           "Ref: ECOA, Fair Housing Act.")
        except Exception as e:
            st.error(f"Fairness analysis error: {e}")

except Exception as e:
    st.error(f"Error loading explainability: {e}")
