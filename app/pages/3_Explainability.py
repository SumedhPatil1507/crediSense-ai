import streamlit as st
import joblib
import warnings
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.feature_engineering import create_features
from src.explainability import get_explainer_and_values
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.fairness import subgroup_metrics, disparate_impact_ratio

model = joblib.load(BASE_DIR / "models/model.pkl")
pre   = model.named_steps["preprocessor"]
mod   = model.named_steps["model"]
expected_cols = list(pre.feature_names_in_)


@st.cache_data
def load_sample():
    df = load_data(str(BASE_DIR / "data" / "loan_cleaned.csv"))
    df = clean_data(df)
    df = create_features(df)
    df_feat = df.drop(columns=["Risk_Flag"], errors="ignore")
    if "Id" not in df_feat.columns:
        df_feat["Id"] = 0
    for col in expected_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0
    return df_feat[expected_cols].head(200), df


@st.cache_data
def load_fairness_data():
    from sklearn.model_selection import train_test_split
    df = load_data(str(BASE_DIR / "data" / "loan_cleaned.csv"))
    df = clean_data(df)
    df = create_features(df)
    _, df_test = train_test_split(df, test_size=0.1, stratify=df["Risk_Flag"], random_state=42)
    return df_test.reset_index(drop=True)


st.set_page_config(layout="wide")
st.title("Model Explainability & Fairness")
st.caption("SHAP global + local explanations | Feature interactions | Fairness audit")

try:
    sample, df_full = load_sample()
    explainer, sv, X_dense, feature_names = get_explainer_and_values(model, sample)

    tabs = st.tabs([
        "Global Importance",
        "Single Prediction",
        "Feature Dependence",
        "Feature Interactions",
        "Fairness Audit",
    ])

    # ── TAB 1: SHAP Summary ────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Global Feature Importance (SHAP Beeswarm)")
        st.caption("Each dot = one applicant. Color = feature value. X-axis = impact on risk score.")

        col_l, col_r = st.columns([2, 1])
        with col_l:
            shap.summary_plot(sv, X_dense, feature_names=feature_names, show=False,
                              plot_size=(10, 6))
            st.pyplot(plt.gcf())
            plt.clf()

        with col_r:
            # Bar chart of mean |SHAP| — top 15
            mean_shap = np.abs(sv).mean(axis=0)
            top_idx   = np.argsort(mean_shap)[::-1][:15]
            df_imp = pd.DataFrame({
                "Feature": [feature_names[i].replace("num__","").replace("cat__","") for i in top_idx],
                "Mean |SHAP|": mean_shap[top_idx]
            })
            fig_bar = px.bar(df_imp, x="Mean |SHAP|", y="Feature", orientation="h",
                              color="Mean |SHAP|", color_continuous_scale="Blues",
                              title="Top 15 Features by Mean Impact")
            fig_bar.update_layout(height=500, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_bar, use_container_width=True)

        st.caption("SHAP: Lundberg & Lee, NeurIPS 2017 — https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html")

    # ── TAB 2: Single Prediction (Waterfall) ──────────────────────────────────
    with tabs[1]:
        st.subheader("Single Prediction Explanation")
        st.caption("Why did the model give this specific applicant their risk score?")

        idx = st.slider("Select applicant index", 0, len(sample) - 1, 0, key="shap_idx")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob = float(model.predict_proba(sample.iloc[[idx]])[0][1])

        k1, k2, k3 = st.columns(3)
        k1.metric("Risk Probability", f"{prob:.2%}")
        k2.metric("Decision", "Approve" if prob < 0.3 else "Review" if prob < 0.6 else "Reject")
        k3.metric("Base Rate", f"{float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value):.2%}")

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1])

        explanation = shap.Explanation(
            values=sv[idx],
            base_values=base_val,
            data=X_dense[idx],
            feature_names=feature_names
        )

        col_wf, col_tbl = st.columns([3, 2])
        with col_wf:
            shap.plots.waterfall(explanation, show=False, max_display=15)
            st.pyplot(plt.gcf())
            plt.clf()

        with col_tbl:
            st.subheader("Top Feature Contributions")
            shap_df = pd.DataFrame({
                "Feature": [f.replace("num__","").replace("cat__","") for f in feature_names],
                "SHAP Value": sv[idx],
                "Direction": ["Increases Risk" if v > 0 else "Decreases Risk" for v in sv[idx]]
            }).sort_values("SHAP Value", key=abs, ascending=False).head(12)
            st.dataframe(
                shap_df.style.format({"SHAP Value": "{:+.4f}"})
                .applymap(lambda v: "color: red" if v == "Increases Risk" else "color: green",
                          subset=["Direction"]),
                use_container_width=True, hide_index=True
            )

    # ── TAB 3: Feature Dependence ──────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Feature Dependence Plot")
        st.caption("How does a feature's value affect its SHAP contribution across all applicants?")

        num_features   = [f for f in feature_names if f.startswith("num__")]
        display_names  = [f.replace("num__", "") for f in num_features]

        d1, d2 = st.columns(2)
        with d1:
            selected = st.selectbox("Primary feature", display_names, key="dep_feat")
        with d2:
            color_feat = st.selectbox("Color by feature", ["Auto"] + display_names, key="dep_color")

        feat_idx = feature_names.index(f"num__{selected}")
        color_idx = "auto" if color_feat == "Auto" else feature_names.index(f"num__{color_feat}")

        fig_dep, ax_dep = plt.subplots(figsize=(10, 5))
        shap.dependence_plot(feat_idx, sv, X_dense, feature_names=feature_names,
                             interaction_index=color_idx, ax=ax_dep, show=False)
        st.pyplot(fig_dep)
        plt.clf()

    # ── TAB 4: Feature Interactions ────────────────────────────────────────────
    with tabs[3]:
        st.subheader("SHAP Feature Interaction Heatmap")
        st.caption("Correlation between SHAP values — which features move together in their impact on risk.")

        # Compute correlation of SHAP values across samples
        shap_df_full = pd.DataFrame(sv, columns=feature_names)
        # Keep only numeric features for readability
        num_cols_shap = [c for c in shap_df_full.columns if c.startswith("num__")]
        shap_corr = shap_df_full[num_cols_shap].corr()
        shap_corr.columns = [c.replace("num__", "") for c in shap_corr.columns]
        shap_corr.index   = [c.replace("num__", "") for c in shap_corr.index]

        fig_heat = px.imshow(shap_corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="SHAP Value Correlation (Numeric Features)",
                              aspect="auto")
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.caption("High positive correlation = features tend to increase/decrease risk together. "
                   "Useful for detecting redundant features and multicollinearity.")

        # SHAP distribution by decision
        st.markdown("---")
        st.subheader("SHAP Distribution by Decision")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            all_probs = model.predict_proba(sample)[:, 1]
        decisions = ["Approve" if p < 0.3 else "Review" if p < 0.6 else "Reject" for p in all_probs]

        sel_feat_int = st.selectbox("Feature to compare", display_names, key="int_feat")
        feat_idx_int = feature_names.index(f"num__{sel_feat_int}")
        df_dist = pd.DataFrame({
            "SHAP Value": sv[:, feat_idx_int],
            "Decision": decisions
        })
        fig_dist = px.violin(df_dist, x="Decision", y="SHAP Value", color="Decision",
                              color_discrete_map={"Approve": "green", "Review": "orange", "Reject": "red"},
                              box=True, title=f"SHAP Distribution of {sel_feat_int} by Decision")
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── TAB 5: Fairness Audit ──────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Fairness & Bias Audit")
        st.caption("Model performance across demographic subgroups. Disparate Impact Ratio < 0.8 = potential bias (4/5ths rule).")

        with st.spinner("Computing fairness metrics on holdout set..."):
            df_fair = load_fairness_data()
            metrics_df = subgroup_metrics(model, df_fair)

        if not metrics_df.empty:
            f1, f2 = st.columns(2)
            with f1:
                st.subheader("Subgroup Performance")
                st.dataframe(
                    metrics_df.style.format({
                        "AUC": "{:.4f}", "F1": "{:.4f}",
                        "approval_rate": "{:.2%}", "default_rate": "{:.2%}"
                    }),
                    use_container_width=True, hide_index=True
                )

            with f2:
                fig_fair = px.bar(metrics_df, x="group_val", y="approval_rate",
                                   color="group_col", barmode="group",
                                   title="Approval Rate by Subgroup",
                                   labels={"approval_rate": "Approval Rate", "group_val": "Group"})
                fig_fair.add_hline(y=0.8, line_dash="dash", line_color="red",
                                    annotation_text="80% fairness threshold")
                st.plotly_chart(fig_fair, use_container_width=True)

            st.markdown("---")
            st.subheader("Disparate Impact Ratio (4/5ths Rule)")
            dir_df = disparate_impact_ratio(metrics_df)
            if not dir_df.empty:
                def highlight_fair(val):
                    return "background-color: #d4edda" if val == "Yes" else "background-color: #f8d7da"
                st.dataframe(
                    dir_df.style.applymap(highlight_fair, subset=["fair"]),
                    use_container_width=True, hide_index=True
                )

            # AUC gap across groups
            st.markdown("---")
            st.subheader("AUC Gap Across Groups")
            fig_auc = px.bar(metrics_df, x="group_val", y="AUC", color="group_col",
                              barmode="group", title="ROC-AUC by Subgroup",
                              labels={"AUC": "ROC-AUC", "group_val": "Group"})
            fig_auc.add_hline(y=0.75, line_dash="dash", line_color="orange",
                               annotation_text="Minimum acceptable AUC")
            st.plotly_chart(fig_auc, use_container_width=True)

            st.caption("Ref: ECOA (Equal Credit Opportunity Act) | Fair Housing Act | "
                       "4/5ths Rule: EEOC Uniform Guidelines on Employee Selection Procedures")
        else:
            st.warning("Could not compute fairness metrics.")

except Exception as e:
    st.error(f"Error loading explainability: {e}")
